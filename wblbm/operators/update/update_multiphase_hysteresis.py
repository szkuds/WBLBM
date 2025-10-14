from functools import partial
import jax
import jax.lax
import jax.numpy as jnp
from jax import jit
import optax

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm.operators.wetting.contact_line_location import ContactLineLocation


class UpdateMultiphaseHysteresis(UpdateMultiphase):
    def __init__(
            self,
            grid: Grid,
            lattice: Lattice,
            tau: float,
            kappa: float,
            interface_width: int,
            rho_l: float,
            rho_v: float,
            bc_config: dict = None,
            force_enabled: bool = False,
            collision_scheme: str = "bgk",
            eos: str = "double-well",
            k_diag=None,
            **kwargs
    ):
        super().__init__(
            grid, lattice, tau, kappa, interface_width, rho_l, rho_v,
            bc_config=bc_config, force_enabled=force_enabled,
            collision_scheme=collision_scheme, eos=eos, k_diag=k_diag, **kwargs
        )

        # Initialize hysteresis parameters from bc_config
        if bc_config and 'hysteresis_params' in bc_config:
            self.hysteresis_params = bc_config['hysteresis_params']
        else:
            raise ValueError("Hysteresis parameters not found in bc_config")

        # Extract hysteresis configuration
        self.ca_advancing = self.hysteresis_params['ca_advancing']
        self.ca_receding = self.hysteresis_params['ca_receding']
        self.rho_mean = (rho_l + rho_v) / 2

        # Initialize contact angle and line operators
        self.contact_angle = ContactAngle(self.rho_mean)
        self.contact_line = ContactLineLocation(self.rho_mean)

        # Optimizer configuration
        self.optimizer = optax.adam(learning_rate=0.01)
        self.max_iterations = 50

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.array, force: jnp.ndarray = None):
        """
        Main update step with hysteresis optimization.

        Args:
            f: Distribution function state
            force: External force

        Returns:
            Updated distribution function after optimization
        """
        # Perform standard multiphase update first
        f_updated = super().__call__(f, force)

        # Extract current state for optimization
        rho_current, u_current, force_total = self.macroscopic(f)

        # Compute contact angle at t+1
        ca_left_tplus1, ca_right_tplus1 = self.contact_angle.compute(
            self.macroscopic(f_updated)[0]
        )

        # Check if optimization is needed and determine type
        optimize_left = self._should_optimize(ca_left_tplus1)
        optimize_right = self._should_optimize(ca_right_tplus1)

        # Use jax.lax.cond to handle optimization branches
        f_optimized = jax.lax.cond(
            optimize_left | optimize_right,
            self._optimize_wetting,
            lambda x: x[0],  # No optimization needed
            (f_updated, f, ca_left_tplus1, ca_right_tplus1)
        )

        return f_optimized

    @partial(jit, static_argnums=(0,))
    def _should_optimize(self, contact_angle: float) -> bool:
        """Check if contact angle is outside hysteresis window."""
        return (contact_angle > self.ca_advancing) | (contact_angle < self.ca_receding)

    @partial(jit, static_argnums=(0,))
    def _optimize_wetting(self, args):
        """Optimize wetting parameters based on contact angle constraints."""
        f_updated, f_prev, ca_left_tplus1, ca_right_tplus1 = args

        # Extract current contact line locations
        rho_prev = self.macroscopic(f_prev)[0]
        rho_updated = self.macroscopic(f_updated)[0]

        ca_left_t, ca_right_t = self.contact_angle.compute(rho_prev)
        cll_left_t, cll_right_t = self.contact_line.compute(rho_prev, ca_left_t, ca_right_t)

        # Determine optimization type for each side
        left_in_window = (ca_left_tplus1 >= self.ca_receding) & (ca_left_tplus1 <= self.ca_advancing)
        right_in_window = (ca_right_tplus1 >= self.ca_receding) & (ca_right_tplus1 <= self.ca_advancing)

        # Optimize left side
        optimized_state_left = jax.lax.cond(
            left_in_window,
            lambda x: self._optimize_contact_line(x, cll_left_t, "left"),
            lambda x: self._optimize_contact_angle(x, ca_left_tplus1, "left"),
            f_updated
        )

        # Optimize right side
        optimized_state_right = jax.lax.cond(
            right_in_window,
            lambda x: self._optimize_contact_line(x, cll_right_t, "right"),
            lambda x: self._optimize_contact_angle(x, ca_right_tplus1, "right"),
            optimized_state_left
        )

        return optimized_state_right

    @partial(jit, static_argnums=(0, 3))
    def _optimize_contact_line(self, f_state, cll_target, side):
        """Optimize for contact line location (hysteresis case)."""
        # Initialize wetting parameters
        wetting_params = self._get_current_wetting_params()

        # Dual path optimization for d_rho and phi
        optimal_params_drho = self._optimize_single_param(
            f_state, wetting_params, cll_target, "d_rho", side, self._cost_function_cll
        )
        optimal_params_phi = self._optimize_single_param(
            f_state, wetting_params, cll_target, "phi", side, self._cost_function_cll
        )

        # Evaluate both paths and choose the best
        cost_drho = self._evaluate_cost(f_state, optimal_params_drho, cll_target, self._cost_function_cll)
        cost_phi = self._evaluate_cost(f_state, optimal_params_phi, cll_target, self._cost_function_cll)

        best_params = jax.lax.cond(
            cost_drho <= cost_phi,
            lambda: optimal_params_drho,
            lambda: optimal_params_phi
        )

        return self._apply_wetting_params_and_update(f_state, best_params)

    @partial(jit, static_argnums=(0, 3))
    def _optimize_contact_angle(self, f_state, ca_target, side):
        """Optimize for contact angle (outside hysteresis window)."""
        # Determine target angle based on which bound was exceeded
        target_angle = jax.lax.cond(
            ca_target > self.ca_advancing,
            lambda: self.ca_advancing,
            lambda: self.ca_receding
        )

        wetting_params = self._get_current_wetting_params()

        # Dual path optimization for d_rho and phi
        optimal_params_drho = self._optimize_single_param(
            f_state, wetting_params, target_angle, "d_rho", side, self._cost_function_ca
        )
        optimal_params_phi = self._optimize_single_param(
            f_state, wetting_params, target_angle, "phi", side, self._cost_function_ca
        )

        # Choose best path
        cost_drho = self._evaluate_cost(f_state, optimal_params_drho, target_angle, self._cost_function_ca)
        cost_phi = self._evaluate_cost(f_state, optimal_params_phi, target_angle, self._cost_function_ca)

        best_params = jax.lax.cond(
            cost_drho <= cost_phi,
            lambda: optimal_params_drho,
            lambda: optimal_params_phi
        )

        return self._apply_wetting_params_and_update(f_state, best_params)

    @partial(jit, static_argnums=(0, 4, 5, 6))
    def _optimize_single_param(self, f_state, initial_params, target, param_name, side, cost_func):
        """Optimize a single wetting parameter using gradient descent."""
        # Initialize optimizer state
        opt_state = self.optimizer.init(initial_params)

        def optimization_step(carry, _):
            opt_state, params = carry

            # Compute cost and gradients
            loss, grads = jax.value_and_grad(
                lambda p: cost_func(f_state, p, target, side)
            )(params)

            # Update parameters
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Clamp parameters to valid ranges
            new_params = self._clamp_params(new_params)

            return (new_opt_state, new_params), loss

        # Run optimization iterations
        (_, final_params), _ = jax.lax.scan(
            optimization_step,
            (opt_state, initial_params),
            jnp.arange(self.max_iterations)
        )

        return final_params

    @partial(jit, static_argnums=(0,))
    def _get_current_wetting_params(self):
        """Extract current wetting parameters from bc_config."""
        wetting_params = self.macroscopic.bc_config.get('wetting_params', {})
        return {
            'd_rho_left': wetting_params.get('d_rho_left', 0.5),
            'd_rho_right': wetting_params.get('d_rho_right', 0.5),
            'phi_left': wetting_params.get('phi_left', 1.5),
            'phi_right': wetting_params.get('phi_right', 1.5),
        }

    @partial(jit, static_argnums=(0,))
    def _clamp_params(self, params):
        """Clamp parameters to valid ranges."""
        return {
            'd_rho_left': jnp.clip(params['d_rho_left'], 1e-6, 0.99),
            'd_rho_right': jnp.clip(params['d_rho_right'], 1e-6, 0.99),
            'phi_left': jnp.maximum(params['phi_left'], 1.0),
            'phi_right': jnp.maximum(params['phi_right'], 1.0),
        }

    @partial(jit, static_argnums=(0, 4))
    def _cost_function_cll(self, f_state, params, cll_target, side):
        """Cost function for contact line location optimization."""
        # Apply parameters and compute new state
        new_state = self._apply_wetting_params_and_update(f_state, params)
        rho_new = self.macroscopic(new_state)[0]

        # Compute new contact angles and line locations
        ca_left, ca_right = self.contact_angle.compute(rho_new)
        cll_left, cll_right = self.contact_line.compute(rho_new, ca_left, ca_right)

        # Return appropriate cost based on side
        cll_new = jax.lax.cond(
            side == "left",
            lambda: cll_left,
            lambda: cll_right
        )

        return jnp.abs(cll_target - cll_new)

    @partial(jit, static_argnums=(0, 4))
    def _cost_function_ca(self, f_state, params, ca_target, side):
        """Cost function for contact angle optimization."""
        # Apply parameters and compute new state
        new_state = self._apply_wetting_params_and_update(f_state, params)
        rho_new = self.macroscopic(new_state)[0]

        # Compute new contact angle
        ca_left, ca_right = self.contact_angle.compute(rho_new)

        ca_new = jax.lax.cond(
            side == "left",
            lambda: ca_left,
            lambda: ca_right
        )

        return jnp.abs(ca_target - ca_new)

    @partial(jit, static_argnums=(0, 4))
    def _evaluate_cost(self, f_state, params, target, cost_func):
        """Evaluate cost function for parameter comparison."""
        return cost_func(f_state, params, target, "left")  # Side doesn't matter for comparison

    @partial(jit, static_argnums=(0,))
    def _apply_wetting_params_and_update(self, f_state, new_params):
        """Apply new wetting parameters and recompute macroscopic state."""
        # Update bc_config with new wetting parameters
        updated_bc_config = self._update_bc_config_with_params(new_params)

        # Create new macroscopic operator with updated parameters
        updated_macroscopic = self._create_updated_macroscopic(updated_bc_config)

        # Recompute macroscopic variables with new parameters
        rho, u, force_total = updated_macroscopic(f_state)

        # Return updated state (simplified - in practice you might need full LBM update)
        return f_state  # Placeholder - implement full update cycle if needed

    def _update_bc_config_with_params(self, new_params):
        """Update boundary condition configuration with new wetting parameters."""
        updated_config = self.macroscopic.bc_config.copy()
        if 'wetting_params' not in updated_config:
            updated_config['wetting_params'] = {}

        updated_config['wetting_params'].update(new_params)
        return updated_config

    def _create_updated_macroscopic(self, updated_bc_config):
        """Create new macroscopic operator with updated wetting parameters."""
        from wblbm.operators.macroscopic.macroscopic_multiphase_dw import MacroscopicMultiphaseDW
        from wblbm.operators.macroscopic.macroscopic_multiphase_cs import MacroscopicMultiphaseCS

        # Determine which macroscopic class to use based on current setup
        if isinstance(self.macroscopic, MacroscopicMultiphaseDW):
            return MacroscopicMultiphaseDW(
                self.grid, self.lattice, self.macroscopic.kappa,
                self.macroscopic.interface_width, self.macroscopic.rho_l,
                self.macroscopic.rho_v, force_enabled=self.force_enabled,
                bc_config=updated_bc_config
            )
        else:
            # Handle other macroscopic types as needed
            return self.macroscopic
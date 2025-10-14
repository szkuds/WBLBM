import jax.numpy as jnp
import jax
from jax import jit
import optax
from functools import partial
from typing import Dict, Tuple, NamedTuple


# Register JAX PyTree for wetting parameters
@jax.tree_util.register_pytree_node_class
class WettingParams(NamedTuple):
    d_rho: jnp.ndarray
    phi: jnp.ndarray

    def tree_flatten(self):
        return (self.d_rho, self.phi), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class WettingOptimizer:
    """JAX/Optax-based optimizer for wetting parameters with proper gradient/laplacian update."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 50):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.optimizer = optax.adam(learning_rate)

    @partial(jit, static_argnums=(0,))
    def clamp_params(self, params: WettingParams) -> WettingParams:
        """Clamp parameters to valid ranges: d_rho ∈ (0, 0.99], phi ∈ [1, ∞)."""
        d_rho_clamped = jnp.clip(params.d_rho, 1e-6, 0.99)
        phi_clamped = jnp.maximum(params.phi, 1.0)
        return WettingParams(d_rho_clamped, phi_clamped)

    @partial(jit, static_argnums=(0,))
    def cost_function_cll(self, params: WettingParams, cll_t: jnp.ndarray,
                          eval_func, f_state: jnp.ndarray) -> jnp.ndarray:
        """Cost function for hysteresis case: minimize |cll_t - cll_t+1|."""
        clamped_params = self.clamp_params(params)
        cll_tplus1 = eval_func(f_state, clamped_params)
        return jnp.abs(cll_t - cll_tplus1)

    @partial(jit, static_argnums=(0,))
    def cost_function_ca(self, params: WettingParams, ca_target: jnp.ndarray,
                         eval_func, f_state: jnp.ndarray) -> jnp.ndarray:
        """Cost function for angle correction: minimize |ca_t+1 - ca_target|."""
        clamped_params = self.clamp_params(params)
        ca_tplus1 = eval_func(f_state, clamped_params)
        return jnp.abs(ca_tplus1 - ca_target)

    @partial(jit, static_argnums=(0,))
    def optimization_step(self, opt_state, params: WettingParams, cost_func,
                          *cost_args) -> Tuple[WettingParams, jnp.ndarray]:
        """Single optimization step."""
        loss, grads = jax.value_and_grad(cost_func)(params, *cost_args)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = self.clamp_params(params)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0,))
    def optimize_cll(self, initial_params: WettingParams, cll_t: jnp.ndarray,
                     eval_func, f_state: jnp.ndarray) -> WettingParams:
        """Optimize wetting parameters for hysteresis case."""
        opt_state = self.optimizer.init(initial_params)

        def scan_body(carry, x):
            opt_state, params = carry
            new_opt_state, new_params, loss = self.optimization_step(
                opt_state, params, self.cost_function_cll,
                cll_t, eval_func, f_state
            )
            return (new_opt_state, new_params), loss

        (final_opt_state, final_params), losses = jax.lax.scan(
            scan_body, (opt_state, initial_params), jnp.arange(self.max_iterations)
        )
        return final_params

    @partial(jit, static_argnums=(0,))
    def optimize_ca(self, initial_params: WettingParams, ca_target: jnp.ndarray,
                    eval_func, f_state: jnp.ndarray) -> WettingParams:
        """Optimize wetting parameters for angle correction."""
        opt_state = self.optimizer.init(initial_params)

        def scan_body(carry, x):
            opt_state, params = carry
            new_opt_state, new_params, loss = self.optimization_step(
                opt_state, params, self.cost_function_ca,
                ca_target, eval_func, f_state
            )
            return (new_opt_state, new_params), loss

        (final_opt_state, final_params), losses = jax.lax.scan(
            scan_body, (opt_state, initial_params), jnp.arange(self.max_iterations)
        )
        return final_params

    @partial(jit, static_argnums=(0,))
    def dual_path_optimise_hysteresis(self, initial_d_rho: jnp.ndarray,
                                      initial_phi: jnp.ndarray, cll_t: jnp.ndarray,
                                      eval_func, f_state: jnp.ndarray) -> WettingParams:
        """Explore both d_rho and phi paths for hysteresis optimization."""
        # Optimize d_rho path
        params_path1 = WettingParams(initial_d_rho, initial_phi)
        optimal_path1 = self.optimize_cll(params_path1, cll_t, eval_func, f_state)

        # Optimize phi path
        params_path2 = WettingParams(initial_d_rho, initial_phi)
        optimal_path2 = self.optimize_cll(params_path2, cll_t, eval_func, f_state)

        # Evaluate both paths and choose the best
        cost1 = self.cost_function_cll(optimal_path1, cll_t, eval_func, f_state)
        cost2 = self.cost_function_cll(optimal_path2, cll_t, eval_func, f_state)

        return jax.lax.cond(cost1 <= cost2, lambda: optimal_path1, lambda: optimal_path2)

    @partial(jit, static_argnums=(0,))
    def dual_path_optimise_angle(self, initial_d_rho: jnp.ndarray,
                                 initial_phi: jnp.ndarray, ca_target: jnp.ndarray,
                                 eval_func, f_state: jnp.ndarray) -> WettingParams:
        """Explore both d_rho and phi paths for angle optimization."""
        # Optimize d_rho path
        params_path1 = WettingParams(initial_d_rho, initial_phi)
        optimal_path1 = self.optimize_ca(params_path1, ca_target, eval_func, f_state)

        # Optimize phi path
        params_path2 = WettingParams(initial_d_rho, initial_phi)
        optimal_path2 = self.optimize_ca(params_path2, ca_target, eval_func, f_state)

        # Evaluate both paths and choose the best
        cost1 = self.cost_function_ca(optimal_path1, ca_target, eval_func, f_state)
        cost2 = self.cost_function_ca(optimal_path2, ca_target, eval_func, f_state)

        return jax.lax.cond(cost1 <= cost2, lambda: optimal_path1, lambda: optimal_path2)


def create_evaluation_function(macroscopic_op, contact_angle_op, contact_line_op):
    """Create evaluation functions for optimization with proper gradient/laplacian update."""

    @jit
    def eval_cll(f_state: jnp.ndarray, params: WettingParams) -> jnp.ndarray:
        """Evaluate contact line location with given parameters."""
        # Update wetting parameters in macroscopic operator
        updated_macroscopic = update_macroscopic_operators(macroscopic_op, params)

        # Compute updated macroscopic variables
        rho, u, force = updated_macroscopic(f_state)

        # Compute contact angles and line locations
        ca_left, ca_right = contact_angle_op.compute(rho)
        cll_left, cll_right = contact_line_op.compute(rho, ca_left, ca_right)

        return cll_left  # or cll_right depending on side

    @jit
    def eval_ca(f_state: jnp.ndarray, params: WettingParams) -> jnp.ndarray:
        """Evaluate contact angle with given parameters."""
        # Update wetting parameters in macroscopic operator
        updated_macroscopic = update_macroscopic_operators(macroscopic_op, params)

        # Compute updated macroscopic variables
        rho, u, force = updated_macroscopic(f_state)

        # Compute contact angles
        ca_left, ca_right = contact_angle_op.compute(rho)

        return ca_left  # or ca_right depending on side

    return eval_cll, eval_ca


@jit
def update_macroscopic_operators(macroscopic_op, new_params: WettingParams):
    """Update gradient and laplacian operators with new wetting parameters."""
    # Extract current bc_config
    current_bc_config = macroscopic_op.bc_config.copy()

    # Update wetting parameters
    if 'wetting_params' not in current_bc_config:
        current_bc_config['wetting_params'] = {}

    # Update with new parameters
    current_bc_config['wetting_params'].update({
        'd_rho_left': new_params.d_rho[0] if new_params.d_rho.size > 1 else new_params.d_rho,
        'd_rho_right': new_params.d_rho[1] if new_params.d_rho.size > 1 else new_params.d_rho,
        'phi_left': new_params.phi[0] if new_params.phi.size > 1 else new_params.phi,
        'phi_right': new_params.phi[1] if new_params.phi.size > 1 else new_params.phi,
    })

    # Create new macroscopic operator with updated parameters
    from wblbm.operators.macroscopic.macroscopic_multiphase_dw import MacroscopicMultiphaseDW
    from wblbm.operators.differential.gradient import Gradient
    from wblbm.operators.differential.laplacian import Laplacian

    # Recreate macroscopic operator with updated bc_config
    updated_macroscopic = MacroscopicMultiphaseDW(
        macroscopic_op.grid, macroscopic_op.lattice, macroscopic_op.kappa,
        macroscopic_op.interface_width, macroscopic_op.rho_l, macroscopic_op.rho_v,
        force_enabled=macroscopic_op.force_enabled, bc_config=current_bc_config
    )

    return updated_macroscopic
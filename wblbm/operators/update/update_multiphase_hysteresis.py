import jax
import jax.numpy as jnp
from jax import jit
import optax

from wblbm.operators.differential import Gradient
from wblbm.operators.differential import Laplacian
from wblbm.operators.update import UpdateMultiphase
from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm.operators.wetting.contact_line_location import ContactLineLocation
from wblbm.operators.wetting.wetting_util import WettingParameters


class UpdateMultiphaseHysteresis(UpdateMultiphase):
    """
    UpdateMultiphase with hysteresis optimization for wetting parameters.
    Optimizes both left and right sides and persists updated operators.
    """

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
        super().__init__(grid, lattice, tau, kappa, interface_width, rho_l, rho_v,
                         bc_config, force_enabled, collision_scheme, eos, k_diag, **kwargs)

        # Extract hysteresis parameters from bc_config
        self.hysteresis_params = bc_config['hysteresis_params']
        self.ca_advancing = self.hysteresis_params.get('ca_advancing', 120.0)
        self.ca_receding = self.hysteresis_params.get('ca_receding', 60.0)

        # Setup contact angle and line location calculators
        rho_mean = (rho_l + rho_v) / 2
        self.contact_angle = ContactAngle(rho_mean)
        self.contact_line_location = ContactLineLocation(rho_mean)

        # Optimizer setup
        self.learning_rate = self.hysteresis_params.get('learning_rate', 0.01)
        self.max_iterations = self.hysteresis_params.get('max_iterations', 20)
        self.optimiser = optax.adam(self.learning_rate)

    def _cost_fucntion_cll(self, cll_t: jnp.ndarray, cll_tplus1: jnp.ndarray):
        return jnp.abs(cll_t - cll_tplus1)

    def _cost_function_ca(self, ca_target: jnp.array, ca_tplus1: jnp.array):
        return jnp.abs(ca_target - ca_tplus1)

    def _update_macroscopic(self, params: WettingParameters):
        """Create new macroscopic operator with updated wetting parameters."""
        new_wetting_params = {
            **self.macroscopic.bc_config.get('wetting_params', {}),
            'd_rho_left': params.d_rho_left,
            'd_rho_right': params.d_rho_right,
            'phi_left': params.phi_left,
            'phi_right': params.phi_right,
        }
        new_bc_config = {
            **self.macroscopic.bc_config,
            'wetting_params': new_wetting_params
        }
        new_gradient = Gradient(self.lattice, bc_config=new_bc_config)
        new_laplacian = Laplacian(self.lattice, bc_config=new_bc_config)
        new_macroscopic = type(self.macroscopic)(
            self.macroscopic.grid,
            self.lattice,
            self.macroscopic.kappa,
            self.macroscopic.beta,
            self.macroscopic.rhol,
            self.macroscopic.rhov,
            force_enabled=self.macroscopic.force_enabled,
            bc_config=new_bc_config
        )
        new_macroscopic.gradient = new_gradient
        new_macroscopic.laplacian = new_laplacian
        return new_macroscopic

    def _reevaluate(self, f: jnp.ndarray, params: WettingParameters):
        temp_macroscopic = self._update_macroscopic(params)
        rho, _, _ = temp_macroscopic(f)
        ca_left, ca_right = self.contact_angle.compute(rho)
        cll_left, cll_right = self.contact_line_location.compute(rho, ca_left, ca_right)
        return ca_left, ca_right, cll_left, cll_right, temp_macroscopic

    def _check_hysteresis_window(self, ca_tplus1: jnp.ndarray):
        return jnp.logical_and(
            ca_tplus1 >= self.ca_receding,
            ca_tplus1 <= self.ca_advancing
        )


    def __call__(self, f_t: jnp.ndarray, force: jnp.ndarray = None):
        f_tplus1 = super(UpdateMultiphaseHysteresis, self).__call__(f_t, force)

        # Get the variables from the previous time step (t)
        rho_t, _, _ = self.macroscopic(f_t)
        ca_left_t, ca_right_t = self.contact_angle.compute(rho_t)
        cll_left_t, cll_right_t = self.contact_line_location.compute(rho_t, ca_left_t, ca_right_t)

        # Get the variables from the next time step (tplus1)
        rho_tplus1, _, _ = self.macroscopic(f_tplus1)
        ca_left_tplus1, ca_right_tplus1 = self.contact_angle.compute(rho_tplus1)
        cll_left_tplus1, cll_right_tplus1 = self.contact_line_location.compute(rho_tplus1, ca_left_tplus1, ca_right_tplus1)

        # Check for both sides if the ca_tplus1 is within the hysteresis window
        hysteresis_window_left = self._check_hysteresis_window(ca_left_tplus1)
        hysteresis_window_right = self._check_hysteresis_window(ca_right_tplus1)

        return f_tplus1

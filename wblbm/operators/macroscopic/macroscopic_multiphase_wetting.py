from functools import partial
import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase
from wblbm.operators.differential.gradient_wetting import GradientWetting
from wblbm.operators.differential.laplacian_wetting import LaplacianWetting

class MacroscopicWetting(MacroscopicMultiphase):
    """
    Calculates macroscopic variables for wetting-aware multiphase simulations.
    Uses wetting-specific differential operators for gradient and Laplacian.
    """

    def __init__(
        self,
        grid: Grid,
        lattice: Lattice,
        kappa: float,
        interface_width: int,
        rho_l: float,
        rho_v: float,
        force_enabled: bool = False,
        wetting_enabled: bool = False,  # Flag to enable wetting-specific operators
    ):
        super().__init__(
            grid, lattice, kappa, interface_width, rho_l, rho_v, force_enabled=force_enabled
        )
        if wetting_enabled:
            self.gradient = GradientWetting(lattice, rho_l, rho_v, interface_width=interface_width)
            self.laplacian = LaplacianWetting(lattice, rho_l, rho_v, interface_width=interface_width)
        self.wetting_enabled = wetting_enabled

    @partial(jit, static_argnums=(0,))
    def __call__(
        self,
        f: jnp.ndarray,
        force: jnp.ndarray = None,
        phi_left: jnp.ndarray = None,
        phi_right: jnp.ndarray = None,
        d_rho_left: jnp.ndarray = None,
        d_rho_right: jnp.ndarray = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        rho, u, force = super().__call__(f, force=force)
        if self.wetting_enabled:
            force_int = self.force_int(rho, phi_left, phi_right, d_rho_left, d_rho_right)
        else:
            force_int = self.force_int(rho)
        u_updated = self.u_new(u, force_int)
        return rho, u, force

    @partial(jit, static_argnums=(0,))
    def chem_pot(
        self,
        rho,
        phi_left=None,
        phi_right=None,
        d_rho_left=None,
        d_rho_right=None,
    ):
        mu_0 = self.eos(rho)
        if self.wetting_enabled:
            chem_pot__ = mu_0 - self.kappa * self.laplacian(
                rho, phi_left, phi_right, d_rho_left, d_rho_right
            )
        else:
            raise ValueError('When using the MacroscopicWetting class wetting must be enabled.')
        return chem_pot__

    @partial(jit, static_argnums=(0,))
    def force_int(
        self,
        rho,
        phi_left=None,
        phi_right=None,
        d_rho_left=None,
        d_rho_right=None,
    ):
        if self.wetting_enabled:
            grad_chem_pot = self.gradient(
                self.chem_pot(rho, phi_left, phi_right, d_rho_left, d_rho_right),
                phi_left,
                phi_right,
                d_rho_left,
                d_rho_right,
            )
        else:
            raise ValueError('When using the MacroscopicWetting class wetting must be enabled.')
        return -rho * grad_chem_pot

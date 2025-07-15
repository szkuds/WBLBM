from functools import partial
import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.macroscopic.macroscopic import Macroscopic
from wblbm.operators.differential.gradient_wetting import GradientWetting
from wblbm.operators.differential.laplacian_wetting import LaplacianWetting

class MacroscopicWetting(Macroscopic):
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
            grid, lattice, force_enabled
        )
        self.kappa = kappa
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.beta = 8 * kappa / (float(interface_width) ** 2 * (rho_l - rho_v) ** 2)
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
        rho, u = super().__call__(f, force=force)
        force_int = self.force_int(rho, phi_left, phi_right, d_rho_left, d_rho_right)
        u_updated = self.u_new(u, force_int)
        if force is None:
            force_total = force_int
        else:
            force_total = force_int + force
        return rho, u_updated, force_total

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

    @partial(jit, static_argnums=(0,))
    def u_new(self, u, force):
        """
        Update velocity with interaction force.
        """
        # Both u and force have shape (nx, ny, 1, 2)
        return u + force / 2

    @partial(jit, static_argnums=(0,))
    def eos(self, rho):
        """Equation of state - extract 2D data for computation"""
        rho_2d = rho[:, :, 0, 0]  # Extract (nx, ny) from (nx, ny, 1, 1)
        eos_2d = (
                2
                * self.beta
                * (rho_2d - self.rho_l)
                * (rho_2d - self.rho_v)
                * (2 * rho_2d - self.rho_l - self.rho_v)
        )

        # Convert back to 4D format
        eos_4d = jnp.zeros_like(rho)
        eos_4d = eos_4d.at[:, :, 0, 0].set(eos_2d)
        return eos_4d

import jax.numpy as jnp
from wblbm.operator.macroscopic.macroscopic import Macroscopic
from wblbm.operator.differential.gradient import Gradient
from wblbm.operator.differential.laplacian import Laplacian
from wblbm.lattice.lattice import Lattice


class MacroscopicMultiphase(Macroscopic):
    """
    Calculates macroscopic variables for multiphase simulations.
    Inherits from Macroscopic and adds multiphase-specific methods.
    """

    def __init__(self, grid, lattice: Lattice, kappa: float, beta: float, rho_l: float, rho_v: float):
        super().__init__(grid, lattice)
        self.kappa = kappa
        self.beta = beta
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.gradient = Gradient(lattice)
        self.laplacian = Laplacian(lattice)

    def __call__(self, f: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculate the macroscopic density and velocity fields from the population distribution.

        Args:
            f (jnp.ndarray): Population distribution, shape (nx, ny, q, 1)

        Returns:
            tuple: (rho, u)
                rho (jnp.ndarray): Density field, shape (nx, ny, 1, 1)
                u (jnp.ndarray): Velocity field, shape (nx, ny, 1, 2)
        """
        rho, u = super().__call__(f)
        force_int = self.force_int(rho)
        u_updated = self.u_new(u, rho)
        return rho, u_updated, force_int

    def eos(self, rho):
        return 2 * self.beta * (rho - self.rho_l) * (rho - self.rho_v) * (2 * rho - self.rho_l - self.rho_v)

    def chem_pot(self, rho):
        """
        Calculate the chemical potential.
        """
        mu_0 = self.eos(rho)
        chem_pot__ = mu_0 - self.kappa * self.laplacian(rho)
        return chem_pot__

    def force_int(self, rho):
        """
        Calculate the interaction force.
        """
        grad_chem_pot = self.gradient(self.chem_pot(rho))
        # Return -rho * grad_chem_pot, shape (2, nx, ny)
        return -rho * grad_chem_pot

    def u_new(self, u, rho):
        """
        Update velocity with interaction force.
        """
        force = self.force_int(rho)
        # u: shape (nx, ny, 1, d), force: (2, nx, ny)
        # Need to broadcast force to shape (nx, ny, 1, d)
        force_broadcast = jnp.moveaxis(force, 0, -1)[..., jnp.newaxis, :]  # (nx, ny, 1, d)
        return u + force_broadcast / 2

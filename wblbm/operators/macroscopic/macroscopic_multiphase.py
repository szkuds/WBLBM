from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.operators.macroscopic.macroscopic import Macroscopic
from wblbm.operators.differential.gradient import Gradient
from wblbm.operators.differential.laplacian import Laplacian
from wblbm.lattice.lattice import Lattice


class MacroscopicMultiphase(Macroscopic):
    """
    Calculates macroscopic variables for multiphase simulations.
    Inherits from Macroscopic and adds multiphase-specific methods.
    """

    def __init__(self, grid: Grid, lattice: Lattice, kappa: float, interface_width: int, rho_l: float, rho_v: float, force_enabled: bool = False):
        super().__init__(grid, lattice, force_enabled=force_enabled)  # Pass force_enabled to parent
        self.kappa = kappa
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.gradient = Gradient(lattice)
        self.laplacian = Laplacian(lattice)
        self.beta = 8 * kappa / (float(interface_width) ** 2 * (rho_l - rho_v) ** 2)

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.ndarray, force: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculate the macroscopic density and velocity fields from the population distribution.

        Args:
            f (jnp.ndarray): Population distribution, shape (nx, ny, q, 1)
            force (jnp.ndarray, optional): External force field, shape (nx, ny, 1, 2)

        Returns:
            tuple: (rho, u, force_int)
                rho (jnp.ndarray): Density field, shape (nx, ny, 1, 1)
                u (jnp.ndarray): Velocity field, shape (nx, ny, 1, 2)
                force_int (jnp.ndarray): Interaction force, shape (nx, ny, 1, 2)
        """
        rho, u = super().__call__(f, force=force)  # Pass force to parent for velocity adjustment
        force_int = self.force_int(rho)
        u_updated = self.u_new(u, force_int)
        return rho, u_updated, force_int

    @partial(jit, static_argnums=(0,))
    def eos(self, rho):
        """Equation of state - extract 2D data for computation"""
        rho_2d = rho[:, :, 0, 0]  # Extract (nx, ny) from (nx, ny, 1, 1)
        eos_2d = 2 * self.beta * (rho_2d - self.rho_l) * (rho_2d - self.rho_v) * (2 * rho_2d - self.rho_l - self.rho_v)

        # Convert back to 4D format
        eos_4d = jnp.zeros_like(rho)
        eos_4d = eos_4d.at[:, :, 0, 0].set(eos_2d)
        return eos_4d

    @partial(jit, static_argnums=(0,))
    def chem_pot(self, rho):
        """
        Calculate the chemical potential.
        """
        mu_0 = self.eos(rho)
        chem_pot__ = mu_0 - self.kappa * self.laplacian(rho)
        return chem_pot__

    @partial(jit, static_argnums=(0,))
    def force_int(self, rho):
        """
        Calculate the interaction force.
        """
        grad_chem_pot = self.gradient(self.chem_pot(rho))  # Shape: (nx, ny, 1, 2)
        # Return -rho * grad_chem_pot, shape (nx, ny, 1, 2)
        return -rho * grad_chem_pot

    @partial(jit, static_argnums=(0,))
    def u_new(self, u, force):
        """
        Update velocity with interaction force.
        """
        # Both u and force have shape (nx, ny, 1, 2)
        return u + force / 2

import jax.numpy as jnp
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice
from .base import CollisionBase


class CollisionBGK(CollisionBase):
    """
    Callable class to perform the collision step of the LBM.
    Implements the BGK collision operators with source terms.
    """

    def __init__(self, grid: Grid, lattice: Lattice, tau: float) -> None:
        """
        Initialize the CollisionBGK operators.

        Args:
            grid (Grid): Grid object containing simulation domain information
            lattice (Lattice): Lattice object containing lattice properties
            tau (float): Relaxation time parameter
        """
        super().__init__(grid, lattice)
        self.tau: float = tau

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Perform the collision step of the LBM.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): Equilibrium distribution function.

        Returns:
            jnp.ndarray: Post-collision distribution function.
        """
        if source is None:
            return (1 - (1 / self.tau)) * f + (1 / self.tau) * feq
        else:
            return (
                (1 - (1 / self.tau)) * f
                + (1 / self.tau) * feq
                + (1 - (1 / (2 * self.tau))) * source
            )

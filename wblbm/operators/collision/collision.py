import jax.numpy as jnp
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice


class Collision:
    """
    Callable class to perform the collision step of the LBM.
    Implements the BGK collision operators with source terms.
    """

    def __init__(self, grid: Grid, lattice: Lattice, tau: float) -> None:
        """
        Initialize the Collision operators.

        Args:
            grid (Grid): Grid object containing simulation domain information
            lattice (Lattice): Lattice object containing lattice properties
            tau (float): Relaxation time parameter
        """
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.tau: float = tau

    def __call__(self, f: jnp.ndarray, feq: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the collision step of the LBM.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): Equilibrium distribution function.

        Returns:
            jnp.ndarray: Post-collision distribution function.
        """
        return (1 - (1 / self.tau)) * f + (1 / self.tau) * feq

from abc import ABC

import jax.numpy as jnp
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice


class CollisionBase(ABC):
    """
    Callable class to perform the collision step of the LBM.
    Implements the BGK collision operators with source terms.
    """

    def __init__(self, grid: Grid, lattice: Lattice) -> None:
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

    def __call__(self, f: jnp.ndarray, feq: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the collision step of the LBM.

        Needs to be implemented by the subclass
        """
        pass

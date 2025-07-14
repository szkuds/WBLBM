import jax.numpy as jnp

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.collision import Collision
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase


class CollisionMultiphase(Collision):
    def __init__(self, grid: Grid, lattice: Lattice, tau: float) -> None:
        super().__init__(grid, lattice, tau)

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        result = super().__call__(f, feq)
        if source is not None:
            result += (1 - (1 / (2 * self.tau))) * source
        return result

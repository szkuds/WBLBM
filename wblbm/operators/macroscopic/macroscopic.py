from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice
from typing import Tuple
from wblbm.operators.differential.gradient import Gradient
from wblbm.utils.timing import time_function, TIMING_ENABLED


class Macroscopic:
    """
    Calculates the macroscopic density and velocity fields from the population distribution.
    """

    def __init__(self, grid: Grid, lattice: Lattice) -> None:
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.cx: jnp.ndarray = jnp.array(lattice.c[0])
        self.cy: jnp.ndarray = jnp.array(lattice.c[1])
        self.gradient = Gradient(lattice)

    @time_function(enable_timing=TIMING_ENABLED)
    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            f (jnp.ndarray): Population distribution, shape (nx, ny, q, 1)
        Returns:
            tuple: (rho, u)
                rho (jnp.ndarray): Density field, shape (nx, ny, 1, 1)
                u (jnp.ndarray): Velocity field, shape (nx, ny, 1, 2)
        """
        gradient = self.gradient
        if self.d == 2:
            # Compute density
            rho = jnp.sum(f, axis=2, keepdims=True)  # (nx, ny, 1, 1)

            # Compute velocity
            cx = self.cx.reshape((1, 1, self.q, 1))
            cy = self.cy.reshape((1, 1, self.q, 1))
            ux = jnp.sum(f * cx, axis=2, keepdims=True)
            uy = jnp.sum(f * cy, axis=2, keepdims=True)
            u = jnp.concatenate([ux, uy], axis=-1) / rho  # (nx, ny, 1, 2)
            return rho, u
        elif self.d == 3:
            raise NotImplementedError("Dimension larger than 2 not supported.")
from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice
from wblbm.utils.timing import time_function, TIMING_ENABLED


class EquilibriumBubble:
    """
    Callable class to calculate the equilibrium population distribution for LBM.
    """

    def __init__(self, grid: Grid, lattice: Lattice, g: float, rho_ref: float) -> None:
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.w = lattice.w
        self.cx = lattice.c[0]
        self.cy = lattice.c[1]
        self.g = g
        self.rho_ref = rho_ref

    @time_function(enable_timing=TIMING_ENABLED)
    @partial(jit, static_argnums=(0,))
    def __call__(self, rho_, u_):
        """
        Calculate the equilibrium distribution function.

        Args:
            rho_ (jnp.ndarray): Density field, shape (nx, ny, 1, 1)
            u_ (jnp.ndarray): Velocity field, shape (nx, ny, 1, 2)

        Returns:
            jnp.ndarray: Equilibrium distribution function, shape (nx, ny, q, 1)
        """
        nx, ny, q = self.nx, self.ny, self.q
        w = self.w
        cx, cy = self.cx, self.cy

        # Extract velocity components
        ux = u_[:, :, 0, 0]  # Shape: (nx, ny, 1)
        uy = u_[:, :, 0, 1]  # Shape: (nx, ny, 1)

        # Squeeze density to match velocity dimensions
        rho = rho_[:, :, 0, 0]  # Shape: (nx, ny, 1)

        # Initialize equilibrium distribution - note the 4D shape
        f_eq = jnp.zeros((nx, ny, q, 1))
        u2 = ux * ux + uy * uy
        y_coords = jnp.arange(ny).reshape(1, -1)
        # Calculate equilibrium for each velocity direction
        for i in range(1, q):
            cu = cx[i] * ux + cy[i] * uy
            cu2 = cu * cu
            f_eq = f_eq.at[:, :, i, 0].set(w[i] * rho * (3 * cu + 4.5 * cu2 - 1.5 * u2) +
                                           w[i] * (3 * self.rho_ref * self.g * y_coords))
        f_eq = f_eq.at[:, :, 0, 0].set(rho - (1 - w[0]) * (3 * self.rho_ref * self.g * y_coords) - w[0] * rho * 1.5 * u2)

        return f_eq

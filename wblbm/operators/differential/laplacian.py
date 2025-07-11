from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.lattice.lattice import Lattice


class Laplacian:
    """
    Callable class to calculate the Laplacian of a 2D grid using the LBM stencil.
    """

    def __init__(self, lattice: Lattice):
        self.w = lattice.w

    @partial(jit, static_argnums=(0,))
    def __call__(self, grid):
        """
        Calculate the Laplacian of a 2D grid.

        Args:
            grid (jnp.ndarray): Input grid, shape (nx, ny, 1, 1)

        Returns:
            jnp.ndarray: Laplacian of the input grid, shape (nx, ny, 1, 1)
        """
        # Extract 2D data from 4D input
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]  # Extract (nx, ny) from (nx, ny, 1, 1)
        else:
            grid_2d = grid

        w = self.w

        laplacian_2d = jnp.zeros_like(grid_2d)
        grid_padded = jnp.pad(grid_2d, pad_width=1, mode='wrap')

        # Side nodes
        grid_ineg1_j0 = grid_padded[:-2, 1:-1]
        grid_ipos1_j0 = grid_padded[2:, 1:-1]
        grid_i0_jneg1 = grid_padded[1:-1, :-2]
        grid_i0_jpos1 = grid_padded[1:-1, 2:]

        # Corner nodes
        grid_ipos1_jpos1 = grid_padded[2:, 2:]
        grid_ineg1_jpos1 = grid_padded[:-2, 2:]
        grid_ineg1_jneg1 = grid_padded[:-2, :-2]
        grid_ipos1_jneg1 = grid_padded[2:, :-2]

        # Central node
        grid_i0_j0 = grid_padded[1:-1, 1:-1]

        laplacian_2d = laplacian_2d.at[:, :].set(6 *
                                                 (w[1] * (grid_ipos1_j0 - grid_i0_j0) +
                                                  w[2] * (grid_i0_jpos1 - grid_i0_j0) +
                                                  w[3] * (grid_ineg1_j0 - grid_i0_j0) +
                                                  w[4] * (grid_i0_jneg1 - grid_i0_j0) +
                                                  w[5] * (grid_ipos1_jpos1 - grid_i0_j0) +
                                                  w[6] * (grid_ineg1_jpos1 - grid_i0_j0) +
                                                  w[7] * (grid_ineg1_jneg1 - grid_i0_j0) +
                                                  w[8] * (grid_ipos1_jneg1 - grid_i0_j0)
                                                  )
                                                 )

        # Convert to 4D format: (nx, ny, 1, 1)
        laplacian_4d = jnp.zeros((grid_2d.shape[0], grid_2d.shape[1], 1, 1))
        laplacian_4d = laplacian_4d.at[:, :, 0, 0].set(laplacian_2d)

        return laplacian_4d

from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.lattice.lattice import Lattice


class Gradient:
    """
    Callable class to calculate the gradient of a field using the provided stencil.
    """

    def __init__(self, lattice: Lattice):
        self.w = lattice.w
        self.c = lattice.c

    @partial(jit, static_argnums=(0,))
    def __call__(self, grid):
        """
        Calculate the gradient using the provided stencil.

        Args:
            grid (jnp.ndarray): Input field, shape (nx, ny, 1, 1)

        Returns:
            jnp.ndarray: Gradient, shape (nx, ny, 1, 2)
        """
        # Extract 2D data from 4D input
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]  # Extract (nx, ny) from (nx, ny, 1, 1)
        else:
            grid_2d = grid

        w = self.w
        c = self.c

        grad_ = jnp.zeros((2, grid_2d.shape[0], grid_2d.shape[1]))
        grid_padded = jnp.pad(grid_2d, pad_width=1, mode="wrap")

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

        grad_ = grad_.at[0, :, :].set(
            3
            * (
                w[1] * c[0, 1] * grid_ipos1_j0
                + w[3] * c[0, 3] * grid_ineg1_j0
                + w[5] * c[0, 5] * grid_ipos1_jpos1
                + w[6] * c[0, 6] * grid_ineg1_jpos1
                + w[7] * c[0, 7] * grid_ineg1_jneg1
                + w[8] * c[0, 8] * grid_ipos1_jneg1
            )
        )

        grad_ = grad_.at[1, :, :].set(
            3
            * (
                w[2] * c[1, 2] * grid_i0_jpos1
                + w[4] * c[1, 4] * grid_i0_jneg1
                + w[5] * c[1, 5] * grid_ipos1_jpos1
                + w[6] * c[1, 6] * grid_ineg1_jpos1
                + w[7] * c[1, 7] * grid_ineg1_jneg1
                + w[8] * c[1, 8] * grid_ipos1_jneg1
            )
        )

        # Convert to 4D format: (nx, ny, 1, 2)
        grad_4d = jnp.zeros((grid_2d.shape[0], grid_2d.shape[1], 1, 2))
        grad_4d = grad_4d.at[:, :, 0, 0].set(grad_[0, :, :])
        grad_4d = grad_4d.at[:, :, 0, 1].set(grad_[1, :, :])

        return grad_4d

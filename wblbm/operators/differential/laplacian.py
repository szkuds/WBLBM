import jax.numpy as jnp
from wblbm.lattice.lattice import Lattice

class Laplacian:
    """
    Callable class to calculate the Laplacian of a 2D grid using the LBM stencil.
    """

    def __init__(self, lattice: Lattice):
        self.w = lattice.w

    def __call__(self, grid):
        """
        Calculate the Laplacian of a 2D grid.

        Args:
            grid (jnp.ndarray): Input 2D grid.

        Returns:
            jnp.ndarray: Laplacian of the input grid.
        """
        w = self.w

        laplacian_ = jnp.zeros_like(grid)
        grid_padded = jnp.pad(grid, pad_width=1, mode='wrap')

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

        laplacian_ = laplacian_.at[:, :].set(6 *
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

        return laplacian_


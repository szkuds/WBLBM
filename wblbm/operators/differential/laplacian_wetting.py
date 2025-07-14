import jax.numpy as jnp
from wblbm.lattice.lattice import Lattice
from .abstract_wetting import AbstractWetting


class LaplacianWetting(AbstractWetting):
    """
    Callable class to calculate the Laplacian of a 2D grid with wetting boundary conditions.
    """

    def laplacian(self, grid, phi_left, phi_right, d_rho_left, d_rho_right):
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]
        else:
            grid_2d = grid

        laplacian_ = jnp.zeros_like(grid_2d)
        grid_padded = self._pad_grid(grid_2d)
        grid_padded = self._apply_wetting_boundary(grid_padded)
        grid_padded = self._apply_wetting_mask(
            grid_padded, phi_left, phi_right, d_rho_left, d_rho_right
        )

        w = self.w

        grid_ineg1_j0 = grid_padded[:-2, 1:-1]
        grid_ipos1_j0 = grid_padded[2:, 1:-1]
        grid_i0_jneg1 = grid_padded[1:-1, :-2]
        grid_i0_jpos1 = grid_padded[1:-1, 2:]
        grid_ipos1_jpos1 = grid_padded[2:, 2:]
        grid_ineg1_jpos1 = grid_padded[:-2, 2:]
        grid_ineg1_jneg1 = grid_padded[:-2, :-2]
        grid_ipos1_jneg1 = grid_padded[2:, :-2]
        grid_i0_j0 = grid_padded[1:-1, 1:-1]

        laplacian_ = laplacian_.at[:, :].set(
            6
            * (
                w[1] * (grid_ipos1_j0 - grid_i0_j0)
                + w[2] * (grid_i0_jpos1 - grid_i0_j0)
                + w[3] * (grid_ineg1_j0 - grid_i0_j0)
                + w[4] * (grid_i0_jneg1 - grid_i0_j0)
                + w[5] * (grid_ipos1_jpos1 - grid_i0_j0)
                + w[6] * (grid_ineg1_jpos1 - grid_i0_j0)
                + w[7] * (grid_ineg1_jneg1 - grid_i0_j0)
                + w[8] * (grid_ipos1_jneg1 - grid_i0_j0)
            )
        )

        return laplacian_

    def compute(self, *args, **kwargs):
        return self.laplacian(*args, **kwargs)

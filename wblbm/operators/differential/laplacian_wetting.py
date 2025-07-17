import jax.numpy as jnp
from wblbm.lattice.lattice import Lattice
from .abstract_wetting import AbstractWetting


class LaplacianWetting(AbstractWetting):
    """
    Callable class to calculate the Laplacian of a 2D grid with wetting boundary conditions.
    """

    def __init__(
        self, lattice: Lattice, rho_l=None, rho_v=None, interface_width: int = 1
    ):
        super().__init__(lattice, rho_l, rho_v, interface_width=interface_width)

    def __call__(self, grid, phi_left, phi_right, d_rho_left, d_rho_right):
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]
        else:
            grid_2d = grid

        edge = "bottom"
        grid_padded = self._pad_grid(grid_2d, edge)
        grid_padded = self._apply_wetting_boundary(grid_padded, edge)
        grid_padded = self._apply_wetting_mask(
            grid_padded, edge, phi_left, phi_right, d_rho_left, d_rho_right
        )

        w = self.w
        neighbors = self._extract_neighbors(grid_padded)

        laplacian_2d = jnp.zeros_like(grid_2d)
        laplacian_2d = laplacian_2d.at[:, :].set(
            6
            * (
                w[1] * (neighbors["ipos1_j0"] - neighbors["i0_j0"])
                + w[2] * (neighbors["i0_jpos1"] - neighbors["i0_j0"])
                + w[3] * (neighbors["ineg1_j0"] - neighbors["i0_j0"])
                + w[4] * (neighbors["i0_jneg1"] - neighbors["i0_j0"])
                + w[5] * (neighbors["ipos1_jpos1"] - neighbors["i0_j0"])
                + w[6] * (neighbors["ineg1_jpos1"] - neighbors["i0_j0"])
                + w[7] * (neighbors["ineg1_jneg1"] - neighbors["i0_j0"])
                + w[8] * (neighbors["ipos1_jneg1"] - neighbors["i0_j0"])
            )
        )

        # Convert to 4D format: (nx, ny, 1, 1)
        laplacian_4d = jnp.zeros((grid_2d.shape[0], grid_2d.shape[1], 1, 1))
        laplacian_4d = laplacian_4d.at[:, :, 0, 0].set(laplacian_2d)
        return laplacian_4d

    def compute(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

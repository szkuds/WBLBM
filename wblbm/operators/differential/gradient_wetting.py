import jax.numpy as jnp
from wblbm.lattice.lattice import Lattice
from .abstract_wetting import AbstractWetting


class GradientWetting(AbstractWetting):
    """
    Calculates gradients for wetting boundary conditions.
    """

    def __init__(self, lattice: Lattice, rho_l=None, rho_v=None):
        super().__init__(lattice, rho_l, rho_v)
        self.c = lattice.c

    def _extract_neighbors(self, grid_padded: jnp.ndarray) -> dict:
        return {
            "ineg1_j0": grid_padded[:-2, 1:-1],
            "ipos1_j0": grid_padded[2:, 1:-1],
            "i0_jneg1": grid_padded[1:-1, :-2],
            "i0_jpos1": grid_padded[1:-1, 2:],
            "ipos1_jpos1": grid_padded[2:, 2:],
            "ineg1_jpos1": grid_padded[:-2, 2:],
            "ineg1_jneg1": grid_padded[:-2, :-2],
            "ipos1_jneg1": grid_padded[2:, :-2],
        }

    def _compute_gradient(self, neighbors: dict, nx: int, ny: int) -> jnp.ndarray:
        grad_ = jnp.zeros((2, nx, ny))
        w = self.w
        c = self.c

        grad_ = grad_.at[0, :, :].set(
            3
            * (
                w[1] * c[0, 1] * neighbors["ipos1_j0"]
                + w[3] * c[0, 3] * neighbors["ineg1_j0"]
                + w[5] * c[0, 5] * neighbors["ipos1_jpos1"]
                + w[6] * c[0, 6] * neighbors["ineg1_jpos1"]
                + w[7] * c[0, 7] * neighbors["ineg1_jneg1"]
                + w[8] * c[0, 8] * neighbors["ipos1_jneg1"]
            )
        )

        grad_ = grad_.at[1, :, :].set(
            3
            * (
                w[2] * c[1, 2] * neighbors["i0_jpos1"]
                + w[4] * c[1, 4] * neighbors["i0_jneg1"]
                + w[5] * c[1, 5] * neighbors["ipos1_jpos1"]
                + w[6] * c[1, 6] * neighbors["ineg1_jpos1"]
                + w[7] * c[1, 7] * neighbors["ineg1_jneg1"]
                + w[8] * c[1, 8] * neighbors["ipos1_jneg1"]
            )
        )

        return grad_

    def gradient_chem_pot(self, grid: jnp.ndarray) -> jnp.ndarray:
        """
        Gradient for chemical potential with wetting boundary.
        Args:
            grid (jnp.ndarray): Input field, shape (nx, ny) or (nx, ny, 1, 1)
        Returns:
            jnp.ndarray: Gradient, shape (2, nx, ny)
        """
        # Extract 2D data from 4D input
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]
        else:
            grid_2d = grid
        grid_padded = self._pad_grid(grid_2d)
        grid_padded = self._apply_wetting_boundary(grid_padded)
        neighbors = self._extract_neighbors(grid_padded)
        return self._compute_gradient(neighbors, grid_2d.shape[0], grid_2d.shape[1])

    def gradient_rho(
        self, grid: jnp.ndarray, phi_left, phi_right, d_rho_left, d_rho_right
    ) -> jnp.ndarray:
        """
        Gradient for density with wetting boundary and CA masks.
        Args:
            grid (jnp.ndarray): Input field, shape (nx, ny) or (nx, ny, 1, 1)
            phi_left, phi_right, d_rho_left, d_rho_right: wetting parameters
        Returns:
            jnp.ndarray: Gradient, shape (2, nx, ny)
        """
        # Extract 2D data from 4D input
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]
        else:
            grid_2d = grid
        grid_padded = self._pad_grid(grid_2d)
        grid_padded = self._apply_wetting_boundary(grid_padded)
        grid_padded = self._apply_wetting_mask(
            grid_padded, phi_left, phi_right, d_rho_left, d_rho_right
        )
        neighbors = self._extract_neighbors(grid_padded)
        return self._compute_gradient(neighbors, grid_2d.shape[0], grid_2d.shape[1])

    def compute(self, *args, **kwargs):
        return (
            self.gradient_rho(*args, **kwargs)
            if len(args) > 1
            else self.gradient_chem_pot(*args, **kwargs)
        )

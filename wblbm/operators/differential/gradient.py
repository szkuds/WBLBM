from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.lattice.lattice import Lattice


class Gradient:
    """
    Callable class to calculate the gradient of a field using the LBM stencil,
    supporting asymmetric per-side padding.
    """

    def __init__(self, lattice: Lattice, bc_config: dict = None):
        self.w = lattice.w
        self.c = lattice.c
        self.bc_config = bc_config
        self.padmode = self._determine_padding_modes()

    def _determine_padding_modes(self):
        """Determine padding modes based on boundary conditions."""
        if not self.bc_config:
            return ["wrap", "wrap", "wrap", "wrap"]  # Default periodic

        pad_mode = ["wrap", "wrap", "wrap", "wrap"]  # [bottom, right, top, left]
        for edge, bc_type in self.bc_config.items():
            if bc_type in ["symmetry", "bounce-back"]:
                if edge == "bottom":
                    pad_mode[0] = "edge"
                elif edge == "right":
                    pad_mode[1] = "edge"
                elif edge == "top":
                    pad_mode[2] = "edge"
                elif edge == "left":
                    pad_mode[3] = "edge"
        return pad_mode

    @partial(jit, static_argnums=(0,))
    def __call__(self, grid, padmode: list = None):
        """
        Calculate the gradient using the provided stencil and per-side boundary modes.

        Args:
            grid (jnp.ndarray): Input field, shape (nx, ny, 1, 1)
            padmode (list, optional): List of padding modes for each pad step

        Returns:
            jnp.ndarray: Gradient field, shape (nx, ny, 1, 2)
        """
        # Use provided padmode or fall back to self.padmode
        effective_padmode = padmode if padmode is not None else self.padmode

        # Extract 2D data from 4D input
        if grid.ndim == 4:
            grid_2d = grid[:, :, 0, 0]  # Extract (nx, ny) from (nx, ny, 1, 1)
        else:
            grid_2d = grid

        w = self.w
        c = self.c

        grad_ = jnp.zeros((2, grid_2d.shape[0], grid_2d.shape[1]))

        # Apply asymmetric per-side padding (same convention/order as Laplacian)
        grid_padded___ = jnp.pad(grid_2d, pad_width=((0, 0), (0, 1)), mode=effective_padmode[0])
        grid_padded__ = jnp.pad(grid_padded___, pad_width=((0, 0), (1, 0)), mode=effective_padmode[1])
        grid_padded_ = jnp.pad(grid_padded__, pad_width=((0, 1), (0, 0)), mode=effective_padmode[2])
        grid_padded = jnp.pad(grid_padded_, pad_width=((1, 0), (0, 0)), mode=effective_padmode[3])

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

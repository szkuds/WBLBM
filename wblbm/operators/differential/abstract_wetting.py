import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Dict
from wblbm.lattice.lattice import Lattice


class AbstractWetting(ABC):
    """
    Abstract base class for wetting boundary condition calculations.
    Supports selectable edges via wetting_config.
    """

    def __init__(self, lattice: Lattice, rho_l=None, rho_v=None, wetting_config: Dict[str, str] = None):
        self.w = lattice.w
        self.c = getattr(lattice, "c", None)
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.wetting_config = wetting_config or {"bottom": "wetting"}
        valid_edges = ["top", "bottom", "left", "right"]
        valid_types = ["wetting"]
        for edge, wc_type in self.wetting_config.items():
            if edge not in valid_edges:
                raise ValueError(f"Invalid edge: {edge}. Must be one of {valid_edges}.")
            if wc_type not in valid_types:
                raise ValueError(f"Invalid wetting type: {wc_type}. Must be one of {valid_types}.")

    def apply_wetting(self, grid: jnp.ndarray, phi_left, phi_right, d_rho_left, d_rho_right) -> jnp.ndarray:
        """
        Apply wetting boundary conditions to all configured edges.
        """
        for edge, wc_type in self.wetting_config.items():
            if wc_type == "wetting":
                grid = self._apply_wetting_to_edge(grid, edge, phi_left, phi_right, d_rho_left, d_rho_right)
        return grid

    def _pad_grid(self, grid: jnp.ndarray, edge: str) -> jnp.ndarray:
        # Pad according to edge orientation
        if edge in ["bottom", "top"]:
            grid_padded = jnp.pad(grid, pad_width=((0, 0), (1, 1)), mode="edge")
            grid_padded = jnp.pad(grid_padded, pad_width=((1, 1), (0, 0)), mode="wrap")
        elif edge in ["left", "right"]:
            grid = grid.T
            grid_padded = jnp.pad(grid, pad_width=((0, 0), (1, 1)), mode="wrap")
            grid_padded = jnp.pad(grid_padded, pad_width=((1, 1), (0, 0)), mode="edge")
            grid_padded = grid_padded.T
        if edge == "top":
            grid_padded = jnp.flip(grid_padded, axis=1)
        elif edge == "right":
            grid_padded = jnp.flip(grid_padded, axis=0)
        return grid_padded

    def _apply_wetting_boundary(self, grid_padded: jnp.ndarray, edge: str) -> jnp.ndarray:
        # Generalize for each edge
        if edge == "bottom":
            grid_padded = grid_padded.at[1:-1, 0].set(
                (1/3 * grid_padded[1:-1, 1] + 1/12 * grid_padded[0:-2, 1] + 1/12 * grid_padded[2:, 1]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[0, 0].set(
                (1/3 * grid_padded[0, 1] + 1/12 * grid_padded[-1, 1] + 1/12 * grid_padded[1, 1]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[-1, 0].set(
                (1/3 * grid_padded[-1, 1] + 1/12 * grid_padded[0, 1] + 1/12 * grid_padded[-2, 1]) /
                (1/3 + 1/12 + 1/12)
            )
        elif edge == "top":
            grid_padded = grid_padded.at[1:-1, -1].set(
                (1/3 * grid_padded[1:-1, -2] + 1/12 * grid_padded[0:-2, -2] + 1/12 * grid_padded[2:, -2]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[0, -1].set(
                (1/3 * grid_padded[0, -2] + 1/12 * grid_padded[-1, -2] + 1/12 * grid_padded[1, -2]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[-1, -1].set(
                (1/3 * grid_padded[-1, -2] + 1/12 * grid_padded[0, -2] + 1/12 * grid_padded[-2, -2]) /
                (1/3 + 1/12 + 1/12)
            )
        elif edge == "left":
            grid_padded = grid_padded.at[0, 1:-1].set(
                (1/3 * grid_padded[1, 1:-1] + 1/12 * grid_padded[1, 0:-2] + 1/12 * grid_padded[1, 2:]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[0, 0].set(
                (1/3 * grid_padded[1, 0] + 1/12 * grid_padded[1, -1] + 1/12 * grid_padded[1, 1]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[0, -1].set(
                (1/3 * grid_padded[1, -1] + 1/12 * grid_padded[1, 0] + 1/12 * grid_padded[1, -2]) /
                (1/3 + 1/12 + 1/12)
            )
        elif edge == "right":
            grid_padded = grid_padded.at[-1, 1:-1].set(
                (1/3 * grid_padded[-2, 1:-1] + 1/12 * grid_padded[-2, 0:-2] + 1/12 * grid_padded[-2, 2:]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[-1, 0].set(
                (1/3 * grid_padded[-2, 0] + 1/12 * grid_padded[-2, -1] + 1/12 * grid_padded[-2, 1]) /
                (1/3 + 1/12 + 1/12)
            )
            grid_padded = grid_padded.at[-1, -1].set(
                (1/3 * grid_padded[-2, -1] + 1/12 * grid_padded[-2, 0] + 1/12 * grid_padded[-2, -2]) /
                (1/3 + 1/12 + 1/12)
            )
        return grid_padded

    def _apply_wetting_mask(
        self, grid_padded: jnp.ndarray, edge: str, phi_left, phi_right, d_rho_left, d_rho_right
    ) -> jnp.ndarray:
        rho_l = self.rho_l
        rho_v = self.rho_v
        # Select boundary slice based on edge
        if edge == "bottom":
            boundary_slice = grid_padded[1:-1, 0]
        elif edge == "top":
            boundary_slice = grid_padded[1:-1, -1]
        elif edge == "left":
            boundary_slice = grid_padded[0, 1:-1]
        elif edge == "right":
            boundary_slice = grid_padded[-1, 1:-1]
        else:
            raise ValueError(f"Unknown edge: {edge}")

        mask1 = boundary_slice < (0.95 * rho_l + 0.05 * rho_v)
        mask2 = boundary_slice > (0.95 * rho_v + 0.05 * rho_l)
        mask_final = mask1 * mask2

        mask1_int = jnp.array(mask1, dtype=int)
        diff_mask1 = jnp.diff(mask1_int)

        transition_index_left_mask1 = (
            jnp.where(diff_mask1 == -1, size=1, fill_value=0)[0] + self.w
        )
        transition_index_right_mask1 = jnp.where(diff_mask1 == 1, size=1, fill_value=0)[
            0
        ] - (self.w + 1)

        indices = jnp.arange(mask_final.shape[0])
        mask_cover_left = jnp.where(
            indices >= transition_index_right_mask1[0], False, mask_final
        )
        mask_cover_right = jnp.where(
            indices <= transition_index_left_mask1[0], False, mask_final
        )

        new_values_left = jnp.minimum(
            jnp.maximum(
                (phi_left * boundary_slice) - d_rho_left,
                0.95 * rho_v + 0.05 * rho_l,
            ),
            0.95 * rho_l + 0.05 * rho_v,
        )

        new_values_right = jnp.minimum(
            jnp.maximum(
                (phi_right * boundary_slice) - d_rho_right,
                0.95 * rho_v + 0.05 * rho_l,
            ),
            0.95 * rho_l + 0.05 * rho_v,
        )

        updated_slice = jnp.where(
            mask_cover_left, new_values_left, boundary_slice
        )
        updated_slice = jnp.where(mask_cover_right, new_values_right, updated_slice)

        # Set back to grid_padded
        if edge == "bottom":
            grid_padded = grid_padded.at[1:-1, 0].set(updated_slice)
        elif edge == "top":
            grid_padded = grid_padded.at[1:-1, -1].set(updated_slice)
        elif edge == "left":
            grid_padded = grid_padded.at[0, 1:-1].set(updated_slice)
        elif edge == "right":
            grid_padded = grid_padded.at[-1, 1:-1].set(updated_slice)
        return grid_padded

    def _apply_wetting_to_edge(self, grid: jnp.ndarray, edge: str, phi_left, phi_right, d_rho_left, d_rho_right) -> jnp.ndarray:
        grid_padded = self._pad_grid(grid, edge)
        grid_padded = self._apply_wetting_boundary(grid_padded, edge)
        grid_padded = self._apply_wetting_mask(grid_padded, edge, phi_left, phi_right, d_rho_left, d_rho_right)
        # Unflip if needed
        if edge == "top":
            grid_padded = jnp.flip(grid_padded, axis=1)
        elif edge == "right":
            grid_padded = jnp.flip(grid_padded, axis=0)
        # Unpad
        grid = grid_padded[1:-1, 1:-1]
        return grid

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

import jax.numpy as jnp
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice


class Equilibrium:
    """
    Callable class to calculate the equilibrium population distribution for LBM.
    Uses the provided feq implementation directly.
    """

    def __init__(self, grid: Grid, lattice: Lattice) -> None:
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.w = lattice.w
        self.cx = lattice.c[0]
        self.cy = lattice.c[1]

    def __call__(self, rho, u):
        """
        Calculate the equilibrium distribution function.

        Args:
            rho (jnp.ndarray): Density field.
            u (jnp.ndarray): Velocity field.

        Returns:
            jnp.ndarray: Equilibrium distribution function.
        """
        nx, ny, q = self.nx, self.ny, self.q
        w = self.w
        c = (self.cx, self.cy)
        f_eq = jnp.zeros((nx, ny, q, 1))
        w, cx, cy, ux, uy = w, c[0], c[1], u[0], u[1]
        f_eq = f_eq.at[:, :, 1, :].set(w[0] * rho[:, :, :, :] * (
                3 * (cx[1] * ux + cy[1] * uy) + 4.5 * (cx[1] * ux + cy[1] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 2].set(w[2] * rho[:, :, 0, :] * (
                3 * (cx[2] * ux + cy[2] * uy) + 4.5 * (cx[2] * ux + cy[2] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 3].set(w[3] * rho[:, :, 0, :] * (
                3 * (cx[3] * ux + cy[3] * uy) + 4.5 * (cx[3] * ux + cy[3] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 4].set(w[4] * rho[:, :, 0, :] * (
                3 * (cx[4] * ux + cy[4] * uy) + 4.5 * (cx[4] * ux + cy[4] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 5].set(w[5] * rho[:, :, 0, :] * (
                3 * (cx[5] * ux + cy[5] * uy) + 4.5 * (cx[5] * ux + cy[5] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 6].set(w[6] * rho[:, :, 0, :] * (
                3 * (cx[6] * ux + cy[6] * uy) + 4.5 * (cx[6] * ux + cy[6] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 7].set(w[7] * rho[:, :, 0, :] * (
                3 * (cx[7] * ux + cy[7] * uy) + 4.5 * (cx[7] * ux + cy[7] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 8].set(w[8] * rho[:, :, 0, :] * (
                3 * (cx[8] * ux + cy[8] * uy) + 4.5 * (cx[8] * ux + cy[8] * uy) ** 2 - 3 / 2 * (ux * ux + uy * uy)))
        f_eq = f_eq.at[:, :, 0].set(rho[:, :, 0, :] * (1 - w[0] * (3 / 2 * (ux * ux + uy * uy))))
        return f_eq

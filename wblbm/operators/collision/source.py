import jax.numpy as jnp
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice
from wblbm.operators.differential.gradient import Gradient
class SourceTerm:
    """
    Callable class to calculate the source term for the LBM equation.
    Uses the provided source_term implementation directly.
    """

    def __init__(self, grid: Grid, lattice: Lattice):
        self.nx: int = grid.nx
        self.ny: int = grid.ny
        self.q: int = lattice.q
        self.d: int = lattice.d
        self.w = lattice.w
        self.cx = lattice.c[0]
        self.cy = lattice.c[1]
        self.gradient = Gradient(lattice)

    def __call__(self, rho: jnp.ndarray, u: jnp.ndarray, force: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the source term for the LBM equation.

        Args:
            u (jnp.ndarray): Velocity field.
            force (jnp.ndarray): Force field.
            rho (jnp.ndarray): Density field.

        Returns:
            jnp.ndarray: Source term.
        """
        nx, ny, q = self.nx, self.ny, self.q
        w = self.w
        c = (self.cx, self.cy)
        d = self.d
        gradient = self.gradient

        def source_term(u_, force_, rho_):
            """
            Calculate the source term for the LBM equation.

            Args:
                u_ (jnp.ndarray): Velocity field.
                force_ (jnp.ndarray): Force field.
                rho_ (jnp.ndarray): Density field.

            Returns:
                jnp.ndarray: Source term.
            """
            cx, cy, fx, fy, ux, uy, grad_rho_x, grad_rho_y = (
                c[0],
                c[1],
                force_[0],
                force_[1],
                u_[0],
                u_[1],
                gradient(rho_)[0],
                gradient(rho_)[1]
            )
            fx_cor = fx + (grad_rho_x / 3)
            fy_cor = fy + (grad_rho_y / 3)
            source_ = jnp.zeros((nx, ny, q))
            source_ = source_.at[:, :, 0].set(w[0] * (3 * (cx[0] * fx + cy[0] * fy) +
                                                      9 * (cx[0] * fx_cor + cy[0] * fy_cor) * (cx[0] * ux + cy[0] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[0] * cx[0] + cy[0] * cy[0]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 1].set(w[1] * (3 * (cx[1] * fx + cy[1] * fy) +
                                                      9 * (cx[1] * fx_cor + cy[1] * fy_cor) * (cx[1] * ux + cy[1] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[1] * cx[1] + cy[1] * cy[1]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 2].set(w[2] * (3 * (cx[2] * fx + cy[2] * fy) +
                                                      9 * (cx[2] * fx_cor + cy[2] * fy_cor) * (cx[2] * ux + cy[2] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[2] * cx[2] + cy[2] * cy[2]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 3].set(w[3] * (3 * (cx[3] * fx + cy[3] * fy) +
                                                      9 * (cx[3] * fx_cor + cy[3] * fy_cor) * (cx[3] * ux + cy[3] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[3] * cx[3] + cy[3] * cy[3]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 4].set(w[4] * (3 * (cx[4] * fx + cy[4] * fy) +
                                                      9 * (cx[4] * fx_cor + cy[4] * fy_cor) * (cx[4] * ux + cy[4] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[4] * cx[4] + cy[4] * cy[4]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 5].set(w[5] * (3 * (cx[5] * fx + cy[5] * fy) +
                                                      9 * (cx[5] * fx_cor + cy[5] * fy_cor) * (cx[5] * ux + cy[5] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[5] * cx[5] + cy[5] * cy[5]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 6].set(w[6] * (3 * (cx[6] * fx + cy[6] * fy) +
                                                      9 * (cx[6] * fx_cor + cy[6] * fy_cor) * (cx[6] * ux + cy[6] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[6] * cx[6] + cy[6] * cy[6]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 7].set(w[7] * (3 * (cx[7] * fx + cy[7] * fy) +
                                                      9 * (cx[7] * fx_cor + cy[7] * fy_cor) * (cx[7] * ux + cy[7] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[7] * cx[7] + cy[7] * cy[7]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            source_ = source_.at[:, :, 8].set(w[8] * (3 * (cx[8] * fx + cy[8] * fy) +
                                                      9 * (cx[8] * fx_cor + cy[8] * fy_cor) * (cx[8] * ux + cy[8] * uy) -
                                                      3 * (ux * fx_cor + uy * fy_cor) +
                                                      .5 * (3 * (cx[8] * cx[8] + cy[8] * cy[8]) - d) * (
                                                              ux * grad_rho_x + uy * grad_rho_y)
                                                      ))
            return source_

        return source_term(u, force, rho)

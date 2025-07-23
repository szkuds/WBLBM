import jax.numpy as jnp
from .base import CollisionBase
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice

M = jnp.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=jnp.float64,
)
M_INV = jnp.linalg.inv(M)


class CollisionMRT(CollisionBase):
    def __init__(self, grid: Grid, lattice: Lattice, k_diag=None, **kwargs):
        super().__init__(grid, lattice)
        if k_diag is None:
            k_diag = jnp.array(
                [
                    1 / kwargs.get("k0", 1.0),
                    1 / kwargs.get("kb", 1.0),
                    1 / kwargs.get("k2", 1.0),
                    1 / kwargs.get("k0", 1.0),
                    1 / kwargs.get("k4", 1.0),
                    1 / kwargs.get("k0", 1.0),
                    1 / kwargs.get("k4", 1.0),
                    1 / kwargs.get("kv", 0.8),
                    1 / kwargs.get("kv", 0.8),
                ],
                dtype=jnp.float64,
            )
        self.K = k_diag

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        m = jnp.einsum("ij,xyj->xyi", M, f[..., 0])
        m_eq = jnp.einsum("ij,xyj->xyi", M, feq[..., 0])
        if source is None:
            S = 0.0
        else:
            S = jnp.einsum("ij,xyj->xyi", M, source[..., 0])
        m_post = m - self.K * (m - m_eq) + (1 - 0.5 * self.K) * S

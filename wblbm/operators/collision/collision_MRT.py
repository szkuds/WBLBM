import jax.numpy as jnp
from .base import CollisionBase
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice

# Moment transformation matrix for D2Q9 lattice
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
)
M_INV = jnp.linalg.inv(M)


class CollisionMRT(CollisionBase):
    """
    Implements the MRT (Multiple Relaxation Time) collision operator for LBM.
    """

    def __init__(self, grid: Grid, lattice: Lattice, k_diag=None, **kwargs):
        """
        Initialize the MRT collision operator.

        Args:
            grid (Grid): Grid object containing simulation domain information
            lattice (Lattice): Lattice object containing lattice properties
            k_diag (jnp.ndarray, optional): Diagonal relaxation rates for moments.
            kwargs: Optional relaxation rates for specific moments.
        """
        super().__init__(grid, lattice)
        if k_diag is None:
            k_diag = jnp.array(
                [
                    kwargs.get("k0", 0.0),
                    kwargs.get("kb", 1.0),
                    kwargs.get("k2", 1.0),
                    kwargs.get("k0", 0.0),
                    kwargs.get("k4", 1.0),
                    kwargs.get("k0", 0.0),
                    kwargs.get("k4", 1.0),
                    kwargs.get("kv", 0.8),
                    kwargs.get("kv", 0.8),
                ]
            )
        self.K = k_diag

    def __call__(
        self, f: jnp.ndarray, feq: jnp.ndarray, source: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Perform the MRT collision step.

        Args:
            f (jnp.ndarray): Distribution function.
            feq (jnp.ndarray): Equilibrium distribution function.
            source (jnp.ndarray, optional): Source term.

        Returns:
            jnp.ndarray: Post-collision distribution function.
        """
        # Transform to moment space
        m = jnp.einsum("ij,xyj->xyi", M, f[..., 0])
        m_eq = jnp.einsum("ij,xyj->xyi", M, feq[..., 0])
        if source is None:
            S = 0.0
        else:
            S = jnp.einsum("ij,xyj->xyi", M, source[..., 0])
        # Relaxation in moment space
        m_post = (
            m
            - self.K[jnp.newaxis, jnp.newaxis, ...] * (m - m_eq)
            + (1 - 0.5 * self.K[jnp.newaxis, jnp.newaxis, ...]) * S
        )
        # Transform back to distribution space
        f_post = jnp.einsum("ji,xyi->xyj", M_INV, m_post)[..., None]
        return f_post

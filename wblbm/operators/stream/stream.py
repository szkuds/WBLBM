import jax.numpy as jnp
from wblbm.lattice.lattice import Lattice


class Streaming:
    """
    Callable class to perform the streaming step of the LBM.
    """

    def __init__(self, lattice: Lattice):
        self.c = lattice.c  # Shape: (2, Q)
        self.q = lattice.q

    def __call__(self, fi_):
        """
        Perform the streaming step of the LBM.

        Args:
            fi_ (jnp.ndarray): Distribution function, shape (nx, ny, q)

        Returns:
            jnp.ndarray: Post-streaming distribution function.
        """
        for i in range(self.q):
            fi_ = fi_.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(fi_[:, :, i], self.c[0, i], axis=0),
                    self.c[1, i], axis=1))
        return fi_

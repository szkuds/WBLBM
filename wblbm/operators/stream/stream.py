import jax.numpy as jnp
from wblbm.lattice.lattice import Lattice
from wblbm.utils.timing import time_function


class Streaming:
    """
    Callable class to perform the streaming step of the LBM.
    """

    def __init__(self, lattice: Lattice):
        self.c = lattice.c  # Shape: (2, Q)
        self.q = lattice.q

    @time_function
    def __call__(self, f):
        """
        Perform the streaming step of the LBM.

        Args:
            f (jnp.ndarray): Distribution function, shape (nx, ny, q, 1)

        Returns:
            jnp.ndarray: Post-streaming distribution function.
        """
        for i in range(self.q):
            f = f.at[:, :, i, 0].set(
                jnp.roll(
                    jnp.roll(f[:, :, i, 0], self.c[0, i], axis=0),
                    self.c[1, i], axis=1))
        return f

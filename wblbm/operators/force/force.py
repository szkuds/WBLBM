from abc import ABC, abstractmethod
import jax.numpy as jnp


class Force(ABC):
    """
    Abstract base class for forces in the simulation.
    Holds a force array of shape (nx, ny, 1, d).
    """

    def __init__(self, force_array: jnp.ndarray):
        if force_array.ndim != 4 or force_array.shape[2] != 1:
            raise ValueError("Force array must have shape (nx, ny, 1, d)")
        self.force = force_array

    @abstractmethod
    def compute_force(self, *args, **kwargs) -> jnp.ndarray:
        """
        Compute the force field based on the density field rho.
        Must return an array of shape (nx, ny, 1, d).
        """
        pass

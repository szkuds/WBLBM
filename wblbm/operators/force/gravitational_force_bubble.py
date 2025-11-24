import jax.numpy as jnp
from wblbm.operators.force.force import Force

class GravityForceMultiphaseBubble(Force):
    """
    Subclass for gravitational force, when using a bubble.
    Note that the force_x contains the (-) which ensures that the return force has a positive x-component
    """

    def __init__(
        self, nx: int, ny: int, d: int, force_g: float, inclination_angle_deg: float, rho_ref: float
    ):
        if d != 2:
            raise ValueError("Currently supports 2D (d=2) only")

        force_x = force_g * -jnp.sin(jnp.deg2rad(inclination_angle_deg))
        force_y = force_g * jnp.cos(jnp.deg2rad(inclination_angle_deg))

        force_array = jnp.zeros((nx, ny, 1, d))
        force_array = force_array.at[:, :, 0, 0].set(force_x)
        force_array = force_array.at[:, :, 0, 1].set(force_y)
        self.rho_ref = rho_ref

        # This ensures that the array has the correct size
        super().__init__(force_array)

    def compute_force(
        self, rho: jnp.ndarray, rho_l: float, rho_v: float
    ) -> jnp.ndarray:
        """
        Returns the constant gravitational force field.
        Ignores rho as gravity is density-independent.
        """
        return - self.force * (rho - self.rho_ref)

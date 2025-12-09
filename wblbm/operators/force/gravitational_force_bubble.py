import jax.numpy as jnp
from wblbm.operators.force.force import Force

class GravityForceMultiphaseBubble(Force):
    """
    Subclass for gravitational force, when using a bubble.
    Note that the force_x contains the (-) which ensures that the return force has a positive x-component
    """

    def __init__(
        self, force_g: float, inclination_angle_deg: float, grid_shape: tuple
    ):
        if grid_shape.__len__() != 2:
            raise ValueError("Currently supports 2D (d=2) only")

        force_x = force_g * -jnp.sin(jnp.deg2rad(inclination_angle_deg))
        force_y = force_g * jnp.cos(jnp.deg2rad(inclination_angle_deg))

        force_array = jnp.zeros((grid_shape[0], grid_shape[1], 1, grid_shape.__len__()))
        force_array = force_array.at[:, :, 0, 0].set(force_x)
        force_array = force_array.at[:, :, 0, 1].set(force_y)

        # This ensures that the array has the correct size
        super().__init__(force_array)

    def compute_force(self, **kwargs) -> jnp.ndarray:
        rho = kwargs.get('rho')
        rho_max = jnp.max(rho)
        if rho is None:
            raise ValueError("GravityForceMultiphaseBubble requires 'rho' in kwargs")
        return -self.force * (rho - rho_max)

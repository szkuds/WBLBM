import jax.numpy as jnp
from wblbm.operators.force.force import Force


class GravityForceMultiphaseDroplet(Force):
    """
    Subclass for gravitational force, constant across the grid.
    """

    def __init__(
        self, nx: int, ny: int, d: int, force_g: float, inclination_angle_deg: float
    ):
        if d != 2:
            raise ValueError("Currently supports 2D (d=2) only")

        force_x = force_g * jnp.sin(jnp.deg2rad(inclination_angle_deg))
        force_y = force_g * -jnp.cos(jnp.deg2rad(inclination_angle_deg))

        force_array = jnp.zeros((nx, ny, 1, d))
        force_array = force_array.at[:, :, 0, 0].set(force_x)
        force_array = force_array.at[:, :, 0, 1].set(force_y)

        super().__init__(force_array)

    def compute_force(
        self, rho: jnp.ndarray, rho_l: float, rho_v: float
    ) -> jnp.ndarray:
        """
        Returns the constant gravitational force field.
        Ignores rho as gravity is density-independent.
        """
        mask = rho > 0.95 * rho_v + 0.05 * rho_l
        return self.force * rho * mask

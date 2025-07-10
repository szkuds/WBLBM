import jax.numpy as jnp
import numpy as np
from wblbm.grid.grid import Grid
from wblbm.lattice.lattice import Lattice
from wblbm.operators.equilibrium.equilibirum import Equilibrium


class Initialise:
    """
    Handles the initialisation of the simulation for various scenarios.
    """

    def __init__(self, grid: Grid, lattice: Lattice):
        self.grid = grid
        self.lattice = lattice
        self.nx, self.ny = grid.nx, grid.ny
        self.q = self.lattice.q
        self.equilibrium = Equilibrium(self.grid, self.lattice)

    def initialise_standard(self, density: float = 1.0, velocity: np.ndarray = np.array([0.0, 0.0])):
        """
        Initialises a standard simulation with uniform density and velocity.

        Args:
            density (float): Initial uniform density.
            velocity (np.ndarray): Initial uniform velocity [ux, uy].

        Returns:
            jnp.ndarray: Initialised population distribution f.
        """
        # Create density and velocity fields with the correct 4D shape
        rho = jnp.full((self.nx, self.ny, 1, 1), density)

        u = jnp.broadcast_to(
            jnp.array(velocity).reshape(1, 1, 1, 2),
            (self.nx, self.ny, 1, 2)
        )

        # Return the equilibrium distribution for this state
        return self.equilibrium(rho, u)

    def initialise_multiphase_droplet(self, rho_l: float, rho_v: float, interface_width: int):
        """
        Initialises a multiphase simulation with a low-density bubble in the center.

        Args:
            rho_l (float): Liquid phase density.
            rho_v (float): Vapour phase (bubble) density.

        Returns:
            jnp.ndarray: Initialised population distribution f.
        """
        # Create a density field with a bubble in the center
        x, y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing='ij')
        center_x, center_y = self.nx // 2, self.ny // 2
        radius = min(self.nx, self.ny) // 4

        # Use tanh for a smooth, stable interface
        distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        rho_field_2d = (rho_l + rho_v) / 2 - (rho_l - rho_v) / 2 * jnp.tanh((distance - radius) / interface_width)

        # Reshape to 4D
        rho = rho_field_2d.reshape((self.nx, self.ny, 1, 1))

        # Initialise with zero velocity
        u = jnp.zeros((self.nx, self.ny, 1, 2))

        # Return the equilibrium distribution
        return self.equilibrium(rho, u)

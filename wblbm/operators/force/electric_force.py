import jax.numpy as jnp
from wblbm.operators.force.force import Force


class ElectricForce(Force):
    """
    Subclass for electrical force with electric potential distribution.
    Solves for electric potential using a separate distribution function h_i.
    """

    def __init__(self, nx: int, ny: int, d: int, permittivity_liquid: float, permittivity_vapour: float,
                 conductivity_liquid: float, conductivity_vapour: float):
        """
        Initialize electrical force.

        Args:
            nx: Grid size in x direction
            ny: Grid size in y direction
            d: Dimensionality (must be 2)
            permittivity_liquid: Electric permittivity of the liquid phase
            permittivity_vapour: Electric permittivity of the vapour phase
            conductivity_liquid: Electrical conductivity of the liquid phase
            conductivity_vapour: Electrical conductivity of the vapour phase
        """
        if d != 2:
            raise ValueError("Currently supports 2D (d=2) only")

        self.name = 'ElectricalForce'

        force_array = jnp.zeros((nx, ny, 1, d))
        super().__init__(force_array)
        self.permittivity_liquid = permittivity_liquid
        self.permittivity_vapour = permittivity_vapour
        self.conductivity_liquid = conductivity_liquid
        self.conductivity_vapour = conductivity_vapour
        self.nx = nx
        self.ny = ny
        self.d = d

        # Initialize h_i distribution function for electric potential
        # 9 velocities for D2Q9 lattice
        self.h_i = jnp.zeros((nx, ny, 9))
        self.U = jnp.zeros((nx, ny))  # Electric potential field

    def compute_force(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        Compute electrical force from electric field gradient.
        U = sum_i h_i,
        E = -∇U,
        F = q * E - .5 * E^2 * ∇ ϵ

        Args:
            rho: Density field of shape (nx, ny, 1)

        Returns:
            Force array of shape (nx, ny, 1, 2)
        """

        return electric_force

    def update_potential(self, h_i: jnp.ndarray) -> None:
        """
        Update electric potential from h_i distribution.
        h_i = Σ h_i

        Args:
            h_i: Distribution function for electric potential, shape (nx, ny, q)
        """
        self.h_i = h_i
        self.U = jnp.sum(h_i, axis=2)

    def equilibrium_h(self, h_i: jnp.ndarray, w_i: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium distribution for electric potential.
        h_i^eq = w_i * U

        Args:
            h_i: Electric potential field, shape (nx, ny, q)
            w_i: Lattice weights, shape (9,)

        Returns:
            Equilibrium distribution, shape (nx, ny, 9)
        """
        return w_i[:, jnp.newaxis, jnp.newaxis] * h_i

    def conductivity(self, rho: jnp.ndarray):
        pass

import jax.numpy as jnp

from wblbm import Gradient
from wblbm.operators.force.force import Force
from wblbm.lattice import Lattice

from numpy import ndarray
from wblbm.operators.wetting.wetting_util import determine_padding_modes


class ElectricForce(Force):
    """
    Subclass for electrical force with electric potential distribution.
    Solves for electric potential using a separate distribution function h_i_prev.
    """

    def __init__(self, permittivity_liquid: float, permittivity_vapour: float,
                 conductivity_liquid: float, conductivity_vapour: float,
                 grid_shape: tuple, lattice_type: str, bc_config: dict = None):
        """
        Initialize electrical force.

        Args:
            Grid contains:
            - nx: Grid size in x direction
            - ny: Grid size in y direction
            - d: Dimensionality (must be 2)
            permittivity_liquid: Electric permittivity of the liquid phase
            permittivity_vapour: Electric permittivity of the vapour phase
            conductivity_liquid: Electrical conductivity of the liquid phase
            conductivity_vapour: Electrical conductivity of the vapour phase
        """
        if grid_shape.__len__() != 2:
            raise ValueError("Currently supports 2D (d=2) only")

        self.name = 'ElectricalForce'

        force_array = jnp.zeros((grid_shape[0], grid_shape[1], 1, grid_shape.__len__()))
        super().__init__(force_array)
        self.permittivity_liquid = permittivity_liquid
        self.permittivity_vapour = permittivity_vapour
        self.conductivity_liquid = conductivity_liquid
        self.conductivity_vapour = conductivity_vapour
        self.nx = grid_shape[0]
        self.ny = grid_shape[1]
        self.d = grid_shape.__len__()
        self.lattice = Lattice(lattice_type)
        self.gradient = Gradient(self.lattice, bc_config=bc_config)
        self.bc_config = bc_config

    def compute_force(self, rho: jnp.ndarray, h_i: jnp.ndarray) -> jnp.ndarray:
        """
        Compute electrical force from electric field gradient.
        U = sum_i h_i_prev,
        E = -∇U,
        F = q * E - .5 * E^2 * ∇ ϵ
        q = ∇*(ϵE)

        Args:
            rho: Density field of shape (nx, ny, 1, 1)
            h_i: Distribution for potential (nx, ny, q, 1)

        Returns:
            Force array of shape (nx, ny, 1, 2)
        """
        conductivity_field = self.conductivity(rho,
                                               conductivity_liquid=self.conductivity_liquid,
                                               conductivity_vapour=self.conductivity_vapour)
        permittivity_field = self.permittivity(rho,
                                               permittivity_liquid=self.permittivity_liquid,
                                               permittivity_vapour=self.permittivity_vapour)
        potential = self.update_potential(h_i)
        electric_field = self.gradient(potential)
        # TODO: This is ugly here would be better to make function within the gradient class to do this operation.
        # TODO: Apart from the divergence it is also good to make the standard gradient more accessible, really the density with wetting is the exception and not the rule/
        eE_x = (permittivity_field*electric_field)[:,:,:,0]
        eE_y = (permittivity_field*electric_field)[:,:,:,1]
        eE_x_grad_x = self.gradient._gradient_standard(eE_x, determine_padding_modes(self.bc_config))[:,:,:,0]
        eE_y_grad_y = self.gradient._gradient_standard(eE_y, determine_padding_modes(self.bc_config))[:,:,:,1]
        q = eE_x_grad_x + eE_y_grad_y
        electric_force = q*electric_field - 0.5 * jnp.dot(electric_field, electric_field) * self.gradient._gradient_standard(permittivity_field, determine_padding_modes(self.bc_config))
        return electric_force

    def update_potential(self, h_i: jnp.ndarray) -> jnp.ndarray:
        """
        Update electric potential from h_i distribution.
        h_i = Σ h_i

        Args:
            h_i_prev: Distribution function for electric potential, shape (nx, ny, q)
        """
        U = jnp.sum(h_i, axis=2)
        return U

    def update_h_i(self, h_i_prev: jnp.ndarray, conductivity: jnp.ndarray):
        h_i_eq = self.equilibrium_h(h_i_prev, self.lattice.w)
        h_i_next = self
        # TODO: This is where I get stuck late, thing is that in the current bgk collision tau is initlised as a float, which will need to be changed.
        return h_i_next


    def equilibrium_h(self, h_i: jnp.ndarray, w_i: ndarray) -> jnp.ndarray:
        """
        Equilibrium distribution for electric potential.
        h_i_prev^eq = w_i * U

        Args:
            h_i: Electric potential field, shape (nx, ny, q)
            w_i: Lattice weights, shape (9,)

        Returns:
            Equilibrium distribution, shape (nx, ny, 9)
        """
        return w_i[:, jnp.newaxis, jnp.newaxis] * h_i

    def conductivity(self, rho: jnp.ndarray, conductivity_liquid: float, conductivity_vapour: float) -> jnp.ndarray:
        return self.rho_to_phi(rho, conductivity_liquid, conductivity_vapour)

    def permittivity(self, rho: jnp.ndarray, permittivity_liquid: float, permittivity_vapour: float) -> jnp.ndarray:
        return self.rho_to_phi(rho, permittivity_liquid, permittivity_vapour)

    def rho_to_phi(self, rho: jnp.ndarray, phi_liquid: float, phi_gas: float) -> jnp.ndarray:
        rho_max = jnp.max(rho)
        rho_min = jnp.min(rho)
        phi = ((rho - rho_min) / (rho_max - rho_min)) * phi_liquid + (
                    1 - ((rho - rho_min) / (rho_max - rho_min))) * phi_gas
        return phi

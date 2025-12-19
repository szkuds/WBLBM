import jax.numpy as jnp
from fontTools.misc.bezierTools import epsilon

from wblbm import Gradient, Streaming
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
        self.stream = Streaming(self.lattice)

    def compute_force(self, **kwargs) -> jnp.ndarray:
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
        rho = kwargs.get('rho')
        h_i = kwargs.get('h_i')
        if rho is None or h_i is None:
            raise ValueError("ElectricForce requires 'rho' and 'h_i' in kwargs")
        conductivity_field = self.conductivity(rho,
                                               conductivity_liquid=self.conductivity_liquid,
                                               conductivity_vapour=self.conductivity_vapour)
        permittivity = self.permittivity(rho,
                                         permittivity_liquid=self.permittivity_liquid,
                                         permittivity_vapour=self.permittivity_vapour)
        potential = self.electric_potential(h_i)
        electric_field_ = jnp.stack(jnp.gradient(-potential[:,:,0,0]), axis=-1)
        electric_field = electric_field_[:,:,jnp.newaxis, :]
        # TODO: This is ugly here would be better to make function within the gradient class to do this operation.
        # TODO: Apart from the divergence it is also good to make the standard gradient more accessible, really the density with wetting is the exception and not the rule/
        eE_x = jnp.zeros(rho.shape)
        eE_y = jnp.zeros(rho.shape)
        q = jnp.zeros(rho.shape)
        eE_x = eE_x.at[:, :, :, 0].set((permittivity * electric_field)[:, :, :, 0])
        eE_y = eE_y.at[:, :, :, 1].set((permittivity * electric_field)[:, :, :, 1])
        eE_x_grad_x = self.gradient.standard(eE_x, determine_padding_modes(self.bc_config))[:, :, :, 0]
        eE_y_grad_y = self.gradient.standard(eE_y, determine_padding_modes(self.bc_config))[:, :, :, 1]
        q = q.at[:, :, :, 0].set(eE_x_grad_x + eE_y_grad_y)
        EE = jnp.expand_dims(jnp.einsum('...i,...i->...', electric_field, electric_field), axis=-1)
        electric_force = q * electric_field - 0.5 * EE * self.gradient.standard(
            permittivity, determine_padding_modes(self.bc_config))
        return electric_force

    def electric_potential(self, h_i: jnp.ndarray) -> jnp.ndarray:
        """
        Update electric potential from h_i distribution.
        h_i = Σ h_i

        Args:
            h_i_prev: Distribution function for electric potential, shape (nx, ny, q)
        """
        U = jnp.sum(h_i, axis=2, keepdims=True)
        return U

    # def boundary_condition(self, f_col, U_0):
    #      f_bc = jnp.zeros_like(f_col)
    #      f_pad_1 = jnp.pad(f_col, 1, mode='wrap')
    #      f_bc = f_bc.at[:, :, :, 0].set(f_col[:, :, :, 0])
    #      f_bc = f_bc.at[0, :, self.lattice.left_indices, 0].set(0)
    #      f_bc = f_bc.at[-1, :, 1, 0].set(U_0 * self.lattice.w[1])
    #      f_bc = f_bc.at[-1, :, 5, 0].set(U_0 * self.lattice.w[5])
    #      f_bc = f_bc.at[-1, :, 8, 0].set(U_0 * self.lattice.w[8])
    #      f_bc = f_bc.at[:, 0, 4, 0].set(f_col[:, -1, 2, 0])
    #      f_bc = f_bc.at[:, 0, 7, 0].set(f_pad_1[:self.nx, -1, 5, 0])
    #      f_bc = f_bc.at[:, 0, 8, 0].set(f_pad_1[2:, -1, 6, 0])
    #      f_bc = f_bc.at[:, -1, 2, 0].set(f_col[:, 0, 4, 0])
    #      f_bc = f_bc.at[:, -1, 5, 0].set(f_pad_1[2:, 0, 7, 0])
    #      f_bc = f_bc.at[:, -1, 6, 0].set(f_pad_1[:self.nx, 0, 8, 0])
    #      return f_bc

    def boundary_condition(self, f_col, U_0) -> jnp.ndarray:

        grid_pad_ = jnp.pad(f_col, ((0, 0), (1, 1),  (0, 0),  (0, 0)), mode='edge')
        grid_pad = jnp.pad(grid_pad_, ((1, 1), (0, 0), (0, 0), (0, 0)), mode='empty')
        grid_pad = grid_pad.at[0,:,:,:].set(self.equilibrium_h(U_0, self.lattice.w)[0,:,:,:])
        return grid_pad

    def update_h_i(self, h_i_prev: jnp.ndarray, conductivity: jnp.ndarray):
        h_i_eq = self.equilibrium_h(h_i_prev, self.lattice.w)
        tau_e = 3 * conductivity + .5
        h_i_col = (1 - (1 / tau_e)) * h_i_prev - (1 / tau_e) * h_i_eq
        h_i_bc = self.boundary_condition(h_i_col, U_0=1e-2)
        h_i_next = self.stream(h_i_bc)[1:-1, 1:-1, :,  :]
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
        return w_i[jnp.newaxis, jnp.newaxis, :, jnp.newaxis] * h_i

    def conductivity(self, rho: jnp.ndarray, conductivity_liquid: float, conductivity_vapour: float) -> jnp.ndarray:
        return self.rho_to_phi(rho, conductivity_liquid, conductivity_vapour)

    def permittivity(self, rho: jnp.ndarray, permittivity_liquid: float, permittivity_vapour: float) -> jnp.ndarray:
        return self.rho_to_phi(rho, permittivity_liquid, permittivity_vapour)

    def phi(self, rho: jnp.ndarray) -> jnp.ndarray:
        return self.rho_to_phi(rho, phi_liquid=-1, phi_gas=1)

    def rho_to_phi(self, rho: jnp.ndarray, phi_liquid: float, phi_gas: float) -> jnp.ndarray:
        rho_max = jnp.max(rho)
        rho_min = jnp.min(rho)
        phi = ((rho - rho_min) / (rho_max - rho_min)) * phi_liquid + (
                1 - ((rho - rho_min) / (rho_max - rho_min))) * phi_gas
        return phi

    def init_h(self):
        h_i = jnp.zeros((self.nx, self.ny, self.lattice.q, 1))
        return h_i

#     def init_h(self, U_0=1e-3):
#         '''Initialize electric potential distribution h_i with potential difference U0 to 0 from left to right'''
#         h_i = jnp.zeros((self.nx, self.ny, self.lattice.q, 1))
#
#         #Linear potential left to right
#         # U_linear = jnp.linspace(0, U_0, self.nx).reshape(self.nx,1)
#         # U = jnp.tile(U_linear, (1, self.ny)) # shape (nx, ny)
#         # grid_pad__ = jnp.pad(h_i, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='edge')
#         # grid_pad_ = jnp.pad(grid_pad__, ((1, 1), (0, 0), (0, 0), (0, 0)), mode='empty')
#         # grid_pad = grid_pad_.at[0, :, :, :].set(self.equilibrium_h(U_0, self.lattice.w)[0, :, :, :])
#         # return grid_pad
#
#         #
#         # #Convert potential U into distribution h_i using lattice weights
#         # for i in range(self.lattice.q):
#         #     h_i = h_i.at[:,:,i,0].set(self.lattice.w[i]*U[:,:])
#
#         return h_i
#
# #    def init_h(self, Ca_E=1e3, sigma=0.1, R=30):
#         ''' Initialise h_i from electric capillary number Ca_E'''
#         # Use Ca_E: Electric capillary number, sigma: surface tension (maybe kappa?), R= droplet radius (maybe get from radius in initialise_wetting class)
#         E0 = jnp.sqrt((Ca_E * sigma)/(self.permittivity_liquid*R))
#
#         #Linear potential over domain
#         x = jnp.linspace(0,1,self.nx).reshape(self.nx,1,1) #domain length
#         U = E0*x #potential field, shape (nx, 1,1)
#         U = jnp.tile(U, (1,self.ny,1)) #shape (nx,ny,1)
#
#         #Convert potential to distribution h_i
#         w_i = self.lattice.w.reshape(1,1,self.lattice.q,1)
#         h_i = w_i*U.reshape(self.nx, self.ny, 1,1)
#
#         return h_i
#
# #    def init_h(self, Ca_e=0.1):
#         '''Initialize h_i from U0, which is determined by the dimensionless electric capillary number Ca_e, surface tension gamma and other constants'''
#         #For now manually inputting constants here to ensure it works, later want to have them imported from outside the function (using self.)
#         h_i = jnp.zeros((self.nx, self.ny, self.lattice.q, 1))
#
#         #Define constants
#         epsilon = 0.01 #electric permeability of the fluid
#         r = self.ny/3.3 #radius of the droplet
#         l = self.nx #domain width, also known as nx
#
#         kappa = 0.04 #free-energy surface tension parameter
#         #w = width of the droplet --> CHECK, now using radius R
#         rho_l = 1.0 #density of the liquid
#         rho_v = 0.001 #density of the vapour
#
#         #Define surface tension from free energy theory
#         gamma = 2/3*kappa/r*jnp.abs(rho_l-rho_v)**2
#
#         #Calculate U0
#         U0 = jnp.sqrt(Ca_e*gamma/epsilon*l**2/r)
#
#         #Uniform electric field E = U0/L
#         x = jnp.arange(self.nx).reshape(self.nx, 1, 1, 1)
#         U = U0*(1-x/l) # linearly decreasing potential from left to right (U0 to 0)
#         U = jnp.broadcast_to(U, (self.nx, self.ny, 1, 1)) #extend the 1D potential across the y direction
#
#         #Distribute potential to h_i
#         w_i = self.lattice.w #lattice weights w_i
#         h_i = w_i[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]*U #h_i = w_i*U
#
#         return h_i
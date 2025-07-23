from .base import BaseSimulation
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase
from wblbm.operators.initialise.init import Initialise
import jax.numpy as jnp


class MultiphaseSimulation(BaseSimulation):
    def __init__(
        self,
        grid_shape,
        lattice_type="D2Q9",
        tau=1.0,
        nt=1000,
        kappa=0.1,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=4,
        force_enabled=False,
        force_obj=None,
        bc_config=None,
        collision_scheme="bgk",
        k_diag=None,
        **kwargs
    ):
        super().__init__(grid_shape, lattice_type, tau, nt)
        self.kappa = kappa
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.interface_width = interface_width
        self.force_enabled = force_enabled
        self.force_obj = force_obj
        self.bc_config = bc_config
        self.collision_scheme = collision_scheme
        self.k_diag = k_diag
        self.setup_operators()
        self.multiphase = True

    def setup_operators(self):
        self.initialiser = Initialise(self.grid, self.lattice)
        self.update = UpdateMultiphase(
            self.grid,
            self.lattice,
            self.tau,
            self.kappa,
            self.interface_width,
            self.rho_l,
            self.rho_v,
            self.bc_config,
            self.force_enabled,
            collision_scheme=self.collision_scheme,
            kvec=self.k_diag,
        )
        self.macroscopic = MacroscopicMultiphase(
            self.grid,
            self.lattice,
            self.kappa,
            self.interface_width,
            self.rho_l,
            self.rho_v,
            self.force_enabled,
        )

    def initialize_fields(self, init_type="multiphase_droplet"):
        if init_type == "multiphase_droplet":
            return self.initialiser.initialise_multiphase_droplet(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble":
            return self.initialiser.initialise_multiphase_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        else:
            return self.initialiser.initialise_standard()

    def run_timestep(self, fprev, it):
        force_ext = None
        if self.force_enabled and self.force_obj:
            rho = jnp.sum(fprev, axis=2, keepdims=True)
            force_ext = self.force_obj.compute_force(rho, self.rho_l, self.rho_v)
        fnext = (
            self.update(fprev, force=force_ext)
            if self.force_enabled
            else self.update(fprev)
        )

        return fnext

from .base import BaseSimulation
from wblbm.operators.update.update import Update
from wblbm.operators.macroscopic.macroscopic import Macroscopic
from wblbm.operators.initialise.init import Initialise
import jax.numpy as jnp


class SinglePhaseSimulation(BaseSimulation):
    def __init__(
        self,
        grid_shape,
        lattice_type="D2Q9",
        tau=1.0,
        nt=1000,
        force_enabled=False,
        force_obj=None,
        bc_config=None,
        collision_scheme="bgk",
        k_diag=None,
        **kwargs
    ):
        super().__init__(grid_shape, lattice_type, tau, nt)

        # Set simulation type flags
        self.macroscopic = None
        self.initialiser = None
        self.update = None
        self.multiphase = False
        self.wetting_enabled = False

        self.force_enabled = force_enabled
        self.force_obj = force_obj
        self.bc_config = bc_config
        self.collision_scheme = collision_scheme
        self.k_diag = k_diag
        self.kwargs = kwargs
        self.setup_operators()

    def setup_operators(self):
        self.initialiser = Initialise(self.grid, self.lattice)
        self.update = Update(
            self.grid,
            self.lattice,
            self.tau,
            bc_config=self.bc_config,
            force_enabled=self.force_enabled,
            collision_scheme=self.collision_scheme,
            kvec=self.k_diag,
            **self.kwargs
        )
        self.macroscopic = self.update.macroscopic
        if self.bc_config:
            from wblbm.operators.boundary_condition.boundary_condition import (
                BoundaryCondition,
            )

            self.boundary_condition = BoundaryCondition(
                self.grid, self.lattice, self.bc_config
            )

    def initialize_fields(self, init_type="standard", *, init_dir=None):
        if init_type == "init_from_file":
            if init_dir is None:
                raise ValueError(
                    "init_from_file requires init_dir pointing to a .npz file"
                )
            return self.initialiser.init_from_npz(init_dir)
        # existing options preserved
        return self.initialiser.initialise_standard()

    def run_timestep(self, fprev, it):
        force_ext = None
        if self.force_enabled and self.force_obj:
            rho = jnp.sum(fprev, axis=2, keepdims=True)
            force_ext = self.force_obj.compute_force(rho)
        fnext = (
            self.update(fprev, force=force_ext)
            if self.force_enabled
            else self.update(fprev)
        )
        return fnext

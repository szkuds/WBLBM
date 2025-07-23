from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.collision.collision_BGK import CollisionBGK
from wblbm.operators.collision.collision_MRT import CollisionMRT
from wblbm.operators.update.update import Update
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase
from wblbm.operators.boundary_condition.boundary_condition import BoundaryCondition


class UpdateMultiphase(Update):
    def __init__(
        self,
        grid: Grid,
        lattice: Lattice,
        tau: float,
        kappa: float,
        interface_width: int,
        rho_l: float,
        rho_v: float,
        bc_config: dict = None,
        force_enabled: bool = False,
        collision_scheme: str = "bgk",
        kvec=None,
    ):
        super().__init__(grid, lattice, tau, bc_config, force_enabled=force_enabled)
        self.macroscopic = MacroscopicMultiphase(
            grid,
            lattice,
            kappa,
            interface_width,
            rho_l,
            rho_v,
            force_enabled=force_enabled,
        )

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.array, force: jnp.ndarray = None):
        # If force_enabled and no force provided, use a simple constant force for testing
        if self.force_enabled and force is None:
            raise TypeError(
                "When the force is enabled an external force needs to be provided"
            )
        elif self.force_enabled:
            rho, u, force_tot = self.macroscopic(f, force=force)
        else:
            rho, u, force_tot = self.macroscopic(
                f
            )  # In this case the total force is only the interaction force
        feq = self.equilibrium(rho, u)
        source = self.source_term(rho, u, force_tot)
        fcol = self.collision(f, feq, source)
        fstream = self.streaming(fcol)
        if self.boundary_condition is not None:
            fbc = self.boundary_condition(fstream, fcol)
            return fbc
        else:
            return fstream

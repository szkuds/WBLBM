from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.boundary_condition.boundary_condition import BoundaryCondition
from wblbm.operators.collision import CollisionBGK, CollisionMRT, SourceTerm
from wblbm.operators.equilibrium.equilibrium import Equilibrium
from wblbm.operators.macroscopic.macroscopic import Macroscopic
from wblbm.operators.stream import Streaming
from wblbm.utils.timing import time_function, TIMING_ENABLED


class Update(object):
    def __init__(
        self,
        grid: Grid,
        lattice: Lattice,
        tau: float,
        bc_config: dict = None,
        force_enabled: bool = False,
        collision_scheme: str = "bgk",
        k_diag=None,
        **kwargs
    ):
        self.grid = grid
        self.lattice = lattice
        self.tau = tau
        self.macroscopic = Macroscopic(grid, lattice, force_enabled=force_enabled)
        self.equilibrium = Equilibrium(grid, lattice)
        # Select collision scheme
        if collision_scheme == "mrt":
            # Extract MRT parameters from kwargs if provided
            mrt_params = {}
            for param in ["k0", "kb", "k2", "k4", "kv"]:
                if param in kwargs:
                    mrt_params[param] = kwargs[param]
            self.collision = CollisionMRT(grid, lattice, k_diag=k_diag, **mrt_params)
        else:
            self.collision = CollisionBGK(grid, lattice, tau)
        self.source_term = SourceTerm(grid, lattice, bc_config)
        self.streaming = Streaming(lattice)
        if bc_config is not None:
            self.boundary_condition = BoundaryCondition(grid, lattice, bc_config)
        else:
            self.boundary_condition = None
        self.force_enabled = force_enabled

    @partial(jit, static_argnums=(0,))
    @time_function(enable_timing=TIMING_ENABLED)
    def __call__(self, f: jnp.ndarray, force: jnp.ndarray = None):
        if self.force_enabled:
            rho, u, force_tot = self.macroscopic(f, force=force)

            # Calculate source term and pass it to collision
            feq = self.equilibrium(rho, u)
            source = self.source_term(rho, u, force_tot)
            fcol = self.collision(f, feq, source)
        else:
            rho, u = self.macroscopic(f)
            feq = self.equilibrium(rho, u)
            fcol = self.collision(f, feq)

        fstream = self.streaming(fcol)
        if self.boundary_condition is not None:
            fbc = self.boundary_condition(fstream, fcol)
            return fbc
        else:
            return fstream

from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.boundary_condition.boundary_condition import BoundaryCondition
from wblbm.operators.collision import Collision, SourceTerm
from wblbm.operators.equilibrium.equilibirum import Equilibrium
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
            force_enabled: bool = False
    ):
        self.grid = grid
        self.lattice = lattice
        self.tau = tau
        self.macroscopic = Macroscopic(grid, lattice, force_enabled=force_enabled)
        self.equilibrium = Equilibrium(grid, lattice)
        self.collision = Collision(grid, lattice, tau)
        self.source_term = SourceTerm(grid, lattice)
        self.streaming = Streaming(lattice)
        if bc_config is not None:
            self.boundary_condition = BoundaryCondition(grid, lattice, bc_config)
        else:
            self.boundary_condition = None
        self.force_enabled = force_enabled

    @partial(jit, static_argnums=(0,))
    @time_function(enable_timing=TIMING_ENABLED)
    def __call__(self, f: jnp.ndarray, force: jnp.ndarray = None):
        # If force_enabled and no force provided, use a simple constant force for testing
        if self.force_enabled and force is None:
            # Example: uniform force in x-direction
            force = jnp.ones((self.grid.nx, self.grid.ny, 1, 2)) * jnp.array([0.01, 0.0])
        rho, u = self.macroscopic(f, force=force) if self.force_enabled else self.macroscopic(f)
        feq = self.equilibrium(rho, u)
        fcol = self.collision(f, feq)
        fstream = self.streaming(fcol)
        if self.boundary_condition is not None:
            fstream = self.boundary_condition(fstream, fcol)
        return fstream

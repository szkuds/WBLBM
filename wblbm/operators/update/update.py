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
            bc_config: dict = None
    ):
        self.grid = grid
        self.lattice = lattice
        self.tau = tau
        self.macroscopic = Macroscopic(grid, lattice)
        self.equilibrium = Equilibrium(grid, lattice)
        self.collision = Collision(grid, lattice, tau)
        self.source_term = SourceTerm(grid, lattice)
        self.streaming = Streaming(lattice)
        if bc_config is not None:
            self.boundary_condition = BoundaryCondition(grid, lattice, bc_config)
        else:
            self.boundary_condition = None

    @partial(jit, static_argnums=(0,))
    @time_function(enable_timing=TIMING_ENABLED)
    def __call__(self, f: jnp.ndarray):
        rho, u = self.macroscopic(f)
        feq = self.equilibrium(rho, u)
        fcol = self.collision(f, feq)
        fstream = self.streaming(fcol)
        if self.boundary_condition is not None:
            fstream = self.boundary_condition(fstream, fcol)
        return fstream

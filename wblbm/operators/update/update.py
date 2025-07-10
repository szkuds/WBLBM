from functools import partial

import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.collision import Collision, SourceTerm
from wblbm.operators.equilibrium.equilibirum import Equilibrium
from wblbm.operators.macroscopic.macroscopic import Macroscopic
from wblbm.operators.stream import Streaming


class Update(object):
    def __init__(
            self,
            grid: Grid,
            lattice: Lattice,
            tau: float
    ):
        self.grid = grid
        self.lattice = lattice
        self.tau = tau
        self.macroscopic = Macroscopic(grid, lattice)
        self.equilibrium = Equilibrium(grid, lattice)
        self.collision = Collision(grid, lattice, tau)
        self.source_term = SourceTerm(grid, lattice)
        self.streaming = Streaming(lattice)

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.ndarray):
        rho, u = self.macroscopic(f)
        feq = self.equilibrium(rho, u)
        fcol = self.collision(f, feq)
        fstream = self.streaming(fcol)
        return fstream

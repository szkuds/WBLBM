import jax.numpy as jnp

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.collision.collision_multiphase import CollisionMultiphase
from wblbm.operators.update.update import Update
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase


class UpdateMultiphase(Update):
    def __init__(
            self,
            grid: Grid,
            lattice: Lattice,
            tau: float,
            kappa: float,
            interface_width: int,
            rho_l: float,
            rho_v: float
    ):
        super().__init__(grid, lattice, tau)
        self.macroscopic = MacroscopicMultiphase(grid, lattice, kappa, interface_width, rho_l, rho_v)
        self.collision = CollisionMultiphase(grid, lattice, tau)

    def __call__(self, f: jnp.array):
        rho, u, force = self.macroscopic(f)
        feq = self.equilibrium(rho, u)
        source = self.source_term(rho, u, force)
        fcol = self.collision(f, feq, source)
        fstream = self.streaming(fcol)
        return fstream


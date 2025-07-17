from .base import BaseSimulation
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.macroscopic.macroscopic_multiphase import MacroscopicMultiphase
from wblbm.operators.initialise.init import Initialise


class MultiphaseSimulation(BaseSimulation):
    def __init__(
        self,
        grid_shape,
        lattice_type="D2Q9",
        tau=1.0,
        nt=1000,
        kappa=0.1,
        rhol=1.0,
        rhov=0.1,
        interface_width=4,
        **kwargs
    ):
        super().__init__(grid_shape, lattice_type, tau, nt)
        self.kappa = kappa
        self.rhol = rhol
        self.rhov = rhov
        self.interface_width = interface_width
        self.setup_operators()

    def setup_operators(self):
        self.initialiser = Initialise(self.grid, self.lattice)
        self.update = UpdateMultiphase(
            self.grid,
            self.lattice,
            self.tau,
            self.kappa,
            self.interface_width,
            self.rhol,
            self.rhov,
        )
        self.macroscopic = MacroscopicMultiphase(
            self.grid,
            self.lattice,
            self.kappa,
            self.interface_width,
            self.rhol,
            self.rhov,
        )

    def initialize_fields(self, init_type="multiphase_droplet"):
        if init_type == "multiphase_droplet":
            return self.initialiser.initialise_multiphase_droplet(
                self.rhol, self.rhov, self.interface_width
            )
        elif init_type == "multiphase_bubble":
            return self.initialiser.initialise_multiphase_bubble(
                self.rhol, self.rhov, self.interface_width
            )
        else:
            return self.initialiser.initialise_standard()

    def run_timestep(self, fprev, it):
        return self.update(fprev)

from .multiphase import MultiphaseSimulation
from wblbm.operators.update.update_multiphase_wetting import UpdateMultiphaseWetting
from wblbm.operators.macroscopic.macroscopic_multiphase_wetting import (
    MacroscopicWetting,
)


class WettingSimulation(MultiphaseSimulation):
    def __init__(
        self,
        grid_shape,
        phi_value=1.0,
        d_rho_value=0.0,
        hysteresis_params=None,
        **kwargs
    ):
        self.macroscopic = None
        self.update = None
        self.hysteresis = None
        self.phi_value = phi_value
        self.d_rho_value = d_rho_value
        self.hysteresis_params = hysteresis_params
        super().__init__(grid_shape, **kwargs)

    def setup_operators(self):
        super().setup_operators()
        # Override with wetting-specific operators
        self.update = UpdateMultiphaseWetting(
            self.grid,
            self.lattice,
            self.tau,
            self.kappa,
            self.interface_width,
            self.rhol,
            self.rhov,
            wetting_enabled=True,
        )
        self.macroscopic = MacroscopicWetting(
            self.grid,
            self.lattice,
            self.kappa,
            self.interface_width,
            self.rhol,
            self.rhov,
            wetting_enabled=True,
        )
        # Setup wetting-specific components
        if self.hysteresis_params:
            defaults = {
                "receding_ca_hydrophobic": None,
                "advancing_ca_hydrophilic": None,
                "receding_ca_hydrophilic": None,
                "cll_threshold": None,
                "ca_threshold": None,
                "change_d_rho": None,
                "change_phi": None,
                "while_limiter": None,
                "nx": None,
                "phi_val": None,
                "d_rho_val": None,
                "w": None,
            }
            for key, value in defaults.items():
                self.hysteresis_params.setdefault(key, value)
            from wblbm.operators.wetting.hysteresis import Hysteresis

            self.hysteresis = Hysteresis(**self.hysteresis_params)

    def initialize_fields(self, init_type="wetting"):
        if init_type == "wetting":
            return self.initialiser.initialise_wetting(
                self.rhol, self.rhov, self.interface_width
            )
        else:
            return super().initialize_fields(init_type)

    def run_timestep(self, fprev, it):
        # You may want to add hysteresis logic here if needed
        return self.update(fprev)

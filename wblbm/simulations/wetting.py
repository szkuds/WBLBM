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
        collision_scheme="bgk",
        k_diag=None,
        **kwargs
    ):
        super().__init__(
            grid_shape, collision_scheme=collision_scheme, k_diag=k_diag, **kwargs
        )
        self.macroscopic = None
        self.update = None
        self.hysteresis = None
        self.phi_value = phi_value
        self.d_rho_value = d_rho_value
        self.hysteresis_params = hysteresis_params
        self.multiphase = True
        self.wetting_enabled = True

    def setup_operators(self):
        super().setup_operators()
        # Override with wetting-specific operators
        self.update = UpdateMultiphaseWetting(
            self.grid,
            self.lattice,
            self.tau,
            self.kappa,
            self.interface_width,
            self.rho_l,
            self.rho_v,
            wetting_enabled=True,
            collision_scheme=self.collision_scheme,
            kvec=self.k_diag,
        )
        self.macroscopic = MacroscopicWetting(
            self.grid,
            self.lattice,
            self.kappa,
            self.interface_width,
            self.rho_l,
            self.rho_v,
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

    def initialize_fields(self, init_type="multiphase_droplet", *, init_dir=None):
        if init_type == "init_from_file":
            if init_dir is None:
                raise ValueError(
                    "init_from_file requires init_dir pointing to a .npz file"
                )
            return self.initialiser.init_from_npz(init_dir)

        if init_type == "multiphase_droplet":
            return self.initialiser.initialise_multiphase_droplet(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble":
            return self.initialiser.initialise_multiphase_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_droplet_top":
            return self.initialiser.initialise_multiphase_droplet_top(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "wetting":
            return self.initialiser.initialise_wetting(
                self.rho_l, self.rho_v, self.interface_width
            )
        else:
            return self.initialiser.initialise_standard()

    def run_timestep(self, fprev, it):
        # You may want to add hysteresis logic here if needed
        return self.update(fprev)

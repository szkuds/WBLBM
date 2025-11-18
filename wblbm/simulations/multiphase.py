from .base import BaseSimulation
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.update.update_multiphase_hysteresis import UpdateMultiphaseHysteresis
from wblbm.operators.macroscopic.macroscopic_multiphase_dw import MacroscopicMultiphaseDW
from wblbm.operators.initialise.initialise import Initialise
import jax.numpy as jnp


class MultiphaseSimulation(BaseSimulation):
    def __init__(
        self,
        grid_shape,
        lattice_type="D2Q9",
        tau=1.0,
        nt=1000,
        kappa=0.1,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=4,
        force_enabled=False,
        force_obj=None,
        bc_config=None,
        collision_scheme="bgk",
        k_diag=None,
        eos="double-well",
        **kwargs
    ):
        super().__init__(grid_shape, lattice_type, tau, nt)
        self.update = None
        self.initialise = None
        self.macroscopic = None
        self.eos = eos
        self.kappa = kappa
        self.rho_l = rho_l
        self.rho_v = rho_v
        self.interface_width = interface_width
        self.force_enabled = force_enabled
        self.force_obj = force_obj
        self.bc_config = bc_config
        self.collision_scheme = collision_scheme
        self.k_diag = k_diag
        self.kwargs = kwargs
        self.bubble = kwargs.get('bubble', False)
        self.rho_ref = self.kwargs.get('rho_ref', False)
        self.g = self.kwargs.get('g', False)
        self.setup_operators()
        self.multiphase = True

    def setup_operators(self):
        self.wetting_enabled = any(bc_type == 'wetting'for bc_type in (self.bc_config or {}).values())
        if self.bubble:
            self.initialise = Initialise(self.grid, self.lattice, self.bubble, self.g, self.rho_ref)
        else:
            self.initialise = Initialise(self.grid, self.lattice, self.bubble)
        if self.bc_config and "hysteresis_params" in self.bc_config:
            self.update = UpdateMultiphaseHysteresis(
                self.grid, self.lattice, self.tau, self.kappa, self.interface_width,
                self.rho_l, self.rho_v, self.bc_config, self.force_enabled,
                collision_scheme=self.collision_scheme, eos=self.eos,
                k_diag=self.k_diag, **self.kwargs
            )
        else:
            self.update = UpdateMultiphase(
                self.grid, self.lattice, self.tau, self.kappa, self.interface_width,
                self.rho_l, self.rho_v, self.bc_config, self.force_enabled,
                collision_scheme=self.collision_scheme, eos=self.eos,
                k_diag=self.k_diag, **self.kwargs
            )
        self.macroscopic = self.update.macroscopic

    def initialize_fields(self, init_type="multiphase_droplet", *, init_dir=None):
        if init_type == "init_from_file":
            if init_dir is None:
                raise ValueError(
                    "init_from_file requires init_dir pointing to a .npz file"
                )
            return self.initialise.init_from_npz(init_dir)

        elif init_type == "multiphase_droplet":
            return self.initialise.initialise_multiphase_droplet(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble":
            return self.initialise.initialise_multiphase_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_droplet_top":
            return self.initialise.initialise_multiphase_droplet_top(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble_bot":
            return self.initialise.initialise_multiphase_bubble_bot(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_bubble_bubble":
            return self.initialise.initialise_multiphase_bubble_bubble(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "multiphase_lateral_bubble_configuration":
            return self.initialise.initialise_multiphase_lateral_bubble_configuration(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "wetting_chem_step":
            return self.initialise.initialise_wetting_chemical_step(
                self.rho_l, self.rho_v, self.interface_width
            )
        elif init_type == "wetting":
            return self.initialise.initialise_wetting(
                self.rho_l, self.rho_v, self.interface_width
            )
        else:
            return self.initialise.initialise_standard()

    def run_timestep(self, f_prev, it):
        force_ext = None
        # TODO: This is where the external force is added,
        #  since this will also be how I want to implement the electric force
        #  I will need to look at how I can best extend this.
        #   At the moment I think the creation of a composite force class will be best
        #    As it will allow for multiple force to be dealt with in a similar manner.
        #   https://www.perplexity.ai/search/in-the-case-of-the-electric-fi-tsEeMkPcQzecNfNYtcWVsw
        if self.force_enabled and self.force_obj:
            rho = jnp.sum(f_prev, axis=2, keepdims=True)
            force_ext = self.force_obj.compute_force(rho, self.rho_l, self.rho_v)
        # TODO: Here I need to add the logic to update the electric field. Also need to add it to the single phase sim.
        f_next = (
            self.update(f_prev, force=force_ext)
            if self.force_enabled
            else self.update(f_prev)
        )
        return f_next

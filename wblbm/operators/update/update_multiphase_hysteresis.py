from functools import partial
import jax.numpy as jnp
from jax import jit
from wblbm.operators.update.update_multiphase import UpdateMultiphase
from wblbm.operators.wetting.hysteresis import Hysteresis
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm.operators.wetting.contact_line_location import ContactLineLocation


class UpdateMultiphaseHysteresis(UpdateMultiphase):
    def __init__(self, grid, lattice, tau, kappa, interface_width, rho_l, rho_v,
                 bc_config=None, force_enabled=False, collision_scheme="bgk",
                 eos="double-well", k_diag=None, **kwargs):

        super().__init__(grid, lattice, tau, kappa, interface_width, rho_l, rho_v,
                         bc_config, force_enabled, collision_scheme, eos, k_diag, **kwargs)

        if bc_config and "hysteresis_params" in bc_config:
            self.hysteresis_enabled = True
            self.bc_config = bc_config
            hparams = bc_config["hysteresis_params"]

            self.hysteresis = Hysteresis(
                advancing_ca_hydrophobic=hparams["advancing_ca_hydrophobic"],
                receding_ca_hydrophobic=hparams["receding_ca_hydrophobic"],
                advancing_ca_hydrophilic=hparams["advancing_ca_hydrophilic"],
                receding_ca_hydrophilic=hparams["receding_ca_hydrophilic"],
                cll_threshold=hparams["cll_threshold"],
                ca_threshold=hparams["ca_threshold"],
                change_d_rho=hparams["change_d_rho"],
                change_phi=hparams["change_phi"],
                while_limiter=hparams["while_limiter"],
                nx=grid.nx,
                phi_val=hparams["phi_val"],
                d_rho_val=hparams["d_rho_val"],
                w=hparams["w"]
            )

            rho_mean = (rho_l + rho_v) / 2
            self.contact_angle = ContactAngle(rho_mean)
            self.contact_line_location = ContactLineLocation(rho_mean)
            self.hysteresis_state = self._init_hysteresis_state()
        else:
            self.hysteresis_enabled = False

    def _init_hysteresis_state(self):
        return {
            'left_step_passed': False,
            'right_step_passed': False,
            'cah_window_left_philic': False,
            'cah_window_right_philic': False,
            'cah_window_left_phobic': False,
            'cah_window_right_phobic': False,
            'pinned_count_left': 0,
            'pinned_count_right': 0,
            'pm_left': 'none',
            'pm_right': 'none'
        }

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.array, force: jnp.ndarray = None, it: int = 0):
        if not self.hysteresis_enabled:
            return super().__call__(f, force)

        if self.force_enabled and force is None:
            raise TypeError("When the force is enabled an external force needs to be provided")
        elif self.force_enabled:
            rho, u, force_tot = self.macroscopic(f, force=force)
        else:
            rho, u, force_tot = self.macroscopic(f)

        updated_params = self._apply_hysteresis_update(f, rho, it)
        self._update_wetting_params(updated_params)

        feq = self.equilibrium(rho, u)
        source = self.source_term(rho, u, force_tot)
        fcol = self.collision(f, feq, source)
        fstream = self.streaming(fcol)

        if self.boundary_condition is not None:
            fbc = self.boundary_condition(fstream, fcol)
            return fbc
        else:
            return fstream

    @partial(jit, static_argnums=(0,))
    def _apply_hysteresis_update(self, f, rho, it):
        def update_func(f_in, phi_left, phi_right, d_rho_left, d_rho_right):
            updated_bc_config = self.bc_config.copy()
            if "wetting_params" in updated_bc_config:
                updated_bc_config["wetting_params"]["phi_left"] = phi_left
                updated_bc_config["wetting_params"]["phi_right"] = phi_right
                updated_bc_config["wetting_params"]["d_rh_oleft"] = d_rho_left
                updated_bc_config["wetting_params"]["d_rho_right"] = d_rho_right
            rho_new, u_new, force_tot_new = self.macroscopic(f_in)
            return f_in, rho_new

        def contact_angle_func(rho_in):
            return self.contact_angle.compute(rho_in)

        def cll_func(rho_in, ca_left, ca_right):
            return self.contact_line_location.compute(rho_in, ca_left, ca_right)

        def get_rho_func(f_in):
            rho_out, _, _ = self.macroscopic(f_in)
            return rho_out

        result = self.hysteresis.apply(
            f, it,
            self.hysteresis_state['left_step_passed'],
            self.hysteresis_state['right_step_passed'],
            self.hysteresis_state['cah_window_left_philic'],
            self.hysteresis_state['cah_window_right_philic'],
            self.hysteresis_state['cah_window_left_phobic'],
            self.hysteresis_state['cah_window_right_phobic'],
            self.hysteresis_state['pinned_count_left'],
            self.hysteresis_state['pinned_count_right'],
            update_func,
            contact_angle_func,
            cll_func,
            get_rho_func
        )

        (phi_left, phi_right, d_rho_left, d_rho_right, left_step_passed, right_step_passed,
         cah_window_left_philic, cah_window_right_philic, cah_window_left_phobic,
         cah_window_right_phobic, pinned_count_left, pinned_count_right, pm_left, pm_right) = result

        self.hysteresis_state.update({
            'left_step_passed': left_step_passed,
            'right_step_passed': right_step_passed,
            'cah_window_left_philic': cah_window_left_philic,
            'cah_window_right_philic': cah_window_right_philic,
            'cah_window_left_phobic': cah_window_left_phobic,
            'cah_window_right_phobic': cah_window_right_phobic,
            'pinned_count_left': pinned_count_left,
            'pinned_count_right': pinned_count_right,
            'pm_left': pm_left,
            'pm_right': pm_right
        })

        return {
            'phi_left': phi_left,
            'phi_right': phi_right,
            'd_rho_left': d_rho_left,
            'd_rho_right': d_rho_right
        }

    def _update_wetting_params(self, params):
        if "wetting_params" in self.bc_config:
            self.bc_config["wetting_params"].update(params)
            if hasattr(self.macroscopic, 'gradient'):
                self.macroscopic.gradient.wettingparams.update(params)
            if hasattr(self.macroscopic, 'laplacian'):
                self.macroscopic.laplacian.wettingparams.update(params)

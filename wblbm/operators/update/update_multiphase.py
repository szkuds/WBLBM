from functools import partial

import jax.lax
import jax.numpy as jnp
from jax import jit

from wblbm.grid import Grid
from wblbm.lattice import Lattice
from wblbm.operators.collision.collision_BGK import CollisionBGK
from wblbm.operators.collision.collision_MRT import CollisionMRT
from wblbm.operators.update.update import Update
from wblbm.operators.macroscopic.macroscopic_multiphase_dw import MacroscopicMultiphaseDW
from wblbm.operators.macroscopic.macroscopic_multiphase_cs import MacroscopicMultiphaseCS
from wblbm.operators.wetting.wetting_util import (has_hysteresis_bc)
from wblbm.operators.boundary_condition.boundary_condition import BoundaryCondition
from wblbm.operators.wetting.contact_angle import ContactAngle
from wblbm.operators.wetting.contact_line_location import ContactLineLocation


class UpdateMultiphase(Update):
    def __init__(
            self,
            grid: Grid,
            lattice: Lattice,
            tau: float,
            kappa: float,
            interface_width: int,
            rho_l: float,
            rho_v: float,
            bc_config: dict = None,
            force_enabled: bool = False,
            collision_scheme: str = "bgk",
            eos: str = "double-well",
            k_diag=None,
            **kwargs
    ):
        super().__init__(
            grid,
            lattice,
            tau,
            bc_config,
            force_enabled=force_enabled,
            collision_scheme=collision_scheme,
            k_diag=k_diag,
            **kwargs
        )
        if eos == "double-well":
            self.macroscopic = MacroscopicMultiphaseDW(
                grid, lattice, kappa, interface_width, rho_l, rho_v, force_enabled=force_enabled, bc_config=bc_config
            )
        #TODO: Need to make sure that the maxwell contruction is done to get the correct starting values.
        elif eos == "carnahan-starling":
            macroscopic_args = dict(
                grid=grid,
                lattice=lattice,
                kappa=kappa,
                interface_width=interface_width,
                rho_l=rho_l,
                rho_v=rho_v,
                a_eos=kwargs['a_eos'],
                b_eos=kwargs['b_eos'],
                r_eos=kwargs['r_eos'],
                t_eos=kwargs['t_eos'],
                force_enabled=force_enabled,
                bc_config=bc_config,
            )
            self.macroscopic = MacroscopicMultiphaseCS(**macroscopic_args)

        else:
            raise ValueError(f"Unknown EOS: {eos}")

    @partial(jit, static_argnums=(0,))
    def __call__(self, f: jnp.array, force: jnp.ndarray = None):
        # If force_enabled and no force provided, use a simple constant force for testing
        if self.force_enabled and force is None:
            raise TypeError(
                "When the force is enabled an external force needs to be provided"
            )
        elif self.force_enabled:
            rho_prev, u, force_tot = self.macroscopic(f, force=force)
        else:
            rho_prev, u, force_tot = self.macroscopic(
                f
            )  # In this case the total force is only the interaction force
        feq = self.equilibrium(rho_prev, u)
        source = self.source_term(rho_prev, u, force_tot)
        fcol = self.collision(f, feq, source)
        fstream = self.streaming(fcol)
        if self.boundary_condition is not None and 'hysteresis_params' in self.boundary_condition.bc_config:
            fbc = self.boundary_condition(fstream, fcol)
            # TODO this is the best spot to add the hysteresis logic
            rho_mean = (self.macroscopic.rho_v + self.macroscopic.rho_l) / 2
            ca = ContactAngle(rho_mean)
            cll = ContactLineLocation(rho_mean)
            rho_new, _, _ = self.macroscopic(fbc, force)
            ca_left_t, ca_right_t = ca.compute(rho_prev)
            ca_left_tplus1, ca_right_tplus1 = ca.compute(rho_new)
            cll_left_t, cll_right_t = cll.compute(rho_prev, ca_left_t, ca_right_t)
            cll_left_tplus1, cll_right_tplus1 = cll.compute(rho_prev, ca_left_tplus1, ca_right_tplus1)
            arg_left = (
                ca_left_t.item(),
                ca_left_tplus1.item(),
                self.boundary_condition.bc_config['hysteresis_params']['advancing_ca'],
                self.boundary_condition.bc_config['hysteresis_params']['receding_ca'],
                self.boundary_condition.bc_config['hysteresis_params']['cll_threshold'],
                self.boundary_condition.bc_config['hysteresis_params']['ca_threshold'],
                self.boundary_condition.bc_config['hysteresis_params']['change_d_rho'],
                self.boundary_condition.bc_config['hysteresis_params']['change_phi'],
                self.boundary_condition.bc_config['wetting_params']['phi_left'],
                self.boundary_condition.bc_config['wetting_params']['d_rho_left'],
                f
            )
            arg_right = (
                ca_right_t.item(),
                ca_right_tplus1.item(),
                self.boundary_condition.bc_config['hysteresis_params']['advancing_ca'],
                self.boundary_condition.bc_config['hysteresis_params']['receding_ca'],
                self.boundary_condition.bc_config['hysteresis_params']['cll_threshold'],
                self.boundary_condition.bc_config['hysteresis_params']['ca_threshold'],
                self.boundary_condition.bc_config['hysteresis_params']['change_d_rho'],
                self.boundary_condition.bc_config['hysteresis_params']['change_phi'],
                self.boundary_condition.bc_config['wetting_params']['phi_right'],
                self.boundary_condition.bc_config['wetting_params']['d_rho_right'],
                f
            )

            def ca_increase():
                pass

            def ca_decrease():
                pass

            def update(f_t):
                if self.force_enabled:
                    rho_prev, u, force_tot = self.macroscopic(f_t, force=force)
                else:
                    rho_prev, u, force_tot = self.macroscopic(
                        f_t)  # In this case the total force is only the interaction force
                feq = self.equilibrium(rho_prev, u)
                source = self.source_term(rho_prev, u, force_tot)
                fcol = self.collision(f, feq, source)
                fstream = self.streaming(fcol)
                fbc = self.boundary_condition(fstream)
                if self.force_enabled:
                    rho_new, _, _ = self.macroscopic(fbc, force=force)
                else:
                    rho_new, _, _ = self.macroscopic(fbc)  # In this case the total force is only the interaction force
                ca_left_tplus1_, ca_right_tplus1_ = ca.compute(rho_new)
                cll_left_tplus1_, cll_left_tplus1_ = cll.compute(rho_new, ca_left_tplus1_, ca_right_tplus1_)
                return ca_left_tplus1_, ca_right_tplus1_, cll_left_tplus1_, cll_left_tplus1_

            def receding_pinned_left(args):
                pass

            def receding_pinned_right(args):
                pass

            def receding_moving_left(args):
                pass

            def receding_moving_right(args):
                pass

            def advancing_pinned_left(args):
                # TODO: The while version of JAX makes things a little tricky, the goal is to find how to do this with
                #       Optax

                cll_threshold, cll_t, cll_tplus1 = args[4], args[0], args[1]

                def while_condition(cll_tplus1_conditional):
                    return (1 + cll_threshold) * cll_t > cll_tplus1_conditional > (1 - cll_threshold) * cll_t

                def while_update_func():
                    pass

                jax.lax.while_loop(while_condition(cll_tplus1), )

            def advancing_pinned_right(args):
                pass

            def advancing_moving_left(args):
                pass

            def advancing_moving_right(args):
                pass

            def receding_left(args):
                ca_tplus1, ca_rec = args[1], args[3]
                jax.lax.cond(ca_tplus1 >= ca_rec, receding_pinned_left, receding_moving_left, args)

            def receding_right(args):
                ca_tplus1, ca_rec = args[1], args[3]
                jax.lax.cond(ca_tplus1 >= ca_rec, receding_pinned_right, receding_moving_right, args)

            def advancing_left(args):
                ca_tplus1, ca_ad = args[1], args[2]
                jax.lax.cond(ca_tplus1 <= ca_ad, advancing_pinned_left, advancing_moving_left, args)

            def advancing_right(args):
                ca_tplus1, ca_ad = args[1], args[2]
                jax.lax.cond(ca_tplus1 <= ca_ad, advancing_pinned_right, advancing_moving_right, args)

            jax.lax.cond(cll_left_t < cll_left_tplus1, receding_left, advancing_left, arg_left)
            jax.lax.cond(cll_right_t < cll_right_tplus1, advancing_right, receding_right, arg_right)

            return fbc
        elif self.boundary_condition is not None:
            fbc = self.boundary_condition(fstream, fcol)
            return fbc

        else:
            return fstream

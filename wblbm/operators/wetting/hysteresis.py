import jax.numpy as jnp
from jax import lax
import logging


class Hysteresis:
    """
    Class to model contact angle hysteresis in wetting simulations.
    JAX-compatible version with conditional operations using jax.lax functions.
    """

    # Numeric constants for return codes
    NONE = 0
    RECEDING_PINNED = 1
    RECEDING_MOVING = 2
    ADVANCING_PINNED = 3
    ADVANCING_MOVING = 4

    def __init__(self, advancing_ca_hydrophobic, receding_ca_hydrophobic,
                 advancing_ca_hydrophilic, receding_ca_hydrophilic,
                 cll_threshold, ca_threshold, change_d_rho, change_phi,
                 while_limiter, nx, phi_val, d_rho_val, w):
        self.advancing_ca_hydrophobic = advancing_ca_hydrophobic
        self.receding_ca_hydrophobic = receding_ca_hydrophobic
        self.advancing_ca_hydrophilic = advancing_ca_hydrophilic
        self.receding_ca_hydrophilic = receding_ca_hydrophilic
        self.cll_threshold = cll_threshold
        self.ca_threshold = ca_threshold
        self.change_d_rho = change_d_rho
        self.change_phi = change_phi
        self.while_limiter = while_limiter
        self.nx = nx
        self.phi_val = phi_val
        self.d_rho_val = d_rho_val
        self.w = w

    def apply(self, f_prev, it, left_step_passed, right_step_passed,
              cah_window_left_philic, ca_hwindow_right_philic,
              ca_hwindow_left_phobic, ca_hwindow_right_phobic,
              pinned_count_left, pinned_count_right,
              update_func, contact_angle_func, cll_func, get_rho_func):
        """
        Apply hysteresis logic using JAX-compatible operations.
        """

        # Initialize parameters
        phi_left = jnp.ones(self.nx)
        d_rho_left = jnp.zeros(self.nx)
        phi_left = phi_left.at[self.nx // 2].set(self.phi_val)
        d_rho_left = d_rho_left.at[self.nx // 2].set(self.d_rho_val)

        phi_right = jnp.ones(self.nx)
        d_rho_right = jnp.zeros(self.nx)
        phi_right = phi_right.at[self.nx // 2].set(self.phi_val)
        d_rho_right = d_rho_right.at[self.nx // 2].set(self.d_rho_val)

        ft = f_prev
        ft_plus1, rho_t = update_func(ft, phi_left, phi_right, d_rho_left, d_rho_right)
        rho_t_plus1 = get_rho_func(ft_plus1)

        ca_left_t, ca_right_t = contact_angle_func(rho_t)
        cll_left_t, cll_right_t = cll_func(rho_t, ca_left_t, ca_right_t)

        ca_left_t_plus1, ca_right_t_plus1 = contact_angle_func(rho_t_plus1)
        cll_left_t_plus1, cll_right_t_plus1 = cll_func(rho_t_plus1, ca_left_t_plus1, ca_right_t_plus1)

        # Left side logic - using JAX conditionals
        ca_left_t_plus1_scalar = jnp.squeeze(ca_left_t_plus1)
        cah_window_left_philic = lax.cond(
            ca_left_t_plus1_scalar >= self.advancing_ca_hydrophilic,  # Now scalar
            lambda x: True,
            lambda x: cah_window_left_philic,
            None
        )

        cll_left_t_plus1_scalar = jnp.squeeze(cll_left_t_plus1)
        left_step_passed = lax.cond(
            cll_left_t_plus1_scalar <= (self.nx // 2 - self.w // 2),
            lambda x: True,
            lambda x: left_step_passed,
            None
        )

        # Set receding and advancing contact angles based on conditions
        receding_ca_left, advancing_ca_left, cll_threshold_left = lax.cond(
            (cll_left_t_plus1_scalar <= (self.nx // 2 - self.w // 2)) & (~left_step_passed),
            lambda x: (self.receding_ca_hydrophobic, self.advancing_ca_hydrophobic, self.cll_threshold),
            lambda x: lax.cond(
                (ca_left_t_plus1_scalar >= self.advancing_ca_hydrophilic) & left_step_passed & (~cah_window_left_philic),
                lambda y: (ca_left_t_plus1_scalar - 1 + 1e-7, ca_left_t_plus1_scalar, self.cll_threshold),
                lambda y: (self.receding_ca_hydrophilic, self.advancing_ca_hydrophilic, self.cll_threshold),
                None
            ),
            None
        )

        # Right side logic
        ca_right_t_plus1_scalar = jnp.squeeze(ca_right_t_plus1)
        ca_hwindow_right_philic = lax.cond(
            ca_right_t_plus1_scalar >= self.advancing_ca_hydrophilic,
            lambda x: True,
            lambda x: ca_hwindow_right_philic,
            None
        )

        cll_right_t_plus1_scalar = jnp.squeeze(cll_right_t_plus1)
        right_step_passed = lax.cond(
            cll_right_t_plus1_scalar >= (self.nx // 2 + self.w // 2),
            lambda x: True,
            lambda x: right_step_passed,
            None
        )

        receding_ca_right, advancing_ca_right, cll_threshold_right = lax.cond(
            (cll_right_t_plus1_scalar >= (self.nx // 2 + self.w // 2)) & (~right_step_passed),
            lambda x: (self.receding_ca_hydrophobic, self.advancing_ca_hydrophobic, self.cll_threshold),
            lambda x: lax.cond(
                (ca_right_t_plus1_scalar >= self.advancing_ca_hydrophilic) & right_step_passed & (~ca_hwindow_right_philic),
                lambda y: (ca_right_t_plus1_scalar, ca_right_t_plus1_scalar + 1 - 1e-7, self.cll_threshold),
                lambda y: (self.receding_ca_hydrophilic, self.advancing_ca_hydrophilic, self.cll_threshold),
                None
            ),
            None
        )

        # Left logic - Main processing
        cll_left_t_scalar = jnp.squeeze(cll_left_t)
        dummy_scan_result_left = (0, phi_left, d_rho_left, True)
        pm_left = lax.cond(
            cll_left_t_plus1_scalar < cll_left_t_scalar,
            lambda x: self._process_left_receding(
                ft, phi_left, phi_right, d_rho_left, d_rho_right,
                ca_left_t_plus1, receding_ca_left, advancing_ca_left,
                cll_left_t, cll_left_t_plus1, cll_threshold_left,
                pinned_count_left, update_func, contact_angle_func, cll_func, get_rho_func
            ),
            lambda x: lax.cond(
                jnp.squeeze(cll_left_t_plus1 > cll_left_t),
                lambda y: self._process_left_advancing(
                    ft, phi_left, phi_right, d_rho_left, d_rho_right,
                    ca_left_t_plus1, receding_ca_left, advancing_ca_left,
                    cll_left_t, cll_left_t_plus1, cll_threshold_left,
                    pinned_count_left, update_func, contact_angle_func, cll_func, get_rho_func
                ),
                lambda y: (self.NONE, dummy_scan_result_left),
                None
            ),
            None
        )

        # Right logic - Main processing
        dummy_scan_result_right = (0, phi_right, d_rho_right, True)
        pm_right = lax.cond(
            jnp.squeeze(cll_right_t_plus1 > cll_right_t),
            lambda x: self._process_right_advancing(
                ft, phi_left, phi_right, d_rho_left, d_rho_right,
                ca_right_t_plus1, receding_ca_right, advancing_ca_right,
                cll_right_t, cll_right_t_plus1, cll_threshold_right,
                pinned_count_right, update_func, contact_angle_func, cll_func, get_rho_func
            ),
            lambda x: lax.cond(
                jnp.squeeze(cll_right_t_plus1 < cll_right_t),
                lambda y: self._process_right_receding(
                    ft, phi_left, phi_right, d_rho_left, d_rho_right,
                    ca_right_t_plus1, receding_ca_right, advancing_ca_right,
                    cll_right_t, cll_right_t_plus1, cll_threshold_right,
                    pinned_count_right, update_func, contact_angle_func, cll_func, get_rho_func
                ),
                lambda y: (self.NONE, dummy_scan_result_right),
                None
            ),
            None
        )

        # Extract only the numeric code for pm_left and pm_right for return
        pm_left_code = pm_left[0]
        pm_right_code = pm_right[0]

        # DEBUG: Print computed contact angles and state variables
        print(f"[DEBUG] it={it} ca_left_t_plus1={float(jnp.squeeze(ca_left_t_plus1)):.2f} ca_right_t_plus1={float(jnp.squeeze(ca_right_t_plus1)):.2f}")
        print(f"[DEBUG] it={it} left_step_passed={left_step_passed} right_step_passed={right_step_passed}")
        print(f"[DEBUG] it={it} cah_window_left_philic={cah_window_left_philic} ca_hwindow_right_philic={ca_hwindow_right_philic}")
        print(f"{it},{jnp.max(phi_left)},{jnp.max(phi_right)},{jnp.max(d_rho_left)},{jnp.max(d_rho_right)}")

        return (phi_left, phi_right, d_rho_left, d_rho_right,
                left_step_passed, right_step_passed,
                cah_window_left_philic, ca_hwindow_right_philic,
                ca_hwindow_left_phobic, ca_hwindow_right_phobic,
                pinned_count_left, pinned_count_right, pm_left_code, pm_right_code)

    def _process_left_receding(self, ft, phi_left, phi_right, d_rho_left, d_rho_right,
                               ca_left_t_plus1, receding_ca_left, advancing_ca_left,
                               cll_left_t, cll_left_t_plus1, cll_threshold_left,
                               pinned_count_left, update_func, contact_angle_func, cll_func, get_rho_func):
        """Process left receding logic using JAX scan for while loop replacement."""

        def body_fun(carry, _):
            count, phi_left_curr, d_rho_left_curr, should_continue = carry

            # Check if we should break
            ca_condition = jnp.abs(1 - self.ca_threshold - receding_ca_left - ca_left_t_plus1) <= (
                        1 - self.ca_threshold) * receding_ca_left
            count_condition = count >= self.while_limiter

            should_break = jnp.squeeze(ca_condition | count_condition)

            # Update logic when not breaking
            phi_update_cond = jnp.min(d_rho_left_curr) >= -1
            drho_update_cond = jnp.min(phi_left_curr) >= 1

            phi_left_new = lax.cond(
                should_break,
                lambda x: phi_left_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_left_t_plus1) >= receding_ca_left,
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: phi_left_curr - self.change_phi,
                        lambda z: phi_left_curr,
                        None
                    ),
                    lambda y: lax.cond(
                        drho_update_cond,
                        lambda z: phi_left_curr,
                        lambda z: phi_left_curr - self.change_phi,
                        None
                    ),
                    None
                ),
                None
            )

            d_rho_left_new = lax.cond(
                should_break,
                lambda x: d_rho_left_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_left_t_plus1) >= receding_ca_left,
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: d_rho_left_curr,
                        lambda z: d_rho_left_curr + self.change_d_rho,
                        None
                    ),
                    lambda y: lax.cond(
                        drho_update_cond,
                        lambda z: d_rho_left_curr - self.change_d_rho,
                        lambda z: d_rho_left_curr,
                        None
                    ),
                    None
                ),
                None
            )

            count_new = lax.cond(should_break, lambda x: count, lambda x: count + 1, None)
            should_continue_new = ~should_break

            return (count_new, phi_left_new, d_rho_left_new, should_continue_new), None

        # Check if this is receding pinned or receding moving
        is_pinned = jnp.squeeze(ca_left_t_plus1 <= receding_ca_left)
        scan_result = lax.scan(body_fun, (0, phi_left, d_rho_left, True), jnp.arange(self.while_limiter))[0]
        return lax.cond(
            is_pinned,
            lambda x: (self.RECEDING_PINNED, scan_result),
            lambda x: (self.RECEDING_MOVING, scan_result),
            None
        )

    def _process_left_advancing(self, ft, phi_left, phi_right, d_rho_left, d_rho_right,
                                ca_left_t_plus1, receding_ca_left, advancing_ca_left,
                                cll_left_t, cll_left_t_plus1, cll_threshold_left,
                                pinned_count_left, update_func, contact_angle_func, cll_func, get_rho_func):
        """Process left advancing logic using JAX scan for while loop replacement."""

        def body_fun(carry, _):
            count, phi_left_curr, d_rho_left_curr, should_continue = carry

            # Check if we should break
            ca_condition = jnp.abs(1 - self.ca_threshold - advancing_ca_left - ca_left_t_plus1) <= (
                        1 - self.ca_threshold) * advancing_ca_left
            count_condition = count >= self.while_limiter

            should_break = jnp.squeeze(ca_condition | count_condition)

            # Update logic when not breaking
            d_rho_update_cond = jnp.max(d_rho_left_curr) <= 0
            phi_update_cond = jnp.min(phi_left_curr) >= 1

            phi_left_new = lax.cond(
                should_break,
                lambda x: phi_left_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_left_t_plus1 <= advancing_ca_left),
                    lambda y: lax.cond(
                        d_rho_update_cond,
                        lambda z: phi_left_curr + self.change_phi,
                        lambda z: phi_left_curr,
                        None
                    ),
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: phi_left_curr,
                        lambda z: phi_left_curr + self.change_phi,
                        None
                    ),
                    None
                ),
                None
            )

            d_rho_left_new = lax.cond(
                should_break,
                lambda x: d_rho_left_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_left_t_plus1 <= advancing_ca_left),
                    lambda y: lax.cond(
                        d_rho_update_cond,
                        lambda z: d_rho_left_curr,
                        lambda z: d_rho_left_curr - self.change_d_rho,
                        None
                    ),
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: d_rho_left_curr + self.change_d_rho,
                        lambda z: d_rho_left_curr,
                        None
                    ),
                    None
                ),
                None
            )

            count_new = lax.cond(should_break, lambda x: count, lambda x: count + 1, None)
            should_continue_new = ~should_break

            return (count_new, phi_left_new, d_rho_left_new, should_continue_new), None

        # Check if this is advancing pinned or advancing moving
        is_pinned = jnp.squeeze(ca_left_t_plus1 >= advancing_ca_left)
        scan_result = lax.scan(body_fun, (0, phi_left, d_rho_left, True), jnp.arange(self.while_limiter))[0]
        return lax.cond(
            is_pinned,
            lambda x: (self.ADVANCING_PINNED, scan_result),
            lambda x: (self.ADVANCING_MOVING, scan_result),
            None
        )

    def _process_right_advancing(self, ft, phi_left, phi_right, d_rho_left, d_rho_right,
                                 ca_right_t_plus1, receding_ca_right, advancing_ca_right,
                                 cll_right_t, cll_right_t_plus1, cll_threshold_right,
                                 pinned_count_right, update_func, contact_angle_func, cll_func, get_rho_func):
        """Process right advancing logic using JAX scan for while loop replacement."""

        def body_fun(carry, _):
            count, phi_right_curr, d_rho_right_curr, should_continue = carry

            # Check if we should break
            ca_condition = jnp.abs(1 - self.ca_threshold - advancing_ca_right - ca_right_t_plus1) <= (
                        1 - self.ca_threshold) * advancing_ca_right
            count_condition = count >= self.while_limiter

            should_break = jnp.squeeze(ca_condition | count_condition)

            # Update logic when not breaking
            drho_update_cond = jnp.max(d_rho_right_curr) <= 0
            phi_update_cond = jnp.min(phi_right_curr) >= 1

            phi_right_new = lax.cond(
                should_break,
                lambda x: phi_right_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_right_t_plus1 <= advancing_ca_right),
                    lambda y: lax.cond(
                        drho_update_cond,
                        lambda z: phi_right_curr + self.change_phi,
                        lambda z: phi_right_curr,
                        None
                    ),
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: phi_right_curr,
                        lambda z: phi_right_curr + self.change_phi,
                        None
                    ),
                    None
                ),
                None
            )

            d_rho_right_new = lax.cond(
                should_break,
                lambda x: d_rho_right_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_right_t_plus1 <= advancing_ca_right),
                    lambda y: lax.cond(
                        drho_update_cond,
                        lambda z: d_rho_right_curr,
                        lambda z: d_rho_right_curr - self.change_d_rho,
                        None
                    ),
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: d_rho_right_curr + self.change_d_rho,
                        lambda z: d_rho_right_curr,
                        None
                    ),
                    None
                ),
                None
            )

            count_new = lax.cond(should_break, lambda x: count, lambda x: count + 1, None)
            should_continue_new = ~should_break

            return (count_new, phi_right_new, d_rho_right_new, should_continue_new), None

        # Check if this is advancing pinned or advancing moving
        is_pinned = jnp.squeeze(ca_right_t_plus1 >= advancing_ca_right)
        scan_result = lax.scan(body_fun, (0, phi_right, d_rho_right, True), jnp.arange(self.while_limiter))[0]
        return lax.cond(
            is_pinned,
            lambda x: (self.ADVANCING_PINNED, scan_result),
            lambda x: (self.ADVANCING_MOVING, scan_result),
            None
        )

    def _process_right_receding(self, ft, phi_left, phi_right, d_rho_left, d_rho_right,
                                ca_right_t_plus1, receding_ca_right, advancing_ca_right,
                                cll_right_t, cll_right_t_plus1, cll_threshold_right,
                                pinned_count_right, update_func, contact_angle_func, cll_func, get_rho_func):
        """Process right receding logic using JAX scan for while loop replacement."""

        def body_fun(carry, _):
            count, phi_right_curr, d_rho_right_curr, should_continue = carry

            # Check if we should break
            ca_condition = jnp.abs(1 - self.ca_threshold - receding_ca_right - ca_right_t_plus1) <= (
                        1 - self.ca_threshold) * receding_ca_right
            count_condition = count >= self.while_limiter

            should_break = jnp.squeeze(ca_condition | count_condition)

            # Update logic when not breaking
            phi_update_cond = jnp.min(d_rho_right_curr) >= -1
            drho_update_cond = jnp.min(phi_right_curr) >= 1

            phi_right_new = lax.cond(
                should_break,
                lambda x: phi_right_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_right_t_plus1 >= receding_ca_right),
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: phi_right_curr - self.change_phi,
                        lambda z: phi_right_curr,
                        None
                    ),
                    lambda y: lax.cond(
                        drho_update_cond,
                        lambda z: phi_right_curr,
                        lambda z: phi_right_curr - self.change_phi,
                        None
                    ),
                    None
                ),
                None
            )

            d_rho_right_new = lax.cond(
                should_break,
                lambda x: d_rho_right_curr,
                lambda x: lax.cond(
                    jnp.squeeze(ca_right_t_plus1 >= receding_ca_right),
                    lambda y: lax.cond(
                        phi_update_cond,
                        lambda z: d_rho_right_curr,
                        lambda z: d_rho_right_curr + self.change_d_rho,
                        None
                    ),
                    lambda y: lax.cond(
                        drho_update_cond,
                        lambda z: d_rho_right_curr - self.change_d_rho,
                        lambda z: d_rho_right_curr,
                        None
                    ),
                    None
                ),
                None
            )

            count_new = lax.cond(should_break, lambda x: count, lambda x: count + 1, None)
            should_continue_new = ~should_break

            return (count_new, phi_right_new, d_rho_right_new, should_continue_new), None

        # Check if this is receding pinned or receding moving
        is_pinned = jnp.squeeze(ca_right_t_plus1 <= receding_ca_right)
        scan_result = lax.scan(body_fun, (0, phi_right, d_rho_right, True), jnp.arange(self.while_limiter))[0]
        return lax.cond(
            is_pinned,
            lambda x: (self.RECEDING_PINNED, scan_result),
            lambda x: (self.RECEDING_MOVING, scan_result),
            None
        )

import jax.numpy as jnp
import logging


class Hysteresis:
    """
    Class to model contact angle hysteresis in wetting simulations.
    """

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

    def apply(self, f_prev, it, left_step_passed, right_step_passed, cah_window_left_philic,
              cah_window_right_philic, cah_window_left_phobic, cah_window_right_phobic,
              pinned_count_left, pinned_count_right, update_func, contact_angle_func, cll_func, get_rho_func):
        # Initialize parameters
        phi_left = jnp.ones(self.nx)
        d_rho_left = jnp.zeros(self.nx)
        phi_left = phi_left.at[(self.nx // 2):].set(self.phi_val)
        d_rho_left = d_rho_left.at[:(self.nx // 2)].set(self.d_rho_val)

        phi_right = jnp.ones(self.nx)
        d_rho_right = jnp.zeros(self.nx)
        phi_right = phi_right.at[(self.nx // 2):].set(self.phi_val)
        d_rho_right = d_rho_right.at[:(self.nx // 2)].set(self.d_rho_val)

        f_t = f_prev
        f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
        rho_tplus1 = get_rho_func(f_tplus1)
        ca_left_t, ca_right_t = contact_angle_func(rho_t)
        cll_left_t, cll_right_t = cll_func(rho_t, ca_left_t, ca_right_t)
        ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
        cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)

        # --- Hysteresis logic ---
        # Left side
        if ca_left_tplus1 < self.advancing_ca_hydrophilic:
            cah_window_left_philic = True
        if cll_left_tplus1 > self.nx // 2 - self.w // 2:
            left_step_passed = True
        if cll_left_tplus1 < self.nx // 2 - self.w // 2 and not left_step_passed:
            receding_ca_left = self.receding_ca_hydrophobic
            advancing_ca_left = self.advancing_ca_hydrophobic
            cll_threshold_left = self.cll_threshold
        elif ca_left_tplus1 > self.advancing_ca_hydrophilic and left_step_passed and not cah_window_left_philic:
            receding_ca_left = ca_left_tplus1 * (1 + 1e-7)
            advancing_ca_left = ca_left_tplus1
            cll_threshold_left = self.cll_threshold
        else:
            receding_ca_left = self.receding_ca_hydrophilic
            advancing_ca_left = self.advancing_ca_hydrophilic
            cll_threshold_left = self.cll_threshold

        # Right side
        if ca_right_tplus1 < self.advancing_ca_hydrophilic:
            cah_window_right_philic = True
        if cll_right_tplus1 > self.nx // 2 - self.w // 2:
            right_step_passed = True
        if cll_right_tplus1 < self.nx // 2 - self.w // 2 and not right_step_passed:
            receding_ca_right = self.receding_ca_hydrophobic
            advancing_ca_right = self.advancing_ca_hydrophobic
            cll_threshold_right = self.cll_threshold
        elif ca_right_tplus1 > self.advancing_ca_hydrophilic and right_step_passed and not cah_window_right_philic:
            receding_ca_right = ca_right_tplus1
            advancing_ca_right = ca_right_tplus1 * (1 - 1e-7)
            cll_threshold_right = self.cll_threshold
        else:
            receding_ca_right = self.receding_ca_hydrophilic
            advancing_ca_right = self.advancing_ca_hydrophilic
            cll_threshold_right = self.cll_threshold

        # --- Left logic ---
        p_m_left = None
        if cll_left_tplus1 > cll_left_t:
            # receding left
            if ca_left_tplus1 >= receding_ca_left:
                count = 0
                p_m_left = 'receding pinned'
                pinned_count_left += 1
                while True:
                    if (1 + cll_threshold_left) * cll_left_t > cll_left_tplus1 > (1 - cll_threshold_left) * cll_left_t:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the left receding pinned while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the left receding pinned while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large')
                        break
                    elif cll_left_tplus1 > cll_left_t:
                        if jnp.max(d_rho_left) > 0:
                            d_rho_left = d_rho_left - self.change_d_rho
                        else:
                            phi_left = phi_left + self.change_phi
                        count += 1
                    else:
                        if jnp.min(phi_left) > 1:
                            phi_left = phi_left - self.change_phi
                        else:
                            d_rho_left = d_rho_left + self.change_d_rho
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
            else:
                if jnp.min(phi_left) > 1:
                    phi_left = phi_left - self.change_phi
                else:
                    d_rho_left = d_rho_left + self.change_d_rho
                f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                rho_tplus1 = get_rho_func(f_tplus1)
                ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
                count = 0
                p_m_left = 'receding moving'
                while True:
                    if (1 + self.ca_threshold) * receding_ca_left > ca_left_tplus1 > (1 - self.ca_threshold) * receding_ca_left:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the left receding moving while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the left receding moving while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large')
                        break
                    elif jnp.min(d_rho_left) > 1:
                        print(f'At it= {it} min(d_rho_left) > 1, in the left receding moving while loop\n'
                              'This means that there is a problem, possibly the threshold values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} min(d_rho_left) > 1, in the left receding moving while loop\n'
                                     'This means that there is a problem, possibly the threshold values for '
                                     'the wetting parameters are too large')
                        break
                    elif ca_left_tplus1 < receding_ca_left:
                        if jnp.min(phi_left) > 1:
                            phi_left = phi_left - self.change_phi
                        else:
                            d_rho_left = d_rho_left + self.change_d_rho
                        count += 1
                    else:
                        if jnp.max(d_rho_left) > 0:
                            d_rho_left = d_rho_left - self.change_d_rho
                        else:
                            phi_left = phi_left + self.change_phi
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
        else:
            # advancing left
            if ca_left_tplus1 <= advancing_ca_left:
                count = 0
                p_m_left = 'advancing pinned'
                pinned_count_left += 1
                while True:
                    if (1 + cll_threshold_left) * cll_left_t > cll_left_tplus1 > (1 - cll_threshold_left) * cll_left_t:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the left advancing pinned while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the left advancing pinned while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large')
                        break
                    elif cll_left_tplus1 < cll_left_t:
                        if jnp.min(phi_left) > 1:
                            phi_left = phi_left - self.change_phi
                        else:
                            d_rho_left = d_rho_left + self.change_d_rho
                        count += 1
                    else:
                        if jnp.max(d_rho_left) > 0:
                            d_rho_left = d_rho_left - self.change_d_rho
                        else:
                            phi_left = phi_left + self.change_phi
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
            else:
                if jnp.max(d_rho_left) > 0:
                    d_rho_left = d_rho_left - self.change_d_rho
                else:
                    phi_left = phi_left + self.change_phi
                f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                rho_tplus1 = get_rho_func(f_tplus1)
                ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
                count = 0
                p_m_left = 'advancing moving'
                while True:
                    if (1 + self.ca_threshold) * advancing_ca_left > ca_left_tplus1 > (1 - self.ca_threshold) * advancing_ca_left:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the left advancing moving while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the left advancing moving while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large')
                        break
                    elif ca_left_tplus1 > advancing_ca_left:
                        if jnp.max(d_rho_left) > 0:
                            d_rho_left = d_rho_left - self.change_d_rho
                        else:
                            phi_left = phi_left + self.change_phi
                        count += 1
                    else:
                        if jnp.min(phi_left) > 1:
                            phi_left = phi_left - self.change_phi
                        else:
                            d_rho_left = d_rho_left + self.change_d_rho
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)

        # --- Right logic ---
        p_m_right = None
        if cll_right_tplus1 > cll_right_t:
            # advancing right
            if ca_right_tplus1 <= advancing_ca_right:
                count = 0
                p_m_right = 'advancing pinned'
                pinned_count_right += 1
                while True:
                    if (1 + cll_threshold_right) * cll_right_t > cll_right_tplus1 > (1 - cll_threshold_right) * cll_right_t:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the right advancing pinned while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the right advancing pinned while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large')
                        break
                    elif cll_right_tplus1 > cll_right_t:
                        if jnp.min(phi_right) > 1:
                            phi_right = phi_right - self.change_phi
                        else:
                            d_rho_right = d_rho_right + self.change_d_rho
                        count += 1
                    else:
                        if jnp.max(d_rho_right) > 0:
                            d_rho_right = d_rho_right - self.change_d_rho
                        else:
                            phi_right = phi_right + self.change_phi
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
            else:
                if jnp.max(d_rho_right) > 0:
                    d_rho_right = d_rho_right - self.change_d_rho
                    count += 1
                else:
                    phi_right = phi_right + self.change_phi
                    count += 1
                f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                rho_tplus1 = get_rho_func(f_tplus1)
                ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
                count = 0
                p_m_right = 'advancing moving'
                while True:
                    if (1 + self.ca_threshold) * advancing_ca_right > ca_right_tplus1 > (1 - self.ca_threshold) * advancing_ca_right:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the right advancing moving while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the right advancing moving while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large')
                        break
                    elif ca_right_tplus1 > advancing_ca_right:
                        if jnp.max(d_rho_right) > 0:
                            d_rho_right = d_rho_right - self.change_d_rho
                        else:
                            phi_right = phi_right + self.change_phi
                        count += 1
                    else:
                        if jnp.min(phi_right) > 1:
                            phi_right = phi_right - self.change_phi
                        else:
                            d_rho_right = d_rho_right + self.change_d_rho
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
        else:
            # receding right
            if ca_right_tplus1 >= receding_ca_right:
                count = 0
                p_m_right = 'receding pinned'
                pinned_count_right += 1
                while True:
                    if (1 + cll_threshold_right) * cll_right_t > cll_right_tplus1 > (1 - cll_threshold_right) * cll_right_t:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the right receding pinned while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large.')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the right receding pinned while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large.')
                        break
                    elif cll_right_tplus1 > cll_right_t:
                        if jnp.min(phi_right) > 1:
                            phi_right = phi_right - self.change_phi
                        else:
                            d_rho_right = d_rho_right + self.change_d_rho
                        count += 1
                    else:
                        if jnp.max(d_rho_right) > 0:
                            d_rho_right = d_rho_right - self.change_d_rho
                        else:
                            phi_right = phi_right + self.change_phi
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
            else:
                if jnp.min(phi_right) > 1:
                    phi_right = phi_right - self.change_phi
                    count += 1
                else:
                    d_rho_right = d_rho_right + self.change_d_rho
                    count += 1
                f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                rho_tplus1 = get_rho_func(f_tplus1)
                ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)
                count = 0
                p_m_right = 'receding moving'
                while True:
                    if (1 + self.ca_threshold) * receding_ca_right > ca_right_tplus1 > (1 - self.ca_threshold) * receding_ca_right:
                        break
                    if count > self.while_limiter:
                        print(f'At it= {it} count>WHILE_LIMITER, in the right receding moving while loop\n'
                              'This means that there is a problem, possibly the adaptation values for '
                              'the wetting parameters are too large')
                        logging.info(f'At it= {it} count>WHILE_LIMITER, in the right receding moving while loop\n'
                                     'This means that there is a problem, possibly the adaptation values for '
                                     'the wetting parameters are too large')
                        break
                    elif ca_right_tplus1 < receding_ca_right:
                        if jnp.min(phi_right) > 1:
                            phi_right = phi_right - self.change_phi
                        else:
                            d_rho_right = d_rho_right + self.change_d_rho
                        count += 1
                    else:
                        if jnp.max(d_rho_right) > 0:
                            d_rho_right = d_rho_right - self.change_d_rho
                        else:
                            phi_right = phi_right + self.change_phi
                        count += 1
                    f_tplus1, rho_t = update_func(f_t, phi_left, phi_right, d_rho_left, d_rho_right)
                    rho_tplus1 = get_rho_func(f_tplus1)
                    ca_left_tplus1, ca_right_tplus1 = contact_angle_func(rho_tplus1)
                    cll_left_tplus1, cll_right_tplus1 = cll_func(rho_tplus1, ca_left_tplus1, ca_right_tplus1)

        # --- Pinning count logic ---
        if ca_left_t < receding_ca_left + 1 or ca_left_t > advancing_ca_left - 1:
            pinned_count_left += -1
        if ca_right_t < receding_ca_right + 1 or ca_right_t > advancing_ca_right - 1:
            pinned_count_right += -1

        return (phi_left, phi_right, d_rho_left, d_rho_right, left_step_passed, right_step_passed,
                cah_window_left_philic, cah_window_right_philic, cah_window_left_phobic,
                cah_window_right_phobic, pinned_count_left, pinned_count_right, p_m_left, p_m_right)

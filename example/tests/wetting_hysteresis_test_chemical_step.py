import numpy as np
from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", True)
# TODO: To just make the implementation of hysteresis slightly simpler, I am not including the chemical step at the moment
#  this logic will need to be added in later by a jax.lax.cond statement which determines if the step has been passed or not.
def test_wetting_hysteresis_simulation():
    """Test LBM wetting implementation with hysteresis enabled."""
    print("\n=== Testing LBM Wetting with Hysteresis ===")

    # Simulation parameters
    grid_shape = (200, 100)
    tau = 0.99
    nt = 2000
    save_interval = 200
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 5

    phi_value = 1.2
    d_rho_value = 0.0

    force_g = 0.0
    inclination_angle = 0
    gravity = GravityForceMultiphaseDroplet(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )

    bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'symmetry',
        'right': 'periodic',
        'wetting_params': {
            'rho_l': rho_l,
            'rho_v': rho_v,
            'phi_left': phi_value,
            'phi_right': phi_value,
            'd_rho_left': d_rho_value,
            'd_rho_right': d_rho_value,
            'width': interface_width
        },
        'hysteresis_params': {
            'advancing_ca_hydrophobic': 91.0,
            'receding_ca_hydrophobic': 89.0,
            'advancing_ca_hydrophilic': 60.0,
            'receding_ca_hydrophilic': 30.0,
            'cll_threshold': 1e-3,
            'ca_threshold': 1e-3,
            'change_d_rho': d_rho_value / 50,
            'change_phi': (phi_value - 1)/50,
            'while_limiter': 1000,
            'phi_val': 1.2,
            'd_rho_val': 0.0,
            'w': interface_width
        }
    }

    sim = Run(
        simulation_type="multiphase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        kappa=kappa,
        rho_l=rho_l,
        rho_v=rho_v,
        interface_width=interface_width,
        save_interval=save_interval,
        bc_config=bc_config,
        force_enabled=True,
        force_obj=gravity,
        phi_value=phi_value,
        d_rho_value=d_rho_value,
        wetting_enabled=True,
        hysteresis_params=bc_config['hysteresis_params'],
        init_type="wetting",
    )

    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    sim_wetting_hysteresis = test_wetting_hysteresis_simulation()

    # Visualize results
    print("\n=== Visualizing Wetting Hysteresis Test Results ===")
    visualise(sim_wetting_hysteresis, "Wetting Hysteresis Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

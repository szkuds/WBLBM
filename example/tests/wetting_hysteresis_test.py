import numpy as np
from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


def wetting_hysteresis_simulation_test():
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

    force_g = 0.0000001
    inclination_angle = 45
    gravity = GravityForceMultiphaseDroplet(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )

    # Add hysteresis parameters to bc_config
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
            'ca_advancing': 90.0,
            'ca_receding': 80.0,
            'learning_rate': 0.01,
            'max_iterations': 20
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
    sim_wetting_hysteresis = wetting_hysteresis_simulation_test()

    # Visualize results
    print("\n=== Visualizing Wetting Hysteresis Test Results ===")
    visualise(sim_wetting_hysteresis, "Wetting Hysteresis Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

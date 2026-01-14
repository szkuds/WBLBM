import numpy as np

from wblbm.operators.force import CompositeForce
from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


def init_for_chem_step_0_inc():
    """Test LBM wetting implementation with hysteresis enabled including a chemical step."""
    print("\n=== Testing LBM Wetting with Hysteresis ===")

    # Simulation parameters
    grid_shape = (201, 101)
    tau = 0.99
    nt = 300000
    save_interval = 10000
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 5

    phi_value = 1.1
    d_rho_value = 0.1

    force_g = 1e-7
    inclination_angle = 0
    gravity = GravityForceMultiphaseDroplet(
        force_g, inclination_angle, grid_shape
    )

    # Add hysteresis parameters to bc_config
    bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'bounce-back',
        'right': 'periodic',
        'wetting_params': {
            'rho_l': rho_l,
            'rho_v': rho_v,
            'phi_left': phi_value,
            'phi_right': phi_value,
            'd_rho_left': d_rho_value,
            'd_rho_right': d_rho_value,
            'width': interface_width
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
        force_obj=[gravity],
        phi_value=phi_value,
        d_rho_value=d_rho_value,
        wetting_enabled=True,
        force_g=force_g,
        inclination_angle=inclination_angle,
        init_type="wetting_chem_step",
        init_dir="/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/results/2025-11-03_chemical_step_example/12-15-56_wetting_simulation_test/data/timestep_199999.npz",
    )

    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    sim_wetting_hysteresis = init_for_chem_step_0_inc()

    # Visualize results
    print("\n=== Visualizing Wetting Hysteresis Test Results ===")
    visualise(sim_wetting_hysteresis, "Wetting Hysteresis Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

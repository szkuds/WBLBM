import numpy as np
from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def wetting_simulation_test():
    """Test wetting implementation with a droplet on a surface under gravity."""
    print("\n=== Testing LBM Wetting Implementation ===")

    # Simulation parameters
    grid_shape = (200, 100)  # nx, ny
    tau = 0.99  # Relaxation time
    nt = 500 # Number of time steps
    save_interval = 100  # Save every 500 steps
    kappa = 0.04  # Surface tension parameter
    rho_l = 1.0  # Liquid density
    rho_v = 0.001  # Vapor density
    interface_width = 5  # Interface width for smooth transition

    # Wetting parameters
    phi_value = 1  # Wetting strength parameter
    d_rho_value = 0.5  # Density adjustment parameter

    # Gravity setup (downward force for droplet settling)
    force_g = 0.0000000  # Small gravity to observe wetting without rapid fall
    inclination_angle = 0  # Vertical gravity
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
        }
    }

    # Initialize and run simulation with wetting enabled
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
        hysteresis_params=None,
        init_type="wetting",
    )

    # Run with wetting initialization
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    sim_wetting = wetting_simulation_test()

    # Visualize results
    print("\n=== Visualizing Wetting Test Results ===")
    visualise(sim_wetting, "Wetting Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

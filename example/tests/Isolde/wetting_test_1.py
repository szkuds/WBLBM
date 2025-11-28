import numpy as np
from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
from wblbm.operators.wetting.contact_angle import ContactAngle
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def test_wetting_simulation():
    """Test wetting implementation with a droplet on a surface under gravity."""
    print("\n=== Testing LBM Wetting Implementation ===")

    # Simulation parameters
    grid_shape = (200, 100)  # nx, ny
    tau = 0.99  # Relaxation time
    nt = 10000  # Number of time steps
    save_interval = 1000  # Save every 500 steps
    kappa = 0.04  # Surface tension parameter
    rho_l = 1.0  # Liquid density
    rho_v = 0.001  # Vapor density
    interface_width = 6  # Interface width for smooth transition

    # Wetting parameters
    phi_value = 0  # Wetting strength parameter
    d_rho_value = 0.6  # Density adjustment parameter

    # Gravity setup (downward force for droplet settling)
    force_g = 0.0000001 # Small gravity to observe wetting without rapid fall
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
            'phi_left': 1.2,
            'phi_right': 1.2,
            'd_rho_left': 0.0,
            'd_rho_right': 0.0,
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
    sim_wetting = test_wetting_simulation()

    # Visualize results
    print("\n=== Visualizing Wetting Test Results ===")
    visualise(sim_wetting, "Wetting Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

#Load saved results
latest_result = ("/Users/isoldeholweg/PycharmProjects/WBLBM/example/tests/Isolde/results/2025-11-28/10-59-26_test_wetting_simulation/data/timestep_9999.npz")
#latest_result = sim.io_handler.data_dir + f"/timestep_{nt-1}.npz"
data = np.load(latest_result)
rho = data['rho']

#Compute contact angle
rho_mean = 0.5 * (1.0 + 0.001)
angle_calc = ContactAngle(rho_mean)
theta_left, theta_right = angle_calc.compute(rho)

print(f"Contact angle (left): {theta_left:.2f}°, {theta_right:.2f}°")

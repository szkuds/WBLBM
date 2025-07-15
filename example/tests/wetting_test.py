import numpy as np
from wblbm import Run, GravityForce, visualise


def test_wetting_simulation():
    """Test wetting implementation with a droplet on a surface under gravity."""
    print("\n=== Testing LBM Wetting Implementation ===")

    # Simulation parameters
    grid_shape = (400, 200)  # nx, ny
    tau = 0.9  # Relaxation time
    nt = 5  # Number of time steps
    save_interval = 1  # Save every 500 steps
    kappa = 0.01  # Surface tension parameter
    rho_l = 1.0  # Liquid density
    rho_v = 0.001  # Vapor density
    interface_width = 5  # Interface width for smooth transition

    # Gravity setup (downward force for droplet settling)
    force_g = 0  # Small gravity to observe wetting without rapid fall
    inclination_angle = 0  # Vertical gravity
    gravity = GravityForce(grid_shape[0], grid_shape[1], 2, force_g, inclination_angle)

    # Boundary conditions: bounce-back at bottom for solid surface
    bc_config = {
        "bottom": "bounce-back",  # Solid wall for wetting
        "top": "symmetry",  # Open top
        "left": "periodic",  # Periodic sides
        "right": "periodic"
    }

    # Initialize and run simulation with wetting enabled
    sim = Run(
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        multiphase=True,
        kappa=kappa,
        rho_l=rho_l,
        rho_v=rho_v,
        interface_width=interface_width,
        save_interval=save_interval,
        bc_config=bc_config,
        force_enabled=True,
        force_obj=gravity,
        wetting_enabled=True,
        hysteresis_params=None
    )

    # Run with wetting initialization
    sim.run(init_type="wetting", verbose=True)
    return sim


if __name__ == "__main__":
    sim_wetting = test_wetting_simulation()

    # Visualize results
    print("\n=== Visualizing Wetting Test Results ===")
    visualise(sim_wetting, "Wetting Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

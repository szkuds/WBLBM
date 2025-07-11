import numpy as np
from wblbm.operators.run.run import Run
from wblbm.force.gravitational_force import GravityForce
from wblbm.utils.plotting import visualise


def test_multiphase_gravity_simulation():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation with Gravity Test ===")

    grid_shape = (200, 200)
    tau = 0.9
    nt = 5000
    save_interval = 100
    kappa = 0.01
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 10

    force_g = 0.000005
    inclination_angle = 0
    gravity = GravityForce(grid_shape[0], grid_shape[1], 2, force_g, inclination_angle)

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
        force_enabled=True,
        force_obj=gravity
    )
    sim.run(init_type='multiphase_bubble', verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = test_multiphase_gravity_simulation()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase with Gravity Force")

    print("\nTest completed!")

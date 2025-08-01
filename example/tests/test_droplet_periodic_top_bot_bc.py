from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphaseDroplet
from wblbm.utils.plotting import visualise


def test_multiphase_gravity_simulation():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation with Gravity Test ===")

    grid_shape = (200, 800)
    tau = 0.9
    nt = 100000
    save_interval = 10000
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 10

    force_g = 0.0000002
    inclination_angle = 45
    gravity = GravityForceMultiphaseDroplet(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )

    bc_config = {
        "top": "periodic",
        "bottom": "periodic",
        "left": "periodic",
        "right": "periodic",
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
        init_type="multiphase_droplet",
    )
    sim.run(verbose=True)
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

from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphaseDroplet
from wblbm.utils.plotting import visualise

def droplet_periodic_top_bot_bc_test():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation with Gravity Test ===")

    grid_shape = (401, 401)
    tau = 0.9
    nt = 40000
    save_interval = 10
    init_file = "/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/results/2025-10-22/10-38-04_droplet_periodic_top_bot_bc_test/data/timestep_800.npz"

    kappa = 0.08
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 10

    force_g = 0.0000005
    inclination_angle = 0
    gravity = GravityForceMultiphaseDroplet(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )
    # Specify MRT collision operator and its rates
    collision = {
        "collision_scheme": "mrt",
        "kv": 1.05,
        "kb": 1.0,
        "k0": 0.0,
        "k2": 1.0,
        "k4": 0.9,
    }
    bc_config = {
        "top": "bounce-back",
        "bottom": "bounce-back",
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
        collision=collision,
        force_enabled=True,
        force_obj=gravity,
        init_type="init_from_file",
        init_dir=init_file
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = droplet_periodic_top_bot_bc_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase with Gravity Force")

    print("\nTest completed!")

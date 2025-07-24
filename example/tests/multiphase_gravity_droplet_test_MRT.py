from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphaseDroplet
from wblbm.utils.plotting import visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)


def test_multiphase_gravity_simulation():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation with Gravity Test ===")

    grid_shape = (200, 800)
    nt = 20000
    save_interval = 1000
    kappa = 0.06
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 12

    force_g = 0.000002
    inclination_angle = 0
    gravity = GravityForceMultiphaseDroplet(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )

    # Specify MRT collision operator and its rates
    collision = {
        "collision_scheme": "mrt",
        "kv": 1 / 0.9,
        "kb": 1.2,
        "k0": 0.0,
        "k2": 1.2,
        "k4": 1.1,
    }

    sim = Run(
        simulation_type="multiphase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        nt=nt,
        kappa=kappa,
        rho_l=rho_l,
        rho_v=rho_v,
        interface_width=interface_width,
        save_interval=save_interval,
        force_enabled=True,
        force_obj=gravity,
        collision=collision,
    )
    sim.run(init_type="multiphase_droplet", verbose=True)
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

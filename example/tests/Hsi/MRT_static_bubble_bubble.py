from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphaseDroplet
from wblbm.utils.plotting import visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)


def test_mrt_static():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation of a static bubble ===")

    grid_shape = (401, 401)
    nt = 50000
    save_interval = 2500
    skip_interval = 0
    kappa = 0.01
    rho_l = 1.0
    rho_v = 0.1
    interface_width = 5
    tau = 0.9

    force_g = 0.00000
    inclination_angle = 0
    gravity = GravityForceMultiphaseDroplet(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )

    bc_config = {
        "top": "periodic",
        "bottom": "periodic",
        "left": "bounce-back",
        "right": "bounce-back",
    }

    # Specify MRT collision operator and its rates
    collision = {
        "collision_scheme": "mrt",
        "kv": 1.05,
        "kb": 1.0,
        "k0": 0.0,
        "k2": 1.0,
        "k4": 0.9,
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
        skip_interval=skip_interval,
        force_enabled=True,
        force_obj=gravity,
        collision=collision,
        init_type="multiphase_lateral_bubble_configuration",
        tau=tau,
        #bc_config=bc_config,
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = test_mrt_static()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase with Gravity Force")

    print("\nTest completed!")

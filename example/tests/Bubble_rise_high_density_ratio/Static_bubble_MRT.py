from wblbm import Run
from wblbm import GravityForceMultiphaseDroplet
from wblbm.utils.plotting import visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)

def mrt_static_test():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation of a static bubble ===")

    grid_shape = (201, 801)
    nt = 5000
    save_interval = 500
    skip_interval = 0
    kappa = 0.01
    rho_l = 1
    rho_v = 0.001
    interface_width = 5
    tau = 0.9

    bc_config = {
        "top": "bounce-back",
        "bottom": "bounce-back",
        "left": "periodic",
        "right": "periodic",
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
        init_type="multiphase_droplet",
        tau=tau,
        collision=collision,
        bc_config=bc_config,
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = mrt_static_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase with Gravity Force")

    print("\nTest completed!")

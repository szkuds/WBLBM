import numpy as np
from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphaseBubble
from wblbm import visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)


def rising_bubble_mrt():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation of a rising bubble ===")

    grid_shape = (401, 401)
    nt = 50000
    save_interval = 5000
    init_file = "/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/Bubble_rise_high_density_ratio/results/2025-08-05/16-24-37/data/timestep_45000.npz"

    kappa = 0.01
    rho_l = 1
    rho_v = .001
    interface_width = 5

    force_g = 0.000002
    inclination_angle = 0
    gravity = GravityForceMultiphaseBubble(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )

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
        force_enabled=True,
        force_obj=gravity,
        init_type="multiphase_bubble",
        init_dir=init_file,
        #tau=.9,
        collision=collision,
        #bc_config=bc_config,
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = rising_bubble_mrt()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase with Gravity Force")

    print("\nTest completed!")

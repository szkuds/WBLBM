import numpy as np
from wblbm.run import Run
from wblbm import GravityForceSinglephase
from wblbm import GravityForceMultiphase
from wblbm import visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def single_phase_gravity_simulation_test():
    """Test a single-phase LBM simulation with gravity."""
    print("\n=== Single-Phase LBM Simulation with Gravity Test ===")

    grid_shape = (100, 100)
    tau = .6
    nt = 20000
    save_interval = 2000

    force_g = 0.000001
    inclination_angle = 0
    gravity = GravityForceSinglephase(
        grid_shape[0], grid_shape[1], 2, force_g, inclination_angle
    )

    sim = Run(
        simulation_type="singlephase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        save_interval=save_interval,
        force_enabled=True,
        force_obj=gravity,
        init_type="standard"
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Single-Phase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_single_phase_gravity = single_phase_gravity_simulation_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_single_phase_gravity, "Single-Phase with Gravity Force")

    print("\nTest completed!")

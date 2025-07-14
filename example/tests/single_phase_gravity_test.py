import numpy as np
from wblbm.operators.run.run import Run
from wblbm.force.gravitational_force import GravityForce
from wblbm.utils.plotting import visualise


def test_single_phase_gravity_simulation():
    """Test a single-phase LBM simulation with gravity."""
    print("\n=== Single-Phase LBM Simulation with Gravity Test ===")

    grid_shape = (200, 200)
    tau = 1.0
    nt = 10000
    save_interval = 2000

    force_g = 0.01
    inclination_angle = 0
    gravity = GravityForce(grid_shape[0], grid_shape[1], 2, force_g, inclination_angle)

    # Add bounce-back boundary conditions to left and right
    bc_config = {"left": "bounce-back", "right": "bounce-back"}

    sim = Run(
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        multiphase=False,
        save_interval=save_interval,
        force_enabled=True,
        force_obj=gravity,
        bc_config=bc_config,
    )
    sim.run(init_type="standard", verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Single-Phase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_single_phase_gravity = test_single_phase_gravity_simulation()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_single_phase_gravity, "Single-Phase with Gravity Force")

    print("\nTest completed!")

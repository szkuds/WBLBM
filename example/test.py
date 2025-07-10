import numpy as np
from wblbm.operators.run.run import Run
from wblbm.utils.plotting import visualise  # Import the new function


def test_basic_simulation():
    """Test a basic single-phase LBM simulation."""
    print("\n=== Basic LBM Simulation Test ===")
    sim = Run(
        grid_shape=(100, 50),
        lattice_type="D2Q9",
        tau=1.0,
        nt=500,
        save_interval=100,
        multiphase=False
    )
    sim.run(init_type='standard', verbose=True)
    return sim


def test_multiphase_simulation():
    """Test a multiphase LBM simulation with a central bubble."""
    print("\n=== Multiphase LBM Simulation Test ===")
    sim = Run(
        grid_shape=(400, 400),
        lattice_type="D2Q9",
        tau=0.9,
        nt=500000,
        save_interval=1000,
        multiphase=True,
        kappa=0.01,
        rho_l=1.0,
        rho_v=0.001,
        interface_width=10
    )
    sim.run(init_type='multiphase_droplet', verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing LBM Codebase with Refactored IO and Initialization")
    print("=" * 60)

    # Run simulations
    sim_basic = test_basic_simulation()
    sim_multiphase = test_multiphase_simulation()

    # Visualize results using the new, centralized function
    print("\n=== Visualizing Results ===")
    visualise(sim_basic, "Basic LBM Simulation")
    visualise(sim_multiphase, "Multiphase LBM Simulation")

    print("\nAll tests completed!")

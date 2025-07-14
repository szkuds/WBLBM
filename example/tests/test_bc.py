import numpy as np
from wblbm import Run
from wblbm import visualise


def test_boundary_conditions():
    """
    Test various boundary conditions in the LBM simulation.

    This function configures a small multiphase simulation with custom boundary conditions,
    runs it for a limited number of steps, and visualizes the results to verify BC behavior.
    """
    print("\n=== Testing LBM Boundary Conditions ===")

    # Define boundary configuration
    bc_config = {
        "top": "bounce-back",  # No-slip wall at top
        "bottom": "bounce-back",  # Symmetric boundary at bottom
        "left": "bounce-back",  # Periodic left-right wrapping
        "right": "bounce-back",
    }

    # Set up the simulation
    sim = Run(
        grid_shape=(200, 200),  # Small grid for quick testing
        lattice_type="D2Q9",
        tau=0.8,
        nt=10000,  # Short run to observe BC effects
        multiphase=True,
        kappa=0.01,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=5,
        save_interval=2000,
        bc_config=bc_config,  # Apply custom BCs
    )

    # Run the simulation with a droplet initialization to test interactions with boundaries
    sim.run(init_type="multiphase_bubble", verbose=True)

    # Visualize results
    print("\n=== Visualizing Boundary Condition Effects ===")
    visualise(sim, "LBM Boundary Condition Test")

    print("\nTest completed! Check the 'results' directory for saved data and plots.")


if __name__ == "__main__":
    test_boundary_conditions()

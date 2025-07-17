from wblbm.run import Run
from wblbm.operators.force import Force
from wblbm.utils.plotting import visualise
import jax.numpy as jnp


class UniformForce(Force):
    """
    A uniform force class that applies a constant force in the x-direction
    across the entire domain.
    """

    def __init__(self, nx: int, ny: int, d: int, force_magnitude: float):
        """
        Initialize uniform force in x-direction.

        Args:
            nx (int): Grid size in x-direction
            ny (int): Grid size in y-direction
            d (int): Number of dimensions (should be 2)
            force_magnitude (float): Magnitude of force in x-direction
        """
        if d != 2:
            raise ValueError("Currently supports 2D (d=2) only")

        # Create force array with uniform force in x-direction
        force_array = jnp.zeros((nx, ny, 1, d))
        force_array = force_array.at[:, :, 0, 0].set(
            force_magnitude
        )  # Force in x-direction
        force_array = force_array.at[:, :, 0, 1].set(0.0)  # No force in y-direction

        super().__init__(force_array)

    def compute_force(self, rho: jnp.ndarray, rhol: float, rhov: float) -> jnp.ndarray:
        """
        Compute the uniform force field.

        Args:
            rho: Density field (not used for uniform force)
            rhol: Liquid density (not used for uniform force)
            rhov: Vapor density (not used for uniform force)

        Returns:
            jnp.ndarray: Uniform force field
        """
        return self.force


def test_boundary_conditions_uniform_force():
    """
    Test boundary conditions with uniform force in x-direction.

    This test applies:
    - Uniform force in x-direction
    - Bounce-back boundary conditions at top and bottom
    - Periodic boundary conditions on left and right
    """
    print("Testing LBM Boundary Conditions with Uniform Force")
    print("=" * 60)

    # Simulation parameters
    grid_shape = (200, 100)  # (nx, ny)
    tau = 1.0
    nt = 100
    save_interval = 10
    force_magnitude = 0.001  # Uniform force in x-direction

    # Create uniform force object
    uniform_force = UniformForce(
        nx=grid_shape[0], ny=grid_shape[1], d=2, force_magnitude=force_magnitude
    )

    # Define boundary conditions
    bc_config = {
        "top": "bounce-back",  # No-slip wall at top
        "bottom": "bounce-back",  # No-slip wall at bottom
        "left": "periodic",  # Periodic boundary on left
        "right": "periodic",  # Periodic boundary on right
    }

    print(f"Grid shape: {grid_shape}")
    print(f"Uniform force magnitude (x-direction): {force_magnitude}")
    print(f"Boundary conditions: {bc_config}")
    print(f"Simulation steps: {nt}")
    print()

    # Set up the simulation
    sim = Run(
        simulation_type="singlephase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        save_interval=save_interval,
        bc_config=bc_config,
        force_enabled=True,
        force_obj=uniform_force,
    )

    # Run the simulation with standard initialization
    sim.run(init_type="standard", verbose=True)

    return sim


if __name__ == "__main__":
    print("Testing LBM Boundary Conditions with Uniform Force")
    print("=" * 60)

    # Run the test
    sim = test_boundary_conditions_uniform_force()

    # Visualize results
    print("\nVisualizing Results...")
    visualise(sim, "Boundary Conditions Test - Uniform Force")

    print("\nTest completed! Check the results directory for saved data and plots.")

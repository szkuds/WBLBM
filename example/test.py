# test_run.py
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from wblbm.operators.run.run import Run


def test_basic_simulation():
    """Test basic LBM simulation"""
    print("=== Basic LBM Simulation Test ===")

    # Create simulation
    sim = Run(
        grid_shape=(100, 50),
        lattice_type="D2Q9",
        tau=1.0,
        nt=500,
        save_interval=50,
        multiphase=False
    )

    # Define initial conditions
    initial_density = 1.0
    initial_velocity = jnp.array([0.01, 0.0])  # Small initial velocity in x-direction

    # Run simulation
    results = sim.run(
        initial_density=initial_density,
        initial_velocity=initial_velocity,
        verbose=True
    )

    # Get final state
    final_rho, final_u = sim.get_final_state()

    # Print some statistics
    print(f"Final average density: {np.mean(final_rho):.4f}")
    print(f"Final max velocity: {np.max(np.sqrt(final_u[..., 0] ** 2 + final_u[..., 1] ** 2)):.6f}")

    # Save results
    sim.save_results("basic_simulation_results.npz")

    return sim, results


def test_multiphase_simulation():
    """Test multiphase LBM simulation"""
    print("\n=== Multiphase LBM Simulation Test ===")

    # Create multiphase simulation
    sim = Run(
        grid_shape=(80, 40),
        lattice_type="D2Q9",
        tau=1.0,
        nt=300,
        save_interval=50,
        multiphase=True,
        kappa=0.1,
        beta=1.0,
        rho_l=1.0,
        rho_v=0.1
    )

    # Custom initialization for multiphase (create a bubble)
    def custom_multiphase_init():
        f = sim.initialise.initialise_population(1.0)
        f = jnp.array(f)

        # Create a density field with a bubble in the center
        nx, ny = sim.grid.nx, sim.grid.ny
        x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        center_x, center_y = nx // 2, ny // 2
        radius = min(nx, ny) // 4

        # Create bubble (low density region)
        distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        rho_field = jnp.where(distance < radius, 0.2, 0.8)

        # Adjust population based on density
        for i in range(sim.lattice.q):
            f = f.at[:, :, i, 0].set(f[:, :, i, 0] * rho_field)

        return f

    # Run simulation with custom initialization
    results = sim.run(
        custom_init=custom_multiphase_init,
        verbose=True
    )

    # Get final state
    final_rho, final_u = sim.get_final_state()

    # Print some statistics
    print(f"Final average density: {np.mean(final_rho):.4f}")
    print(f"Final density range: [{np.min(final_rho):.4f}, {np.max(final_rho):.4f}]")
    print(f"Final max velocity: {np.max(np.sqrt(final_u[..., 0] ** 2 + final_u[..., 1] ** 2)):.6f}")

    # Save results
    sim.save_results("multiphase_simulation_results.npz")

    return sim, results


def test_with_callback():
    """Test simulation with custom output callback"""
    print("\n=== Simulation with Custom Callback ===")

    # Custom callback function
    def my_callback(iteration, rho, u, f):
        if iteration % 100 == 0:
            avg_rho = np.mean(rho)
            max_vel = np.max(np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2))
            print(f"  Callback at step {iteration}: avg_rho={avg_rho:.4f}, max_vel={max_vel:.6f}")

    # Create simulation with callback
    sim = Run(
        grid_shape=(60, 30),
        lattice_type="D2Q9",
        tau=1.0,
        nt=400,
        save_interval=100,
        output_callback=my_callback
    )

    # Run simulation
    results = sim.run(initial_density=1.0, verbose=True)

    return sim, results


def visualize_results(sim, results, title="LBM Simulation Results"):
    """Visualize simulation results"""
    try:
        import matplotlib.pyplot as plt

        # Get final state
        final_rho, final_u = sim.get_final_state()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot density
        im1 = axes[0].imshow(final_rho[:, :, 0, 0].T, origin='lower', cmap='viridis')
        axes[0].set_title('Final Density')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])

        # Plot velocity magnitude
        vel_mag = np.sqrt(final_u[:, :, 0, 0] ** 2 + final_u[:, :, 0, 1] ** 2)
        im2 = axes[1].imshow(vel_mag.T, origin='lower', cmap='plasma')
        axes[1].set_title('Velocity Magnitude')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])

        # Plot velocity field
        x, y = np.meshgrid(np.arange(sim.grid.nx), np.arange(sim.grid.ny))
        stride = max(1, sim.grid.nx // 20)  # Adjust arrow density
        axes[2].quiver(
            x[::stride, ::stride], y[::stride, ::stride],
            final_u[::stride, ::stride, 0, 0].T,
            final_u[::stride, ::stride, 0, 1].T,
            vel_mag[::stride, ::stride].T,
            cmap='plasma'
        )
        axes[2].set_title('Velocity Field')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_aspect('equal')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == "__main__":
    # Run tests
    print("Testing LBM Run Class")
    print("=" * 50)

    # Test 1: Basic simulation
    sim1, results1 = test_basic_simulation()

    # Test 2: Multiphase simulation
    sim2, results2 = test_multiphase_simulation()

    # Test 3: Simulation with callback
    sim3, results3 = test_with_callback()

    # Visualize results (if matplotlib is available)
    print("\n=== Visualization ===")
    visualize_results(sim1, results1, "Basic LBM Simulation")
    visualize_results(sim2, results2, "Multiphase LBM Simulation")

    print("\nAll tests completed successfully!")

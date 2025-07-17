import time
import numpy as np
import matplotlib.pyplot as plt
from wblbm.run import Run


def complexity_analysis():
    """Analyze how timing scales with grid size."""

    grid_sizes = [(50, 50), (100, 100), (150, 150), (200, 200), (1000, 1000)]
    timing_results = []

    for nx, ny in grid_sizes:
        print(f"\nTesting grid size: {nx}x{ny}")

        sim = Run(
            simulation_type="multiphase",
            grid_shape=(nx, ny),
            lattice_type="D2Q9",
            tau=1.0,
            nt=10,  # Just a few steps for timing
            kappa=0.01,
            rho_l=1.0,
            rho_v=0.1,
            interface_width=4,
            save_interval=10,
            bc_config={"top": "symmetry", "bottom": "bounce-back"},  # Example BCs
        )

        f_prev = sim.simulation.initialize_fields("multiphase_bubble")

        # Warm up
        for _ in range(3):
            f_next = sim.simulation.run_timestep(f_prev, 0)
            if hasattr(sim.simulation, "boundary_condition"):
                f_next = sim.simulation.boundary_condition(f_next, f_next)
            f_prev = f_next

        # Time multiple iterations
        times = []
        for _ in range(10):
            start = time.perf_counter()
            f_next = sim.simulation.run_timestep(f_prev, 0)
            if hasattr(sim.simulation, "boundary_condition"):
                f_next = sim.simulation.boundary_condition(f_next, f_next)
            if hasattr(f_next, "block_until_ready"):
                f_next.block_until_ready()
            end = time.perf_counter()
            times.append(end - start)
            f_prev = f_next

        avg_time = np.mean(times) * 1000  # Convert to ms
        grid_points = nx * ny

        timing_results.append(
            {
                "grid_size": f"{nx}x{ny}",
                "grid_points": grid_points,
                "avg_time_ms": avg_time,
                "time_per_point_us": (avg_time * 1000) / grid_points,
            }
        )

        print(f"Average time per iteration: {avg_time:.3f} ms")
        print(f"Time per grid point: {(avg_time * 1000) / grid_points:.3f} μs")

    # Display results
    print("\n=== Complexity Analysis Results ===")
    print(f"{'Grid Size':<12} {'Points':<10} {'Time (ms)':<12} {'μs/point':<12}")
    print("-" * 50)

    for result in timing_results:
        print(
            f"{result['grid_size']:<12} {result['grid_points']:<10} "
            f"{result['avg_time_ms']:<12.3f} {result['time_per_point_us']:<12.3f}"
        )


if __name__ == "__main__":
    complexity_analysis()

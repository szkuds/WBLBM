import time
import numpy as np
from wblbm.run import Run
from wblbm import time_function, TIMING_ENABLED


def detailed_timing_analysis():
    """Run a small simulation with detailed timing for each component."""

    print("=== Detailed Timing Analysis ===")

    sim = Run(
        simulation_type="multiphase",
        grid_shape=(1000, 1000),
        lattice_type="D2Q9",
        tau=1.0,
        nt=100,
        kappa=0.01,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=4,
        save_interval=50,
        bc_config={"top": "symmetry", "bottom": "bounce-back"},
    )

    # Initialize
    f_prev = sim.simulation.initialize_fields("multiphase_bubble")

    # Warm up JAX compilation
    print("Warming up JAX compilation...")
    for _ in range(5):
        f_next = sim.simulation.run_timestep(f_prev, 0)
        if hasattr(sim.simulation, "boundary_condition"):
            f_next = sim.simulation.boundary_condition(f_next, f_next)
        f_prev = f_next

    print("\nStarting detailed timing analysis...")

    total_times = {
        "macroscopic": [],
        "equilibrium": [],
        "collision": [],
        "streaming": [],
        "total_update": [],
    }

    for i in range(50):
        start_total = time.perf_counter()

        # Macroscopic calculation
        start = time.perf_counter()
        rho, u, force = sim.simulation.macroscopic(f_prev)
        if hasattr(rho, "block_until_ready"):
            rho.block_until_ready()
        macro_time = time.perf_counter() - start

        # Equilibrium calculation
        start = time.perf_counter()
        feq = sim.simulation.update.equilibrium(rho, u)
        if hasattr(feq, "block_until_ready"):
            feq.block_until_ready()
        eq_time = time.perf_counter() - start

        # Source term
        start = time.perf_counter()
        source = sim.simulation.update.source_term(rho, u, force)
        if hasattr(source, "block_until_ready"):
            source.block_until_ready()
        source_time = time.perf_counter() - start

        # CollisionBGK
        start = time.perf_counter()
        fcol = sim.simulation.update.collision(f_prev, feq, source)
        if hasattr(fcol, "block_until_ready"):
            fcol.block_until_ready()
        collision_time = time.perf_counter() - start

        # Streaming
        start = time.perf_counter()
        fstream = sim.simulation.update.streaming(fcol)
        if hasattr(fstream, "block_until_ready"):
            fstream.block_until_ready()
        stream_time = time.perf_counter() - start

        # Apply boundary conditions if present
        if hasattr(sim.simulation, "boundary_condition"):
            fstream = sim.simulation.boundary_condition(fstream, fcol)
            if hasattr(fstream, "block_until_ready"):
                fstream.block_until_ready()
        total_time = time.perf_counter() - start_total

        total_times["macroscopic"].append(macro_time + source_time)
        total_times["equilibrium"].append(eq_time)
        total_times["collision"].append(collision_time)
        total_times["streaming"].append(stream_time)
        total_times["total_update"].append(total_time)

        f_prev = fstream

        if i % 10 == 0:
            print(f"Completed {i + 1}/50 timing iterations")

    print("\n=== Timing Results (milliseconds) ===")
    print(
        f"{'Component':<15} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'% of Total':<12}"
    )
    print("-" * 70)

    total_mean = np.mean(total_times["total_update"]) * 1000

    for component, times in total_times.items():
        if component != "total_update":
            times_ms = np.array(times) * 1000
            mean_time = np.mean(times_ms)
            std_time = np.std(times_ms)
            min_time = np.min(times_ms)
            max_time = np.max(times_ms)
            percentage = (mean_time / total_mean) * 100

            print(
                f"{component:<15} {mean_time:<8.3f} {std_time:<8.3f} {min_time:<8.3f} {max_time:<8.3f} {percentage:<12.1f}"
            )

    times_ms = np.array(total_times["total_update"]) * 1000
    mean_time = np.mean(times_ms)
    std_time = np.std(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)

    print("-" * 70)
    print(
        f"{'TOTAL':<15} {mean_time:<8.3f} {std_time:<8.3f} {min_time:<8.3f} {max_time:<8.3f} {'100.0':<12}"
    )


if __name__ == "__main__":
    detailed_timing_analysis()

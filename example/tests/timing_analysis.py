import time
import numpy as np
from wblbm import Run
from wblbm import time_function, TIMING_ENABLED


def detailed_timing_analysis():
    """Run a small simulation with detailed timing for each component."""

    print("=== Detailed Timing Analysis ===")

    # Use a moderate grid size for meaningful timing
    sim = Run(
        grid_shape=(1000, 1000),
        lattice_type="D2Q9",
        tau=1.0,
        nt=100,  # Fewer steps for timing analysis
        multiphase=True,
        kappa=0.01,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=4,
        save_interval=50,
        bc_config={"top": "symmetry", "bottom": "bounce-back"},  # Example BCs
    )

    # Initialize
    f_prev = sim.initialiser.initialise_multiphase_bubble(
        sim.rho_l, sim.rho_v, sim.interface_width
    )

    # Warm up JAX compilation
    print("Warming up JAX compilation...")
    for _ in range(5):
        f_next = sim.update(f_prev)
        # Apply boundary conditions if present
        if hasattr(sim, "boundary_condition") and sim.boundary_condition is not None:
            f_next = sim.boundary_condition(f_next, f_next)
        f_prev = f_next

    print("\nStarting detailed timing analysis...")

    # Time individual components
    total_times = {
        "macroscopic": [],
        "equilibrium": [],
        "collision": [],
        "streaming": [],
        "total_update": [],
    }

    for i in range(50):  # Time 50 iterations
        start_total = time.perf_counter()

        # Time macroscopic calculation
        start = time.perf_counter()
        rho, u, force = sim.macroscopic_multiphase(f_prev)
        if hasattr(rho, "block_until_ready"):
            rho.block_until_ready()
        macro_time = time.perf_counter() - start

        # Time equilibrium calculation
        start = time.perf_counter()
        feq = sim.update.equilibrium(rho, u)
        if hasattr(feq, "block_until_ready"):
            feq.block_until_ready()
        eq_time = time.perf_counter() - start

        # Time source term
        start = time.perf_counter()
        source = sim.update.source_term(rho, u, force)
        if hasattr(source, "block_until_ready"):
            source.block_until_ready()
        source_time = time.perf_counter() - start

        # Time collision
        start = time.perf_counter()
        fcol = sim.update.collision(f_prev, feq, source)
        if hasattr(fcol, "block_until_ready"):
            fcol.block_until_ready()
        collision_time = time.perf_counter() - start

        # Time streaming
        start = time.perf_counter()
        fstream = sim.update.streaming(fcol)
        if hasattr(fstream, "block_until_ready"):
            fstream.block_until_ready()
        stream_time = time.perf_counter() - start

        # Apply boundary conditions if present
        if hasattr(sim, "boundary_condition") and sim.boundary_condition is not None:
            fstream = sim.boundary_condition(fstream, fcol)
            if hasattr(fstream, "block_until_ready"):
                fstream.block_until_ready()
        total_time = time.perf_counter() - start_total

        # Store times
        total_times["macroscopic"].append(
            macro_time + source_time
        )  # Include source term
        total_times["equilibrium"].append(eq_time)
        total_times["collision"].append(collision_time)
        total_times["streaming"].append(stream_time)
        total_times["total_update"].append(total_time)

        f_prev = fstream

        if i % 10 == 0:
            print(f"Completed {i + 1}/50 timing iterations")

    # Calculate and display statistics
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

    # Total
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

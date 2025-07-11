from wblbm.operators.run.run import Run


def profile_lbm():
    """Profile a small LBM simulation to identify bottlenecks."""

    # Use a smaller grid for profiling to focus on computational patterns
    sim = Run(
        grid_shape=(50, 50),  # Smaller grid for profiling
        lattice_type="D2Q9",
        tau=1.0,
        nt=100,  # Only profile 100 steps
        multiphase=True,  # Profile the more complex multiphase case
        kappa=0.01,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=4,
        save_interval=50
    )

    # Run with profiling
    sim.run_with_profiling(init_type='multiphase_bubble', profile_steps=100)


if __name__ == "__main__":
    profile_lbm()

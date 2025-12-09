from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphaseBubble
from wblbm.utils.plotting import visualise
import jax

# this line is added for debugging
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def mrt_rising_cs_test():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation of a static bubble ===")

    grid_shape = (201, 801)
    nt = 100000
    save_interval = 5000
    skip_interval = 0
    kappa = 0.008
    rho_l = 12.18
    rho_v = 0.015
    interface_width = 8
    init_file = "/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/Bubble_rise_high_density_ratio_CS/results/2025-12-09/00-38-49_mrt_static_cs_test/data/timestep_99999.npz"

    force_g = 1e-8
    inclination_angle = 0
    gravity = GravityForceMultiphaseBubble(
        force_g, inclination_angle, grid_shape
    )

    bc_config = {
        "top": "periodic",
        "bottom": "periodic",
        "left": "periodic",
        "right": "periodic",
    }

    # Specify MRT collision operator and its rates
    collision = {
        "collision_scheme": "mrt",
        "kv": 1.05,
        "kb": 1.0,
        "k0": 0.0,
        "k2": 1.0,
        "k4": 0.9,
    }

    # Maxwell construction for Carnahan-Starling EOS
    rho_c = 3.5
    p_c = 0.001
    Tr = 0.5

    # Calculate EOS parameters
    b_eos = 0.5218 / rho_c
    a_eos = ((b_eos ** 2) * p_c) / ((0.3773 ** 2) * 0.4963)
    R_eos = 1.0
    Tc = (0.3773 * a_eos) / (b_eos * R_eos)
    T_eos = Tr * Tc

    sim = Run(
        simulation_type="multiphase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        nt=nt,
        kappa=kappa,
        rho_l=rho_l,
        rho_v=rho_v,
        bubble=False,
        interface_width=interface_width,
        save_interval=save_interval,
        skip_interval=skip_interval,
        force_enabled=True,
        force_obj=[gravity],
        collision=collision,
        init_type="init_from_file",
        init_dir=init_file,
        bc_config=bc_config,
        rho_ref=rho_l,
        g=force_g,
        eos="carnahan-starling",
        a_eos=a_eos,
        b_eos=b_eos,
        r_eos=R_eos,
        t_eos=T_eos,
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = mrt_rising_cs_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase with Gravity Force")

    print("\nTest completed!")

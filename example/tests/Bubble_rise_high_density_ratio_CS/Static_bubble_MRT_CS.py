from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphase
from wblbm.utils.plotting import visualise
import jax


# this line is added for debugging
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def mrt_static_cs_test():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation of a static bubble ===")

    grid_shape = (201, 801)
    nt = 200000
    save_interval = 1000
    skip_interval = 0
    init_file = '/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/Bubble_rise_high_density_ratio_CS/results/2025-12-09/00-02-25_mrt_static_cs_test/data/timestep_35000.npz'

    # Specify MRT collision operator and its rates
    collision = {
        "collision_scheme": "mrt",
        "kv": 1.1,
        "kb": 0.5,
        "k0": 0.0,
        "k2": .6,
        "k4": 1.2,
    }

    bc_config = {
        "top": "bounce-back",
        "bottom": "bounce-back",
        "left": "bounce-back",
        "right": "bounce-back",
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
        interface_width=interface_width,
        save_interval=save_interval,
        skip_interval=skip_interval,
        collision=collision,
        init_type="multiphase_bubble",
        init_dir=init_file,
        eos="carnahan-starling",
        a_eos=a_eos,
        b_eos=b_eos,
        r_eos=R_eos,
        t_eos=T_eos,
        bc_config=bc_config
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = mrt_static_cs_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase LBM Simulation of a static bubble")

    print("\nTest completed!")

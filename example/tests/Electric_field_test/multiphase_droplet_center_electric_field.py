from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphase, ElectricForce
from wblbm.utils.plotting import visualise
import jax

# this line is added for debugging
#jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def multiphase_gravity_simulation_test():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation with Gravity Test ===")
    # simulation config
    grid_shape = (200, 800)
    lattice_type = "D2Q9"
    nt = 10000
    save_interval = 1000

    # multiphase config
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 8

    # BGK config
    tau = 0.9

    # Electric field config
    permittivity_liquid = 1.0
    permittivity_vapour = 1.0
    conductivity_liquid = 1.0
    conductivity_vapour = 1.0

    # Gravitational force config
    force_g = 0.000002
    inclination_angle = 0

    # setting up the forces
    gravity = GravityForceMultiphase(
        force_g, inclination_angle, grid_shape
    )

    electric = ElectricForce(
        permittivity_liquid=permittivity_liquid,
        permittivity_vapour=permittivity_vapour,
        conductivity_liquid=conductivity_liquid,
        conductivity_vapour=conductivity_vapour,
        grid_shape=grid_shape
    )

    sim = Run(
        simulation_type="multiphase",
        grid_shape=grid_shape,
        lattice_type=lattice_type,
        tau=tau,
        nt=nt,
        kappa=kappa,
        rho_l=rho_l,
        rho_v=rho_v,
        interface_width=interface_width,
        save_interval=save_interval,
        force_enabled=True,
        force_obj=[gravity, electric],
        init_type="multiphase_droplet",
    )
    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    print("Testing Multiphase LBM Codebase with Gravity")
    print("=" * 60)

    # Run simulation
    sim_multiphase_gravity = multiphase_gravity_simulation_test()

    # Visualize results
    print("\n=== Visualizing Results ===")
    visualise(sim_multiphase_gravity, "Multiphase with Gravity Force")

    print("\nTest completed!")

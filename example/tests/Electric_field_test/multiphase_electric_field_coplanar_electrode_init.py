from wblbm.run import Run
from wblbm.operators.force import GravityForceMultiphaseDroplet, ElectricForce
from wblbm.utils.plotting import visualise
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# this line is added for debugging
#jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def multiphase_gravity_simulation_test():
    """Test a multiphase LBM simulation with gravity and a central droplet."""
    print("\n=== Multiphase LBM Simulation with Gravity Test ===")
    # simulation config
    grid_shape = (201, 101)
    lattice_type = "D2Q9"
    nt = 200000
    save_interval = 10000

    # multiphase config
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 5

    # BGK config
    tau = 0.99

    # Electric field config
    permittivity_liquid = 1
    permittivity_vapour = .01
    conductivity_liquid = 1
    conductivity_vapour = .001
    U_0 = 8e-2

    bc_config = {
        "top": "bounce-back",
        "bottom": "bounce-back",
        "left": "bounce-back",
        "right": "bounce-back",
    }

    # Gravitational force config
    force_g = 1e-7
    inclination_angle = 0

    # setting up the forces
    gravity = GravityForceMultiphaseDroplet(
        force_g, inclination_angle, grid_shape
    )

    electric = ElectricForce(
        permittivity_liquid=permittivity_liquid,
        permittivity_vapour=permittivity_vapour,
        conductivity_liquid=conductivity_liquid,
        conductivity_vapour=conductivity_vapour,
        grid_shape=grid_shape,
        lattice_type=lattice_type,
        U_0=U_0
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
        force_obj=[gravity],
        init_type="wetting_chem_step",
        init_dir='/Users/sbszkudlarek/TUD_LBM/results/2026-01-09/Electric_field_working_example_high_grav/11-42-14_multiphase_gravity_simulation_test/data/timestep_199999.npz',
        permittivity_liquid=permittivity_liquid,
        permittivity_vapour=permittivity_vapour,
        conductivity_liquid=conductivity_liquid,
        conductivity_vapour=conductivity_vapour,
        U_0=U_0,
        bc_config=bc_config


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
#
# #Check to see if U correct
#extract h_i from data
data_dir = sim_multiphase_gravity.io_handler.data_dir
print(data_dir)

t = 19999  # last saved timestep (nt=100, save_interval=10)
file = f"{data_dir}/timestep_{t}.npz"

data = jnp.load(file)
print(data.files)

h_i = data["h"]  # (nx, ny, q, 1)
U = jnp.sum(h_i, axis=2)[:, :, 0]  # (nx, ny)
print("U=", U)

ny = U.shape[1]
y_mid= ny//4
U_line = U[:,y_mid]

plt.plot(U)
plt.xlabel("x (lattice units)")
plt.ylabel("Electric potential U")
plt.title("Electric potential U along x")
plt.grid()
plt.show()
#
# plt.plot(U_line)
# plt.xlabel("x (lattice units)")
# plt.ylabel("Electric potential U")
# plt.title("Electric potential U along x")
# plt.grid()
# plt.show()
# #
# # #Check to see if E is correct
# # #E = -jnp.gradient(U_line)
# #
# # #plt.plot(E)
# # #plt.xlabel("x (lattice units)")
# # #plt.ylabel("Electric potential E")
# # #plt.title("Electric potential E along x")
# # #plt.grid()
# # #plt.show()

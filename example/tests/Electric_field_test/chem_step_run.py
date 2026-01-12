import numpy as np

from wblbm.operators.force import CompositeForce
from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
from wblbm.operators.force import ElectricForce
import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


def run_chem_step_45_inc_hysteresis():
    """Test LBM wetting implementation with hysteresis enabled including a chemical step."""
    print("\n=== Testing LBM Wetting with Hysteresis ===")

    # Simulation parameters
    grid_shape = (201, 101)
    lattice_type = "D2Q9"
    tau = 0.99
    nt = 80000
    save_interval = 200
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 5

    phi_value = 1.1
    d_rho_value = 0.2

    force_g = 1e-7
    inclination_angle = 45
    gravity = GravityForceMultiphaseDroplet(
        force_g, inclination_angle, grid_shape
    )

    # Electric field config
    permittivity_liquid = 1
    permittivity_vapour = 0.01
    conductivity_liquid = 1
    conductivity_vapour = 0.001
    U_0 = 5e-2
    electric = ElectricForce(
        permittivity_liquid=permittivity_liquid,
        permittivity_vapour=permittivity_vapour,
        conductivity_liquid=conductivity_liquid,
        conductivity_vapour=conductivity_vapour,
        grid_shape=grid_shape,
        lattice_type=lattice_type,
        U_0=U_0
    )

    # Add hysteresis parameters to bc_config
    bc_config = {
        'left': 'periodic',
        'bottom': 'wetting',
        'top': 'symmetry',
        'right': 'periodic',
        'chemical_step': {
            'chemical_step_location': .5,
            'chemical_step_edge': 'bottom',
            'ca_advancing_pre_step': 110.0,
            'ca_receding_pre_step': 90.0,
            'ca_advancing_post_step': 70.0,
            'ca_receding_post_step': 60.0,
        },
        'wetting_params': {
            'rho_l': rho_l,
            'rho_v': rho_v,
            'phi_left': phi_value,
            'phi_right': phi_value,
            'd_rho_left': d_rho_value,
            'd_rho_right': d_rho_value,
            'width': interface_width
        },
        'hysteresis_params': {
            'ca_advancing': 90.0,
            'ca_receding': 80.0,
            'learning_rate': 0.05,
            'max_iterations': 10
        }
    }

    sim = Run(
        simulation_type="multiphase",
        grid_shape=grid_shape,
        lattice_type="D2Q9",
        tau=tau,
        nt=nt,
        kappa=kappa,
        rho_l=rho_l,
        rho_v=rho_v,
        interface_width=interface_width,
        save_interval=save_interval,
        bc_config=bc_config,
        force_enabled=True,
        force_obj=[gravity],
        phi_value=phi_value,
        d_rho_value=d_rho_value,
        wetting_enabled=True,
        force_g=force_g,
        inclination_angle=inclination_angle,
        permittivity_liquid=permittivity_liquid,
        permittivity_vapour=permittivity_vapour,
        conductivity_liquid=conductivity_liquid,
        conductivity_vapour=conductivity_vapour,
        U_0=U_0,
        init_type="init_from_file",
        init_dir="/Users/sbszkudlarek/TUD_LBM/results/2026-01-09/Electric_field_with_hysteresis_low_grav/16-57-29_init_for_chem_step_0_inc_wetting_only/data/timestep_199999.npz",
    )

    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    sim_wetting_hysteresis = run_chem_step_45_inc_hysteresis()

    # Visualize results
    print("\n=== Visualizing Wetting Hysteresis Test Results ===")
    visualise(sim_wetting_hysteresis, "Wetting Hysteresis Implementation Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

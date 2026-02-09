import numpy as np

from wblbm.operators.force import CompositeForce
from wblbm.run import Run
from wblbm import GravityForceMultiphaseDroplet, visualise
import jax

jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_disable_jit", True)


def init_for_bubble_top_chem_step():
    """Test LBM wetting implementation with a bubble at the top of the domain."""
    print("\n=== Testing LBM Bubble at Top with Wetting ===")

    # Simulation parameters
    grid_shape = (201, 101)
    tau = 0.99
    nt = 1000
    save_interval = nt / 10
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 5

    phi_value = 1
    d_rho_value = 0

    force_g = 1e-7
    inclination_angle = 0
    gravity = GravityForceMultiphaseDroplet(
        force_g, inclination_angle, grid_shape
    )

    # Bubble initialization parameters
    init_radius = 0.25  # Fraction of ny for bubble radius
    init_location = 0.25  # Fraction of nx for horizontal center

    # Boundary config with wetting at top for bubble
    bc_config = {
        'right': 'periodic',
        'bottom': 'bounce-back',
        'top': 'wetting',
        'left': 'periodic',
        'wetting_params': {
            'rho_l': rho_l,
            'rho_v': rho_v,
            'phi_left': phi_value,
            'phi_right': phi_value,
            'd_rho_left': d_rho_value,
            'd_rho_right': d_rho_value,
            'width': interface_width
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
        force_enabled=False,
        force_obj=[gravity],
        phi_value=phi_value,
        d_rho_value=d_rho_value,
        wetting_enabled=True,
        force_g=force_g,
        inclination_angle=inclination_angle,
        init_type="multiphase_bubble_chem_step",
        init_radius=init_radius,
        init_location=init_location,
    )

    sim.run(verbose=True)
    return sim


if __name__ == "__main__":
    sim_bubble_top = init_for_bubble_top_chem_step()

    # Visualize results
    print("\n=== Visualizing Bubble at Top Test Results ===")
    visualise(sim_bubble_top, "Bubble at Top Wetting Test")

    print("\nTest completed! Check the 'results' directory for data and plots.")

"""
Test for multiphase droplet simulation with electric field force.
Implements electric potential distribution (h_i) alongside density distribution (f_i).
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from wblbm.run import Run
from wblbm.operators.force import (
    ElectricForce,
    GravityForceMultiphase,
    CompositeForce,
    collision_h_i,
    equilibrium_h
)
from wblbm.operators.stream.stream import Streaming
from wblbm.utils.plotting import visualise

# Disable JIT for debugging (enable for production)
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def initialize_charge_density(grid_shape, droplet_radius=30, charge_value=1.0):
    """
    Initialize charge density field.
    Charge is concentrated in/around the droplet interface.

    Args:
        grid_shape: Tuple of (nx, ny)
        droplet_radius: Radius of the droplet
        charge_value: Charge density value

    Returns:
        Charge density array of shape (nx, ny)
    """
    nx, ny = grid_shape
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')

    # Center of droplet
    center_x = nx // 2
    center_y = ny // 2

    # Distance from center
    distance = jnp.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # Charge in a thin shell around the droplet (interface region)
    interface_width = 8
    charge_density = jnp.where(
        jnp.abs(distance - droplet_radius) < interface_width,
        charge_value,
        0.0
    )

    return charge_density


def initialize_electric_potential(grid_shape, boundary_value=1.0):
    """
    Initialize electric potential field.
    Set higher potential at top boundary, lower at bottom.

    Args:
        grid_shape: Tuple of (nx, ny)
        boundary_value: Potential difference magnitude

    Returns:
        Initial potential array of shape (nx, ny)
    """
    nx, ny = grid_shape
    y = jnp.arange(ny)

    # Linear potential gradient from top to bottom
    U = boundary_value * (1.0 - 2.0 * y / ny)

    return jnp.tile(U[jnp.newaxis, :], (nx, 1))


def multiphase_electric_field_simulation_test():
    """
    Test a multiphase LBM simulation with electric field force on a central droplet.
    """
    print("\n=== Multiphase LBM Simulation with Electric Field Test ===")

    # Simulation parameters
    grid_shape = (201, 201)
    nt = 5000
    save_interval = 500
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 8
    tau = 0.9

    # Electric field parameters
    permittivity_liquid = 1.0
    permittivity_vapour = 1.0
    conductivity_liquid = 1.0
    conductivity_vapour = 1.0

    tau_e = 0.8  # Relaxation time for electrical distribution
    electric_potential_boundary = 1.0

    # Initialize forces
    print("Initializing forces...")

    # Electrical force
    electrical_force = ElectricForce(
        nx=grid_shape[0],
        ny=grid_shape[1],
        d=2,
        permittivity_liquid=permittivity_liquid,
        permittivity_vapour=permittivity_vapour,
        conductivity_liquid=conductivity_liquid,
        conductivity_vapour=conductivity_vapour
    )

    # Gravity (optional - set to zero for pure electric field test)
    gravity = GravityForceMultiphase(
        nx=grid_shape[0],
        ny=grid_shape[1],
        d=2,
        force_g=0.0,  # Disable gravity to see pure electric effects
        inclination_angle_deg=0.0
    )

    # Composite force (gravity + electrical)
    combined_force = CompositeForce(gravity, electrical_force)

    # Initialize charge density and electric potential
    charge_density = initialize_charge_density(grid_shape, droplet_radius=30, charge_value=0.1)
    initial_potential = initialize_electric_potential(grid_shape, boundary_value=electric_potential_boundary)

    print(f"Charge density range: [{jnp.min(charge_density):.4e}, {jnp.max(charge_density):.4e}]")
    print(f"Initial potential range: [{jnp.min(initial_potential):.4e}, {jnp.max(initial_potential):.4e}]")

    # Initialize h_i distribution for electric potential
    from wblbm.lattice import Lattice
    lattice = Lattice()
    w_i = lattice.w
    c_i = lattice.c

    h_i = equilibrium_h(initial_potential, w_i)
    electrical_force.update_potential(h_i)

    print(f"Initial h_i shape: {h_i.shape}")
    print(f"Initial electric potential (U) shape: {electrical_force.U.shape}")

    # Create simulation
    print("\nCreating simulation...")
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
        force_enabled=True,
        force_obj=combined_force,
        init_type="multiphase_droplet",
    )

    # Store electric field data during simulation
    electric_field_data = {
        'time_steps': [],
        'h_i': [],
        'U': [],
        'electric_field_x': [],
        'electric_field_y': [],
        'charge_density': charge_density,
    }

    # Main simulation loop with electric field evolution
    print("\nStarting simulation with electric field evolution...")
    fprev = sim.simulation.initialize_fields("multiphase_droplet")

    for it in range(nt):
        # Evolve electrical distribution (h_i)
        electrical_force.U = jnp.sum(h_i, axis=2)
        electrical_force.update_potential(h_i)

        # Collision step for h_i
        h_i = collision_h_i(h_i, electrical_force.U, tau_e, w_i)

        # Streaming step for h_i
        h_i = stream_h_i_vectorized(h_i, c_i)

        # Update electrical force with current potential
        electrical_force.update_potential(h_i)

        # Run LBM timestep with forces
        fprev = sim.simulation.run_timestep(fprev, it)

        # Check for NaN
        if jnp.isnan(fprev).any():
            print(f"NaN encountered at timestep {it}. Stopping simulation.")
            break

        # Save electric field data periodically
        if it % save_interval == 0 or it == nt - 1:
            electric_field_data['time_steps'].append(it)
            electric_field_data['h_i'].append(np.array(h_i))
            electric_field_data['U'].append(np.array(electrical_force.U))

            # Compute electric field
            E_x = -jnp.gradient(electrical_force.U, axis=0)
            E_y = -jnp.gradient(electrical_force.U, axis=1)
            electric_field_data['electric_field_x'].append(np.array(E_x))
            electric_field_data['electric_field_y'].append(np.array(E_y))

            # Print diagnostics
            if it % save_interval == 0:
                print(f"Step {it}/{nt}: U_range=[{jnp.min(electrical_force.U):.4e}, "
                      f"{jnp.max(electrical_force.U):.4e}], "
                      f"h_i_sum_range=[{jnp.min(jnp.sum(h_i, axis=2)):.4e}, "
                      f"{jnp.max(jnp.sum(h_i, axis=2)):.4e}]")

        # Save data at intervals
        if (it > sim.skip_interval) and (it % save_interval == 0 or it == nt - 1):
            sim._save_data(it, fprev)

    print("\nSimulation completed!")
    print(f"Final potential range: [{jnp.min(electrical_force.U):.4e}, {jnp.max(electrical_force.U):.4e}]")

    return sim, electric_field_data, h_i, electrical_force


def test_electric_field_only():
    """
    Test pure electric field evolution without LBM dynamics.
    Useful for validating the electrical distribution function.
    """
    print("\n=== Pure Electric Field Evolution Test ===")

    grid_shape = (201, 201)
    nt = 1000
    permittivity = 1.0
    tau_e = 0.8

    from wblbm.lattice import Lattice
    lattice = Lattice("D2Q9")
    w_i = lattice.w
    c_i = lattice.c

    # Initialize potential and h_i
    initial_potential = initialize_electric_potential(grid_shape, boundary_value=1.0)
    h_i = equilibrium_h(initial_potential, w_i)

    print(f"Grid shape: {grid_shape}")
    print(f"Initial potential range: [{jnp.min(initial_potential):.4e}, {jnp.max(initial_potential):.4e}]")

    # Evolve electric field
    potential_history = [initial_potential]

    for it in range(nt):
        # Collision
        U = jnp.sum(h_i, axis=2)
        h_i = collision_h_i(h_i, U, tau_e, w_i)

        # Streaming
        h_i = stream_h_i_vectorized(h_i, c_i)

        if it % 100 == 0:
            U_final = jnp.sum(h_i, axis=2)
            potential_history.append(U_final)
            print(f"Step {it}: U_range=[{jnp.min(U_final):.4e}, {jnp.max(U_final):.4e}]")

    U_final = jnp.sum(h_i, axis=2)
    print(f"Final potential range: [{jnp.min(U_final):.4e}, {jnp.max(U_final):.4e}]")

    return h_i, potential_history


def test_composite_force_with_gravity_and_electric():
    """
    Test composite force combining both gravitational and electrical forces.
    """
    print("\n=== Multiphase LBM Simulation with Gravity + Electric Field ===")

    grid_shape = (201, 201)
    nt = 3000
    save_interval = 300
    kappa = 0.04
    rho_l = 1.0
    rho_v = 0.001
    interface_width = 8
    tau = 0.9

    # Force parameters
    force_g = 0.00001
    permittivity = 1.0
    tau_e = 0.8

    print("Initializing composite forces (gravity + electric)...")

    # Gravity force
    gravity = GravityForceMultiphase(
        nx=grid_shape[0],
        ny=grid_shape[1],
        d=2,
        force_g=force_g,
        inclination_angle_deg=0.0
    )

    # Electrical force
    electrical_force = ElectricForce(
        nx=grid_shape[0],
        ny=grid_shape[1],
        d=2,
        permittivity=permittivity
    )

    # Composite force
    combined_force = CompositeForce(gravity, electrical_force)

    # Initialize charge density and potential
    charge_density = initialize_charge_density(grid_shape, droplet_radius=30, charge_value=0.05)
    initial_potential = initialize_electric_potential(grid_shape, boundary_value=0.5)

    from wblbm.lattice import Lattice
    lattice = Lattice()
    w_i = lattice.w
    c_i = lattice.c

    h_i = equilibrium_h(initial_potential, w_i)
    electrical_force.update_potential(h_i)

    # Create simulation
    print("Creating simulation with composite forces...")
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
        force_enabled=True,
        force_obj=combined_force,
        init_type="multiphase_droplet",
    )

    # Run simulation
    print("\nStarting simulation...")
    fprev = sim.simulation.initialize_fields("multiphase_droplet")

    for it in range(nt):
        # Update electrical distribution
        h_i = collision_h_i(h_i, electrical_force.U, tau_e, w_i, 1.0/3.0)
        h_i = stream_h_i_vectorized(h_i, c_i)
        electrical_force.update_potential(h_i)

        # Run LBM timestep
        fprev = sim.simulation.run_timestep(fprev, it)

        if jnp.isnan(fprev).any():
            print(f"NaN encountered at timestep {it}. Stopping simulation.")
            break

        if (it > sim.skip_interval) and (it % save_interval == 0 or it == nt - 1):
            sim._save_data(it, fprev)
            if it % save_interval == 0:
                print(f"Step {it}/{nt}: Simulation progressing...")

    print("Simulation completed!")
    return sim


if __name__ == "__main__":
    print("=" * 70)
    print("MULTIPHASE DROPLET WITH ELECTRIC FIELD TEST")
    print("=" * 70)

    # Test 1: Pure electric field evolution
    print("\n" + "=" * 70)
    print("TEST 1: Pure Electric Field Evolution")
    print("=" * 70)
    h_i_final, potential_history = test_electric_field_only()

    # Test 2: Multiphase droplet with electric field
    print("\n" + "=" * 70)
    print("TEST 2: Multiphase Droplet with Electric Field")
    print("=" * 70)
    sim_electric, electric_field_data, h_i, electrical_force = multiphase_electric_field_simulation_test()

    # Visualize results
    print("\n=== Visualizing Results (Electric Field Only) ===")
    try:
        visualise(sim_electric, "Multiphase Droplet with Electric Field")
    except Exception as e:
        print(f"Visualization error (non-critical): {e}")

    # Test 3: Combined gravity + electric field
    print("\n" + "=" * 70)
    print("TEST 3: Multiphase Droplet with Gravity + Electric Field")
    print("=" * 70)
    sim_combined = test_composite_force_with_gravity_and_electric()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


"""
Surface tension calculation script using the Young-Laplace equation.
For 2D circular droplets: ΔP = σ/R

This script runs multiple simulations with different droplet radii,
measures the pressure difference between inside and outside of the droplet,
and fits the Young-Laplace equation to extract the surface tension.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from wblbm import Run


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_single_simulation(config: dict, radius: float, num_iterations: int) -> Run:
    """
    Run a single simulation with a droplet of specified radius.

    Args:
        config: Base configuration dictionary
        radius: Droplet radius in lattice units
        num_iterations: Number of iterations to run

    Returns:
        Run: Completed simulation object
    """
    # Create simulation with periodic boundaries (isolated droplet)
    bc_config = {
        "top": "periodic",
        "bottom": "periodic",
        "left": "periodic",
        "right": "periodic",
    }

    # Get collision config if present, otherwise use BGK
    collision_config = config.get('collision', {"collision_scheme": "bgk"})
    if isinstance(collision_config, str):
        collision_config = {"collision_scheme": collision_config}

    sim = Run(
        simulation_type="multiphase",
        grid_shape=tuple(config['grid_shape']),
        lattice_type=config.get('lattice_type', 'D2Q9'),
        nt=num_iterations,
        kappa=config['kappa'],
        rho_l=config['rho_l'],
        rho_v=config['rho_v'],
        interface_width=config['interface_width'],
        save_interval=num_iterations + 1,  # Don't save intermediate steps
        skip_interval=0,
        init_type="multiphase_droplet_variable_radius",
        init_radius=radius,  # Pass radius for variable initialization
        tau=config.get('tau', 1.0),
        collision=collision_config,
        bc_config=bc_config,
        simulation_name=f"surface_tension_R{radius:.1f}",
    )

    # Run the simulation
    sim.run(verbose=False)

    return sim


def calculate_pressure_difference(sim: Run, interface_width: int) -> float:
    """
    Calculate pressure difference between inside and outside of droplet.

    Args:
        sim: Completed simulation object
        interface_width: Interface width for sampling offset

    Returns:
        float: Pressure difference (P_inside - P_outside)
    """
    # Get the final distribution function
    f = sim.simulation.f if hasattr(sim.simulation, 'f') else None

    if f is None:
        raise ValueError("Could not retrieve distribution function from simulation")

    # Get macroscopic variables
    macroscopic = sim.simulation.macroscopic
    rho, _, _ = macroscopic(f, force=None)

    # Calculate pressure field
    pressure = macroscopic.pressure(rho)

    # Extract 2D arrays
    p_2d = np.array(pressure[:, :, 0, 0])
    nx, ny = p_2d.shape

    # Inside: center of domain (well inside droplet)
    center_x, center_y = nx // 2, ny // 2
    p_inside = p_2d[center_x, center_y]

    # Outside: sample at corners (vapor region, away from interface)
    margin = 3 * interface_width
    corners = [
        (margin, margin),
        (margin, ny - margin - 1),
        (nx - margin - 1, margin),
        (nx - margin - 1, ny - margin - 1)
    ]

    p_outside_values = [p_2d[x, y] for x, y in corners]
    p_outside = np.mean(p_outside_values)

    delta_p = float(p_inside - p_outside)

    print(f"    P_inside = {p_inside:.6e}, P_outside = {p_outside:.6e}, ΔP = {delta_p:.6e}")

    return delta_p


def fit_surface_tension(radii: np.ndarray, delta_pressures: np.ndarray) -> tuple:
    """
    Fit surface tension from pressure differences using Young-Laplace equation.
    For 2D: ΔP = σ/R

    Args:
        radii: Array of droplet radii
        delta_pressures: Array of pressure differences

    Returns:
        tuple: (surface_tension, r_squared, intercept)
    """
    inverse_radii = 1.0 / radii

    # Linear fit: ΔP = σ * (1/R) + intercept
    coeffs = np.polyfit(inverse_radii, delta_pressures, deg=1)
    surface_tension = coeffs[0]  # Slope is surface tension
    intercept = coeffs[1]

    # Calculate R² for fit quality
    predicted = np.polyval(coeffs, inverse_radii)
    ss_res = np.sum((delta_pressures - predicted) ** 2)
    ss_tot = np.sum((delta_pressures - np.mean(delta_pressures)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return surface_tension, r_squared, intercept


def plot_results(radii: np.ndarray, delta_pressures: np.ndarray,
                 surface_tension: float, r_squared: float,
                 output_path: str):
    """Plot and save the results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: ΔP vs 1/R
    inverse_radii = 1.0 / radii
    ax1.scatter(inverse_radii, delta_pressures, s=100, c='blue', label='Simulation data')

    # Fit line
    x_fit = np.linspace(min(inverse_radii) * 0.9, max(inverse_radii) * 1.1, 100)
    y_fit = surface_tension * x_fit
    ax1.plot(x_fit, y_fit, 'r--', linewidth=2,
             label=f'Linear fit: σ = {surface_tension:.6f}')

    ax1.set_xlabel('1/R (1/lattice units)', fontsize=12)
    ax1.set_ylabel('ΔP (lattice units)', fontsize=12)
    ax1.set_title(f'Young-Laplace Relation (R² = {r_squared:.4f})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ΔP vs R
    ax2.scatter(radii, delta_pressures, s=100, c='blue', label='Simulation data')
    r_fit = np.linspace(min(radii) * 0.9, max(radii) * 1.1, 100)
    dp_fit = surface_tension / r_fit
    ax2.plot(r_fit, dp_fit, 'r--', linewidth=2, label=f'ΔP = σ/R')

    ax2.set_xlabel('R (lattice units)', fontsize=12)
    ax2.set_ylabel('ΔP (lattice units)', fontsize=12)
    ax2.set_title('Pressure Difference vs Radius', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to: {output_path}")


def main():
    """Main function to run surface tension calculation."""
    print("=" * 60)
    print("Surface Tension Calculation using Young-Laplace Equation")
    print("=" * 60)

    # Prompt user for config path
    config_path_str = input("\nPlease enter the path to the config.json file: ").strip()
    config_path = Path(config_path_str)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))

    nx, ny = config['grid_shape']
    min_dim = min(nx, ny)

    # Define radii range: from min_dim/5 to min_dim/2
    num_simulations = 5
    num_iterations = 200000
    r_min = min_dim / 6
    r_max = min_dim / 3
    radii = np.linspace(r_min, r_max, num_simulations)

    print(f"\nGrid size: {nx} x {ny}")
    print(f"Radii to test: {radii}")
    print(f"Number of iterations per simulation: {num_iterations}")
    print(f"κ = {config['kappa']}, ρ_l = {config['rho_l']}, ρ_v = {config['rho_v']}")
    print()

    # Run simulations and collect results
    delta_pressures = []

    for i, radius in enumerate(radii):
        print(f"\n[{i + 1}/{num_simulations}] Simulating droplet with R = {radius:.2f}")

        sim = run_single_simulation(
            config=config,
            radius=radius,
            num_iterations=num_iterations
        )

        delta_p = calculate_pressure_difference(
            sim=sim,
            interface_width=config['interface_width']
        )
        delta_pressures.append(delta_p)

    # Convert to numpy arrays
    radii = np.array(radii)
    delta_pressures = np.array(delta_pressures)

    # Fit surface tension
    surface_tension, r_squared, intercept = fit_surface_tension(radii, delta_pressures)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nSurface Tension (σ): {surface_tension:.6f}")
    print(f"Fit Quality (R²): {r_squared:.6f}")
    print(f"Fit Intercept: {intercept:.6e}")
    print("\nData points:")
    print("-" * 40)
    print(f"{'Radius':<15} {'1/R':<15} {'ΔP':<15}")
    print("-" * 40)
    for r, dp in zip(radii, delta_pressures):
        print(f"{r:<15.4f} {1 / r:<15.6f} {dp:<15.6e}")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Save numerical data
    np.savez(
        output_dir / 'surface_tension_results.npz',
        radii=radii,
        delta_pressures=delta_pressures,
        surface_tension=surface_tension,
        r_squared=r_squared
    )

    # Plot results
    plot_results(
        radii=radii,
        delta_pressures=delta_pressures,
        surface_tension=surface_tension,
        r_squared=r_squared,
        output_path=str(output_dir / 'surface_tension_plot.png')
    )

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


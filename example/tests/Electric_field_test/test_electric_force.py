import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from wblbm.operators.force import ElectricForce


def test_electric_force_with_plots():
    """Test electric force calculation for leaky dielectric with detailed plots."""

    # Setup
    nx, ny = 128, 128
    grid_shape = (nx, ny)

    # Create electric force operator
    elec_force = ElectricForce(
        permittivity_liquid=10.0,
        permittivity_vapour=1.0,
        conductivity_liquid=0.1,
        conductivity_vapour=0.01,
        grid_shape=grid_shape,
        lattice_type='D2Q9',
        bc_config=None
    )

    # Create test density field (circular droplet)
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Circular droplet at center
    r = jnp.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    rho = jnp.where(r < 0.2, 1.0, 0.1)  # High density inside, low outside
    rho = rho[:, :, jnp.newaxis, jnp.newaxis]

    # Create potential field (uniform E-field in x-direction)
    u_pot = X[:, :, jnp.newaxis, jnp.newaxis]

    # Convert to h_i distribution
    h_i = elec_force.equilibrium_h(
        jnp.tile(u_pot, (1, 1, 9, 1)),
        elec_force.lattice.w
    )

    # Extract electric field components for plotting
    e_x = -jnp.gradient(u_pot[:, :, 0, 0], axis=0)
    e_y = -jnp.gradient(u_pot[:, :, 0, 0], axis=1)
    e_x = e_x[:, :, jnp.newaxis, jnp.newaxis]
    e_y = e_y[:, :, jnp.newaxis, jnp.newaxis]
    e_sq = e_x**2 + e_y**2

    # Get permittivity field
    epsilon = elec_force.permittivity(
        rho,
        permittivity_liquid=elec_force.permittivity_liquid,
        permittivity_vapour=elec_force.permittivity_vapour
    )

    # Compute charge density contribution (rho_e = div(epsilon * E))
    epsilon_e_x = epsilon * e_x
    epsilon_e_y = epsilon * e_y
    d_epsilon_e_x_dx = jnp.gradient(epsilon_e_x[:, :, 0, 0], axis=0)
    d_epsilon_e_y_dy = jnp.gradient(epsilon_e_y[:, :, 0, 0], axis=1)
    rho_e = (d_epsilon_e_x_dx + d_epsilon_e_y_dy)[:, :, jnp.newaxis, jnp.newaxis]

    # Compute gradient of permittivity
    grad_epsilon_x = jnp.gradient(epsilon[:, :, 0, 0], axis=0)[:, :, jnp.newaxis, jnp.newaxis]
    grad_epsilon_y = jnp.gradient(epsilon[:, :, 0, 0], axis=1)[:, :, jnp.newaxis, jnp.newaxis]

    # Compute force contributions separately
    f_x_coulomb = rho_e * e_x
    f_y_coulomb = rho_e * e_y
    f_x_dielectric = -0.5 * e_sq * grad_epsilon_x
    f_y_dielectric = -0.5 * e_sq * grad_epsilon_y

    # Total force
    f_x = f_x_coulomb + f_x_dielectric
    f_y = f_y_coulomb + f_y_dielectric

    # Compute total force using the method
    force = elec_force.compute_force(rho=rho, h_i=h_i)

    # Validate
    assert force.shape == (nx, ny, 1, 2), f"Wrong shape: {force.shape}"
    assert not jnp.isnan(force).any(), "Force contains NaN"

    # Compute magnitudes for visualization
    f_coulomb_mag = jnp.sqrt(f_x_coulomb**2 + f_y_coulomb**2)
    f_dielectric_mag = jnp.sqrt(f_x_dielectric**2 + f_y_dielectric**2)
    f_total_mag = jnp.sqrt(f_x**2 + f_y**2)
    epsilon_2d = epsilon[:, :, 0, 0]
    rho_2d = rho[:, :, 0, 0]

    # Convert to numpy for plotting
    f_coulomb_mag = np.array(f_coulomb_mag[:, :, 0, 0])
    f_dielectric_mag = np.array(f_dielectric_mag[:, :, 0, 0])
    f_total_mag = np.array(f_total_mag[:, :, 0, 0])
    f_x_coulomb_plot = np.array(f_x_coulomb[:, :, 0, 0])
    f_y_coulomb_plot = np.array(f_y_coulomb[:, :, 0, 0])
    f_x_dielectric_plot = np.array(f_x_dielectric[:, :, 0, 0])
    f_y_dielectric_plot = np.array(f_y_dielectric[:, :, 0, 0])
    f_x_plot = np.array(f_x[:, :, 0, 0])
    f_y_plot = np.array(f_y[:, :, 0, 0])
    epsilon_plot = np.array(epsilon_2d)
    rho_plot = np.array(rho_2d)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))

    # Row 1: Density and Permittivity
    ax1 = plt.subplot(4, 4, 1)
    im1 = ax1.imshow(rho_plot.T, origin='lower', cmap='viridis')
    ax1.set_title('Density Field', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='ρ')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2 = plt.subplot(4, 4, 2)
    im2 = ax2.imshow(epsilon_plot.T, origin='lower', cmap='plasma')
    ax2.set_title('Permittivity Field ε(ρ)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='ε')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    ax3 = plt.subplot(4, 4, 3)
    e_x_plot = np.array(e_x[:, :, 0, 0])
    e_y_plot = np.array(e_y[:, :, 0, 0])
    e_mag = np.sqrt(e_x_plot**2 + e_y_plot**2)
    im3 = ax3.imshow(e_mag.T, origin='lower', cmap='coolwarm')
    ax3.set_title('Electric Field Magnitude |E|', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='|E|')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    ax4 = plt.subplot(4, 4, 4)
    rho_e_plot = np.array(rho_e[:, :, 0, 0])
    im4 = ax4.imshow(rho_e_plot.T, origin='lower', cmap='RdBu_r')
    ax4.set_title('Charge Density ρ_e = div(εE)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='ρ_e')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    # Row 2: Coulombic force magnitude and vector field
    ax5 = plt.subplot(4, 4, 5)
    im5 = ax5.imshow(f_coulomb_mag.T, origin='lower', cmap='hot')
    ax5.set_title('Coulombic Force Magnitude |ρ_e*E|', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='|f_coulomb|')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')

    ax6 = plt.subplot(4, 4, 6)
    im6 = ax6.imshow(f_coulomb_mag.T, origin='lower', cmap='hot')
    skip = max(nx, ny) // 16
    X_skip = np.arange(0, nx, skip)
    Y_skip = np.arange(0, ny, skip)
    X_mesh, Y_mesh = np.meshgrid(X_skip, Y_skip, indexing='ij')
    ax6.quiver(X_mesh, Y_mesh,
               f_x_coulomb_plot[::skip, ::skip].T,
               f_y_coulomb_plot[::skip, ::skip].T,
               color='white', scale=None, scale_units='xy', angles='xy', width=0.003)
    ax6.set_title('Coulombic Force Vector Field', fontsize=12, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')

    ax7 = plt.subplot(4, 4, 7)
    grad_eps_mag = np.sqrt(np.array(grad_epsilon_x[:, :, 0, 0])**2 +
                           np.array(grad_epsilon_y[:, :, 0, 0])**2)
    im7 = ax7.imshow(grad_eps_mag.T, origin='lower', cmap='YlOrRd')
    ax7.set_title('Permittivity Gradient |∇ε|', fontsize=12, fontweight='bold')
    plt.colorbar(im7, ax=ax7, label='|∇ε|')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')

    ax8 = plt.subplot(4, 4, 8)
    E_sq_plot = np.array(e_sq[:, :, 0, 0])
    im8 = ax8.imshow(E_sq_plot.T, origin='lower', cmap='Spectral_r')
    ax8.set_title('Electric Field Squared |E|²', fontsize=12, fontweight='bold')
    plt.colorbar(im8, ax=ax8, label='|E|²')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')

    # Row 3: Dielectric force magnitude and vector field
    ax9 = plt.subplot(4, 4, 9)
    im9 = ax9.imshow(f_dielectric_mag.T, origin='lower', cmap='hot')
    ax9.set_title('Dielectric Force Magnitude |-0.5*|E|²*∇ε|', fontsize=12, fontweight='bold')
    plt.colorbar(im9, ax=ax9, label='|f_dielectric|')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')

    ax10 = plt.subplot(4, 4, 10)
    im10 = ax10.imshow(f_dielectric_mag.T, origin='lower', cmap='hot')
    ax10.quiver(X_mesh, Y_mesh,
                f_x_dielectric_plot[::skip, ::skip].T,
                f_y_dielectric_plot[::skip, ::skip].T,
                color='white', scale=None, scale_units='xy', angles='xy', width=0.003)
    ax10.set_title('Dielectric Force Vector Field', fontsize=12, fontweight='bold')
    ax10.set_xlabel('x')
    ax10.set_ylabel('y')

    # Force component comparison
    ax11 = plt.subplot(4, 4, 11)
    coulomb_max = np.max(f_coulomb_mag)
    dielectric_max = np.max(f_dielectric_mag)
    x_pos = np.arange(2)
    heights = [coulomb_max, dielectric_max]
    bars = ax11.bar(x_pos, heights, color=['#FF6B6B', '#4ECDC4'])
    ax11.set_ylabel('Maximum Force Magnitude', fontsize=11, fontweight='bold')
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(['Coulombic\n(ρ_e*E)', 'Dielectric\n(-0.5*|E|²*∇ε)'])
    ax11.set_title('Force Contribution Comparison', fontsize=12, fontweight='bold')
    for i, (bar, h) in enumerate(zip(bars, heights)):
        ax11.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2e}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Spatial distribution comparison along midline
    ax12 = plt.subplot(4, 4, 12)
    mid_idx = ny // 2
    ax12.plot(f_coulomb_mag[:, mid_idx], label='Coulombic Force', linewidth=2, color='#FF6B6B')
    ax12.plot(f_dielectric_mag[:, mid_idx], label='Dielectric Force', linewidth=2, color='#4ECDC4')
    ax12.plot(f_total_mag[:, mid_idx], label='Total Force', linewidth=2.5, color='black', linestyle='--')
    ax12.set_xlabel('x position', fontsize=11)
    ax12.set_ylabel('Force Magnitude', fontsize=11)
    ax12.set_title(f'Force Profiles (y = {mid_idx})', fontsize=12, fontweight='bold')
    ax12.legend(loc='best')
    ax12.grid(True, alpha=0.3)

    # Row 4: Total force magnitude and vector field
    ax13 = plt.subplot(4, 4, 13)
    im13 = ax13.imshow(f_total_mag.T, origin='lower', cmap='hot')
    ax13.set_title('Total Force Magnitude |F|', fontsize=12, fontweight='bold')
    plt.colorbar(im13, ax=ax13, label='|F|')
    ax13.set_xlabel('x')
    ax13.set_ylabel('y')

    ax14 = plt.subplot(4, 4, 14)
    im14 = ax14.imshow(f_total_mag.T, origin='lower', cmap='hot')
    ax14.quiver(X_mesh, Y_mesh,
                f_x_plot[::skip, ::skip].T,
                f_y_plot[::skip, ::skip].T,
                color='white', scale=None, scale_units='xy', angles='xy', width=0.003)
    ax14.set_title('Total Force Vector Field', fontsize=12, fontweight='bold')
    ax14.set_xlabel('x')
    ax14.set_ylabel('y')

    # Force contribution ratio
    ax15 = plt.subplot(4, 4, 15)
    ratio = np.divide(f_coulomb_mag, f_dielectric_mag + 1e-16,
                     where=(f_dielectric_mag + 1e-16) != 0,
                     out=np.zeros_like(f_dielectric_mag))
    ratio = np.clip(ratio, -10, 10)  # Clip extreme values for visualization
    im15 = ax15.imshow(ratio.T, origin='lower', cmap='seismic', vmin=-5, vmax=5)
    ax15.set_title('Force Contribution Ratio\n(Coulombic / Dielectric)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im15, ax=ax15, label='Ratio')
    ax15.set_xlabel('x')
    ax15.set_ylabel('y')

    # Statistics panel
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis('off')
    stats_text = (
        f'Force Statistics:\n\n'
        f'Coulombic Force:\n'
        f'  Max: {coulomb_max:.3e}\n'
        f'  Mean: {np.mean(f_coulomb_mag):.3e}\n'
        f'  Median: {np.median(f_coulomb_mag):.3e}\n\n'
        f'Dielectric Force:\n'
        f'  Max: {dielectric_max:.3e}\n'
        f'  Mean: {np.mean(f_dielectric_mag):.3e}\n'
        f'  Median: {np.median(f_dielectric_mag):.3e}\n\n'
        f'Total Force:\n'
        f'  Max: {np.max(f_total_mag):.3e}\n'
        f'  Mean: {np.mean(f_total_mag):.3e}\n'
        f'  Grid: {nx} × {ny}\n\n'
        f'Physical Properties:\n'
        f'  ε_liquid: {elec_force.permittivity_liquid}\n'
        f'  ε_vapour: {elec_force.permittivity_vapour}\n'
        f'  σ_liquid: {elec_force.conductivity_liquid}\n'
        f'  σ_vapour: {elec_force.conductivity_vapour}'
    )
    ax16.text(0.05, 0.95, stats_text, transform=ax16.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Leaky Dielectric Electric Force Analysis\n' +
                 'F = ρ_e*E - 0.5*|E|²*∇ε',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    # Print summary
    print("✓ Force Computation Summary")
    print("=" * 60)
    print(f"Grid Size: {nx} × {ny}")
    print(f"\nCoulombic Force Contribution (ρ_e * E):")
    print(f"  Max Magnitude: {coulomb_max:.3e}")
    print(f"  Mean Magnitude: {np.mean(f_coulomb_mag):.3e}")
    print(f"  Std Dev: {np.std(f_coulomb_mag):.3e}")
    print(f"\nDielectric Force Contribution (-0.5 * |E|² * ∇ε):")
    print(f"  Max Magnitude: {dielectric_max:.3e}")
    print(f"  Mean Magnitude: {np.mean(f_dielectric_mag):.3e}")
    print(f"  Std Dev: {np.std(f_dielectric_mag):.3e}")
    print(f"\nTotal Force (F = Coulombic + Dielectric):")
    print(f"  Max Magnitude: {np.max(f_total_mag):.3e}")
    print(f"  Mean Magnitude: {np.mean(f_total_mag):.3e}")
    print(f"  Std Dev: {np.std(f_total_mag):.3e}")
    print(f"\nDominant Contribution:")
    dominant_ratio = coulomb_max / (dielectric_max + 1e-16)
    if dominant_ratio > 1:
        print(f"  Coulombic Force is dominant (ratio: {dominant_ratio:.2f}x)")
    else:
        print(f"  Dielectric Force is dominant (ratio: {1/dominant_ratio:.2f}x)")
    print(f"\nValidation:")
    print(f"  ✓ Shape: {force.shape}")
    print(f"  ✓ No NaN values: {not jnp.isnan(force).any()}")

    # Force should be concentrated at interface (where ∇ε is large)
    force_magnitude = jnp.sqrt(force[:, :, 0, 0] ** 2 + force[:, :, 0, 1] ** 2)
    center_force = force_magnitude[nx // 2, ny // 2]
    max_force = jnp.max(force_magnitude)

    print(f"\nInterface Analysis:")
    print(f"  Center force: {center_force:.6f}")
    print(f"  Max force: {max_force:.6f}")
    print(f"  Force x-component mean: {jnp.mean(force[:, :, 0, 0]):.6f}")
    print(f"  Force y-component mean: {jnp.mean(force[:, :, 0, 1]):.6f}")

    assert max_force > center_force, "Force should peak at interface, not center"
    print(f"\n  ✓ Force correctly peaks at interface!")
    print("=" * 60)

    return fig


if __name__ == "__main__":
    fig = test_electric_force_with_plots()
    # Use explicit backend
    plt.switch_backend('Qt5Agg')  # Try TkAgg (most compatible)

    # Use block=True to ensure window stays open
    plt.show(block=True)






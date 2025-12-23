import jax.numpy as jnp
from wblbm.operators.force import ElectricForce


def test_electric_force():
    """Test electric force calculation for leaky dielectric."""

    # Setup
    nx, ny = 64, 64
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

    # Create test density field (droplet in center)
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Circular droplet
    r = jnp.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    rho = jnp.where(r < 0.2, 1.0, 0.1)  # High density inside, low outside
    rho = rho[:, :, jnp.newaxis, jnp.newaxis]

    # Create simple potential field (uniform E-field in x-direction)
    potential = X[:, :, jnp.newaxis, jnp.newaxis]

    # Convert to hi distribution
    h_i = elec_force.equilibrium_h(
        jnp.tile(potential, (1, 1, 9, 1)),
        elec_force.lattice.w
    )

    # Compute force
    force = elec_force.compute_force(rho=rho, h_i=h_i)

    # Tests
    assert force.shape == (nx, ny, 1, 2), f"Wrong shape: {force.shape}"
    assert not jnp.isnan(force).any(), "Force contains NaN"

    # Force should be concentrated at interface (where ∇ε is large)
    # Check that force magnitude peaks near droplet boundary
    force_magnitude = jnp.sqrt(force[:, :, 0, 0] ** 2 + force[:, :, 0, 1] ** 2)

    # The maximum force should be at the interface, not at center
    center_force = force_magnitude[nx // 2, ny // 2]
    max_force = jnp.max(force_magnitude)

    print(f"Center force: {center_force:.6f}")
    print(f"Max force: {max_force:.6f}")
    print(f"Force shape: {force.shape}")
    print(f"Force x-component mean: {jnp.mean(force[:, :, 0, 0]):.6f}")
    print(f"Force y-component mean: {jnp.mean(force[:, :, 0, 1]):.6f}")

    assert max_force > center_force, "Force should peak at interface, not center"
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_electric_force()

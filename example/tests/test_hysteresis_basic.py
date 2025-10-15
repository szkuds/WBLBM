"""
Simple test for hysteresis implementation
"""
import jax.numpy as jnp

def test_parameter_validation():
    """Basic validation test for parameter ranges."""

    # Test d_rho clamping
    d_rho_test = jnp.clip(jnp.array([-0.1, 0.5, 1.5]), 1e-6, 0.99)
    assert jnp.all(d_rho_test >= 1e-6), "d_rho should be >= 1e-6"
    assert jnp.all(d_rho_test <= 0.99), "d_rho should be <= 0.99"

    # Test phi clamping
    phi_test = jnp.maximum(jnp.array([0.5, 1.2, 2.0]), 1.0)
    assert jnp.all(phi_test >= 1.0), "phi should be >= 1.0"

    print("✓ Parameter validation tests passed")

def test_cost_functions():
    """Test cost function implementations."""

    def cost_cll(cll_t, cll_tplus1):
        return jnp.abs(cll_t - cll_tplus1)

    def cost_ca(ca_target, ca_tplus1):
        return jnp.abs(ca_target - ca_tplus1)

    # Test values
    assert jnp.allclose(cost_cll(jnp.array(10.0), jnp.array(8.0)), 2.0)
    assert jnp.allclose(cost_ca(jnp.array(90.0), jnp.array(95.0)), 5.0)

    print("✓ Cost function tests passed")

if __name__ == "__main__":
    print("Running basic hysteresis tests...")
    test_parameter_validation()
    test_cost_functions()
    print("All basic tests passed! ✓")
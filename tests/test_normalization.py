from jax import numpy as jnp

from dynax import normalization_coefficients


def test_normalization_coefficients():

    assert_tol = 1e-6

    # Test scalar input
    alpha, tau = normalization_coefficients(1, None, 1, verbosity=3)
    assert jnp.isclose(alpha, 1, rtol=assert_tol)
    assert jnp.isclose(tau, 1, rtol=assert_tol)

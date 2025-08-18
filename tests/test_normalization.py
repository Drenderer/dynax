import numpy as np
import pytest
from jax import numpy as jnp

from dynax import normalization_coefficients


class TestNormalizationCoefficients:
    assert_tol = 1e-3

    @pytest.mark.parametrize(
        "std_v, std_a",
        [
            (1, 1),
            (1, None),
            (None, 1),
            (None, None),
        ],
    )
    def test_already_normalized(self, std_v, std_a):
        alpha, tau = normalization_coefficients(1, std_v, std_a)
        assert jnp.isclose(alpha, 1, rtol=self.assert_tol)
        assert jnp.isclose(tau, 1, rtol=self.assert_tol)

    def test_with_analytic_solution(self):
        alpha, tau = normalization_coefficients(2, 4, 8)
        assert jnp.isclose(alpha, 0.5, rtol=self.assert_tol)
        assert jnp.isclose(tau, 0.5, rtol=self.assert_tol)

        alpha, tau = normalization_coefficients(
            jnp.array([1, 2]), jnp.array([2, 4]), jnp.array([4, 8])
        )
        assert jnp.allclose(alpha, jnp.array([1, 0.5]), rtol=self.assert_tol)
        assert jnp.isclose(tau, 0.5, rtol=self.assert_tol)

    def test_with_all_three_inputs(self):
        std_y = jnp.array([1.0, 2.0])
        std_v = jnp.array([0.5, 1.0])
        std_a = jnp.array([0.25, 0.5])
        alpha, tau = normalization_coefficients(std_y, std_v, std_a)
        assert alpha.shape == std_y.shape
        assert isinstance(tau.item(), float)
        assert tau > 0

    def test_weights_zero_for_v_and_a(self):
        """If w_v and w_a are zero, reduces to simple 1/std_y, tau=1."""
        std_y = jnp.array([5.0])
        alpha, tau = normalization_coefficients(
            std_y,
            std_v=jnp.array([10.0]),
            std_a=jnp.array([15.0]),
            w_v=0.0,
            w_a=0.0,
        )
        np.testing.assert_allclose(alpha, [1 / 5.0], rtol=self.assert_tol)
        np.testing.assert_allclose(tau, 1.0, rtol=self.assert_tol)

    def test_zero_std_in_y(self):
        """Zeros in std_y should not cause division by zero."""
        std_y = jnp.array([0.0, 2.0])
        alpha, tau = normalization_coefficients(std_y)
        assert jnp.isfinite(alpha).all()
        assert jnp.isfinite(tau)

    def test_large_std_values(self):
        """Check stability with very large std values."""
        std_y = jnp.array([1e6, 1e7])
        std_v = jnp.array([1e5, 1e6])
        std_a = jnp.array([1e4, 1e5])
        alpha, tau = normalization_coefficients(std_y, std_v, std_a)
        assert alpha.shape == std_y.shape
        assert tau > 0

    def test_negative_weights_raises(self):
        std_y = jnp.array([1.0])
        with pytest.raises(AssertionError):
            normalization_coefficients(std_y, w_y=-1.0)

    def test_convergence_parameters(self):
        """Verify it respects tolerance and maxiter without crashing."""
        std_y = jnp.array([1.0, 2.0])
        std_v = jnp.array([0.5, 1.0])
        std_a = jnp.array([0.25, 0.5])
        with pytest.warns(UserWarning):  # Warning for not converging.
            alpha, tau = normalization_coefficients(
                std_y, std_v, std_a, tol=1e-8, maxiter=5
            )
        assert alpha.shape == std_y.shape
        assert tau > 0

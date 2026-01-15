"""Tests for the NeuralODE derivative model."""

import equinox as eqx
import pytest
from jax import grad
from jax import numpy as jnp

from dynax import NeuralODE


class TestNeuralODEForwardPass:
    """Test forward pass of NeuralODE."""

    def test_forward_pass_state_dependent_only(self, getkey):
        """Test forward pass with state-dependent model."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        dy = model(t=None, y=y, u=None)

        assert dy.shape == (3,)
        assert jnp.all(jnp.isfinite(dy))

    def test_forward_pass_with_time(self, getkey):
        """Test forward pass with time-dependent model."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=True,
            width_sizes=[16],
            key=key,
        )

        t = jnp.array(0.5)
        y = jnp.array([1.0, 2.0, 3.0])
        dy = model(t=t, y=y, u=None)

        assert dy.shape == (3,)
        assert jnp.all(jnp.isfinite(dy))

    def test_forward_pass_with_input(self, getkey):
        """Test forward pass with input."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])
        dy = model(t=None, y=y, u=u)

        assert dy.shape == (3,)
        assert jnp.all(jnp.isfinite(dy))

    def test_forward_pass_all_dependencies(self, getkey):
        """Test forward pass with time, state, and input dependencies."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            state_dependent=True,
            time_dependent=True,
            width_sizes=[16],
            key=key,
        )

        t = jnp.array(0.5)
        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])
        dy = model(t=t, y=y, u=u)

        assert dy.shape == (3,)
        assert jnp.all(jnp.isfinite(dy))

    def test_forward_pass_without_state(self, getkey):
        """Test forward pass with only input dependence."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            state_dependent=False,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        u = jnp.array([0.5, -0.5])
        dy = model(t=None, y=None, u=u)

        assert dy.shape == (3,)
        assert jnp.all(jnp.isfinite(dy))


class TestNeuralODEInputValidation:
    """Test input validation and error handling."""

    def test_time_required_for_time_dependent(self, getkey):
        """Test that time is required for time-dependent model."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=True,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])

        with pytest.raises(AssertionError):
            model(t=None, y=y, u=None)

    def test_time_must_be_scalar(self, getkey):
        """Test that time must be a scalar."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=True,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        t = jnp.array([0.5, 0.6])  # Not scalar

        with pytest.raises(AssertionError):
            model(t=t, y=y, u=None)

    def test_state_required_for_state_dependent(self, getkey):
        """Test that state is required for state-dependent model."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        with pytest.raises(AssertionError):
            model(t=None, y=None, u=None)

    def test_state_shape_validation(self, getkey):
        """Test that state shape is validated."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0])  # Wrong size

        with pytest.raises(AssertionError):
            model(t=None, y=y, u=None)

    def test_input_required_for_input_size_model(self, getkey):
        """Test that input is required when input_size > 0."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])

        with pytest.raises(AssertionError):
            model(t=None, y=y, u=None)

    def test_input_shape_validation(self, getkey):
        """Test that input shape is validated."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5])  # Wrong size

        with pytest.raises(AssertionError):
            model(t=None, y=y, u=u)

    def test_input_must_be_none_for_no_input_model(self, getkey):
        """Test that input must be None when input_size == 0."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])

        with pytest.raises(AssertionError):
            model(t=None, y=y, u=u)


class TestNeuralODEWithJAX:
    """Test integration with JAX transformations."""

    def test_jit_compilation(self, getkey):
        """Test that forward pass can be jitted with equinox.filter_jit."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])

        # Use equinox.filter_jit for eqx.Module
        jitted_forward = eqx.filter_jit(lambda m, t, y, u: m(t=t, y=y, u=u))

        dy = jitted_forward(model, None, y, u)

        assert dy.shape == (3,)
        assert jnp.all(jnp.isfinite(dy))

    def test_gradient_computation(self, getkey):
        """Test that gradients can be computed with respect to state."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=0,
            state_dependent=True,
            time_dependent=False,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])

        def loss_fn(y):
            dy = model(t=None, y=y, u=None)
            return jnp.sum(dy**2)

        loss = loss_fn(y)
        grads = grad(loss_fn)(y)

        assert jnp.all(jnp.isfinite(grads))
        assert grads.shape == y.shape

    def test_gradient_with_time_and_input(self, getkey):
        """Test gradient computation with time and input."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            state_dependent=True,
            time_dependent=True,
            width_sizes=[16],
            key=key,
        )

        t = jnp.array(0.5)
        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])

        def loss_fn(y):
            dy = model(t=t, y=y, u=u)
            return jnp.sum(dy**2)

        grads = grad(loss_fn)(y)

        assert jnp.all(jnp.isfinite(grads))
        assert grads.shape == y.shape

    def test_jit_with_all_arguments(self, getkey):
        """Test jitted forward pass with all arguments using filter_jit."""
        key = getkey()
        model = NeuralODE(
            state_size=2,
            input_size=1,
            state_dependent=True,
            time_dependent=True,
            width_sizes=[8],
            key=key,
        )

        jitted_forward = eqx.filter_jit(lambda m, t, y, u: m(t=t, y=y, u=u))

        t = jnp.array(0.1)
        y = jnp.array([1.0, 2.0])
        u = jnp.array([0.5])

        dy = jitted_forward(model, t, y, u)

        assert dy.shape == (2,)
        assert jnp.all(jnp.isfinite(dy))


class TestNeuralODEOutputProperties:
    """Test output properties and behavior."""

    def test_output_shape(self, getkey):
        """Test that output shape matches state size."""
        key = getkey()
        state_size = 5
        model = NeuralODE(
            state_size=state_size,
            input_size=3,
            width_sizes=[32],
            key=key,
        )

        y = jnp.ones(state_size)
        u = jnp.ones(3)
        dy = model(t=None, y=y, u=u)

        assert dy.shape == (state_size,)

    def test_output_dtype(self, getkey):
        """Test output dtype consistency."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])
        dy = model(t=None, y=y, u=u)

        assert dy.dtype == y.dtype

    def test_deterministic_forward_pass(self, getkey):
        """Test that forward pass is deterministic."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            width_sizes=[16],
            key=key,
        )

        y = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])

        dy1 = model(t=None, y=y, u=u)
        dy2 = model(t=None, y=y, u=u)

        assert jnp.allclose(dy1, dy2)

    def test_different_inputs_produce_different_outputs(self, getkey):
        """Test that different inputs produce different outputs."""
        key = getkey()
        model = NeuralODE(
            state_size=3,
            input_size=2,
            width_sizes=[16],
            key=key,
        )

        y1 = jnp.array([1.0, 2.0, 3.0])
        y2 = jnp.array([2.0, 3.0, 4.0])
        u = jnp.array([0.5, -0.5])

        dy1 = model(t=None, y=y1, u=u)
        dy2 = model(t=None, y=y2, u=u)

        # Outputs should be different for different inputs
        assert not jnp.allclose(dy1, dy2)

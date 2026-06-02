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

        x = jnp.array([1.0, 2.0, 3.0])
        dx = model(t=None, x=x, u=None, args=None)

        assert dx.shape == (3,)
        assert jnp.all(jnp.isfinite(dx))

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
        x = jnp.array([1.0, 2.0, 3.0])
        dx = model(t=t, x=x, u=None, args=None)

        assert dx.shape == (3,)

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

        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])
        dx = model(t=None, x=x, u=u, args=None)

        assert dx.shape == (3,)

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
        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, -0.5])
        dx = model(t=t, x=x, u=u, args=None)

        assert dx.shape == (3,)

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
        dx = model(t=None, x=None, u=u, args=None)

        assert dx.shape == (3,)

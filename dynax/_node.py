from collections.abc import Callable, Sequence

import equinox as eqx
import klax
from jax import numpy as jnp
from jax.nn import softplus
from jax.nn.initializers import Initializer, he_normal, zeros
from jaxtyping import Array, PRNGKeyArray, Scalar, Shaped
from klax._wrappers import Constraint, Unwrappable


class NeuralODE(eqx.Module):
    r"""Derivative model for a neural ODE.

    This implements the following equation:
    $$ \dot{y} = f(t, y, u) $$
    where $f$ is a multi-layer perceptron (MLP), depending on time $t\in\mathbb{R}$,
    the state vector $y(t)\in\mathbb{R}^n$, and an input vector $u(t)\in\mathbb{R}^m$.
    """

    mlp: klax.nn.MLP
    state_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    time_dependent: bool = eqx.field(static=True)
    state_dependent: bool = eqx.field(static=True)

    def __init__(
        self,
        state_size: int,
        input_size: int,
        *,
        time_dependent: bool = False,
        state_dependent: bool = True,
        width_sizes: Sequence[int],
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        weight_wrap: type[Constraint] | type[Unwrappable[Array]] | None = None,
        bias_wrap: type[Constraint] | type[Unwrappable[Array]] | None = None,
        dtype: type | None = None,
        key: PRNGKeyArray,
    ):
        """Create a neural ode.

        Args:
            state_size: Number of state variables.
            input_size: Number of input variables. Can be zero to indicate
                no input dependence.
            time_dependent: If true then the time is fed as an additional
                array entry to the MLP. Defaults to False.
            state_dependent: If true then the state is fed as an additional
                array entry to the MLP. Defaults to True.
            width_sizes: The sizes of each hidden layer in a list.
            weight_init: The weight initializer of type
                `jax.nn.initializers.Initializer`. (Defaults to he_normal().)
            bias_init: The bias initializer of type
                `jax.nn.initializers.Initializer`. (Defaults to zeros.)
            activation: The activation function after each hidden layer.
                (Defaults to `jax.nn.softplus`).
            final_activation: The activation function after the output layer.
                (Defaults to the identity.)
            use_bias: Whether to add on a bias to internal layers.
                (Defaults to `True`.)
            use_final_bias: Whether to add on a bias to the final layer.
                (Defaults to `True`.)
            weight_wrap: An optional wrapper that is passed to all weights.
            bias_wrap: An optional wrapper that is passed to all biases.
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for
                parameter initialisation.

        """
        self.state_size = state_size
        self.input_size = input_size
        self.time_dependent = time_dependent
        self.state_dependent = state_dependent

        in_size = state_dependent * state_size + input_size + time_dependent
        if in_size == 0:
            raise ValueError(
                "The neural ODE must depend on at least one of time, state, or input."
            )
        out_size = state_size
        self.mlp = klax.nn.MLP(
            in_size,
            out_size,
            width_sizes=width_sizes,
            weight_init=weight_init,
            bias_init=bias_init,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            weight_wrap=weight_wrap,
            bias_wrap=bias_wrap,
            dtype=dtype,
            key=key,
        )

    def __call__(
        self,
        t: Scalar | None,
        y: Shaped[Array, "n"] | None,
        u: Shaped[Array, "m"] | None,
    ) -> Shaped[Array, "n"]:
        """Evaluate the neural ODE's derivative."""
        nn_input = []

        if self.time_dependent:
            assert t is not None, (
                "Time t must not be None, since the model is time-dependent."
            )
            assert jnp.ndim(t) == 0, (
                f"Time t must be a scalar, but has shape {t.shape}."
            )
            t = jnp.expand_dims(t, axis=0)
            nn_input.append(t)

        if self.state_dependent:
            assert y is not None, (
                "State y must not be None, since the model is state-dependent."
            )
            assert y.shape[0] == self.state_size, (
                f"State y must have shape ({self.state_size},), but has shape {y.shape}."
            )
            nn_input.append(y)

        if self.input_size > 0:
            assert u is not None, (
                "Input u must not be None, since the model requires inputs. \
                    If the model should not depend on inputs, set input_size=0."
            )
            assert u.shape[0] == self.input_size, (
                f"Input u must have shape ({self.input_size},), but has shape {u.shape}."
            )
            nn_input.append(u)
        else:
            assert u is None, (
                "Input u must be None, since the model does not require inputs. \
                    If the model should depend on inputs, set input_size>0."
            )

        nn_input = jnp.concat(nn_input, axis=0)
        return self.mlp(nn_input)

from collections.abc import Callable, Sequence

import equinox as eqx
import klax
from jax.flatten_util import ravel_pytree
from jax.nn import softplus
from jax.nn.initializers import Initializer, he_normal, zeros
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar, Shaped
from klax._wrappers import Constraint, Unwrappable


class NeuralODE(eqx.Module):
    r"""Derivative model for a neural ODE.

    This implements the following equation:
    $$ \dot{y} = f(t, y, u; \text{args}) $$
    where $f$ is a multi-layer perceptron (MLP), depending on time $t\in\mathbb{R}$,
    the state vector $y(t)\in\mathbb{R}^n$, an input vector $u(t)\in\mathbb{R}^m$
    and a parameter vector $\text{args}\in\mathbb{R}^m$.
    """

    mlp: klax.nn.MLP
    state_size: int
    input_size: int
    parameter_size: int
    time_dependent: bool
    state_dependent: bool
    input_dependent: bool
    parameter_dependent: bool

    def __init__(
        self,
        state_size: int,
        input_size: int | None = None,
        parameter_size: int | None = None,
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
            state_size: Number of elements in the state vector `x`.
            input_size: Number of elements in the input vector `u`.
                Can be `None` to indicate no input dependence.
            parameter_size: Number of elements in the raveled parameter PyTree
                `args`; Essentially, the number of scalar parameters.
                Can be `None` to indicate no parameter dependence.
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
                parameter initialization.

        """
        self.state_size = state_size
        self.input_size = input_size
        self.parameter_size = parameter_size
        self.time_dependent = time_dependent
        self.state_dependent = state_dependent
        self.input_dependent = input_size is not None
        self.parameter_dependent = parameter_size is not None

        in_size = state_dependent * state_size + time_dependent
        if input_size is not None:
            in_size += input_size
        if parameter_size is not None:
            in_size += parameter_size

        if in_size == 0:
            raise ValueError(
                "The neural ODE must depend on at least one of time, state, input or parameters."
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
        x: Shaped[Array, "n"] | None,
        u: Shaped[Array, "m"] | None = None,
        args: PyTree[Array] | None = None,
    ) -> Shaped[Array, "n"]:
        """Evaluate the neural ODE's derivative."""
        tree = []

        if self.time_dependent:
            tree.append(t)

        if self.state_dependent:
            tree.append(x)

        if self.input_dependent:
            tree.append(u)

        if self.parameter_dependent:
            tree.append(args)

        try:
            flat, _ = ravel_pytree(tree)
        except Exception as e:
            raise ValueError(
                "Could not ravel all inputs into a single 1D array for the "
                "MLP. Perhaps you passed None as t, x, u or args even though "
                "the NeuralODE is set to depend on time, state, input or "
                f"parameters? Original error: {e}"
            ) from e

        return self.mlp(flat)

"""Port-Hamiltonian Systems."""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar

from ._typing import RealScalarLike


class ISOPHS(eqx.Module):
    R"""Input-State-Output port-Hamiltonian Systems (ISOPHS).

    Implementation of an input-state-output port-Hamiltonian system:

    $$ 
    \begin{align}
        \dot{\boldsymbol{x}} &= (\boldsymbol{J}-\boldsymbol{R})\frac{\partial\mathcal{H}}{\partial \boldsymbol{x}} + \boldsymbol{G}\boldsymbol{u}, \\
        \bigl[\boldsymbol{y} &= \boldsymbol{G}^\intercal \frac{\partial\mathcal{H}}{\partial \boldsymbol{x}}\bigr]
    \end{align}    
    $$

    where

    - $\boldsymbol{x}(t)$ is the state,
    - $\boldsymbol{u}(t)$ is the input,
    - $\boldsymbol{y}(t)$ is the output,
    - $\mathcal{H}(\boldsymbol{x}, \text{args})$ is the Hamiltonian function given by `hamiltonian`,
    - $\boldsymbol{J}(\boldsymbol{x}, \text{args})$ is the structure matrix given by `structure_matrix`,
    - $\boldsymbol{R}(\boldsymbol{x}, \text{args})$ is the dissipation matrix given by `dissipation_matrix`,
    - $\boldsymbol{G}(\boldsymbol{x}, \text{args})$ is the input matrix given by `input_matrix`,
    - and $\text{args}$ is an optional set of parameters.

    !!! note
        The usage of the output equation is optional, but requires that the input
        matrix (`input_matrix`) is defined.
    """

    hamiltonian: Callable[[Float[Array, "n"], PyTree], Scalar]
    structure_matrix: Callable[
        [Float[Array, "n"], PyTree], Float[Array, "n n"]
    ]  # noqa: F722, F821
    dissipation_matrix: (
        Callable[[Float[Array, "n"], PyTree], Float[Array, "n n"]] | None
    )  # noqa: F722, F821
    input_matrix: (
        Callable[[Float[Array, "n"], PyTree], Float[Array, "n m"]] | None
    )  # noqa: F722, F821

    def __init__(
        self,
        hamiltonian: Callable[[Float[Array, "n"], PyTree], Scalar],
        structure_matrix: Callable[
            [Float[Array, "n"], PyTree], Float[Array, "n n"]
        ],  # noqa: F722, F821
        dissipation_matrix: Callable[
            [Float[Array, "n"], PyTree], Float[Array, "n n"]
        ]  # noqa: F722, F821
        | None = None,
        input_matrix: Callable[
            [Float[Array, "n"], PyTree], Float[Array, "n m"]
        ]
        | None = None,  # noqa: F722, F821
    ):
        R"""Initialize the ISOPHS.

        Args:
            hamiltonian: Function or submodel computing the Hamiltonian
                $\mathcal{H}(\boldsymbol{x}, \text{args})$ as a function of the state vector
                and the parameters.
            structure_matrix: Function or submodel computing the structure
                matrix $\boldsymbol{J}(\boldsymbol{x}, \text{args})$ as a function of the state
                vector and the parameters. From the port-Hamiltonian modeling
                view the structure matrix is required to be skew-symmetric,
                i.e., $\boldsymbol{J} = -\boldsymbol{J}^\intercal$.
            dissipation_matrix: Function or submodel computing the dissipation
                matrix $\boldsymbol{R}(\boldsymbol{x}, \text{args})$ as a function of the state
                vector. From the port-Hamiltonian modeling view the dissipation
                matrix is required to be symmetric positive semi-definite, i.e.,
                $\boldsymbol{R} = \boldsymbol{R}^\intercal,\,\boldsymbol{R}\succ0$.
                `None` can be used to indicate that the system does not have a
                dissipation matrix.
            input_matrix: Function or submodel computing the input matrix
                $\boldsymbol{G}(\boldsymbol{x}, \text{args})$ as a function of the state vector.
                `None` can be used to indicate that the system does not have
                external inputs $\boldsymbol{u}$.
                Defaults to None.

        """
        #     Tipp:
        # Consider using [klax](https://drenderer.github.io/klax/)
        # for convenient implementations of [matrix valued functions](https://drenderer.github.io/klax/api/nn/matrices/).
        self.hamiltonian = hamiltonian
        self.structure_matrix = structure_matrix
        self.dissipation_matrix = dissipation_matrix
        self.input_matrix = input_matrix

    def __call__(
        self,
        t: RealScalarLike,
        x: Float[Array, "n"],
        u: Float[Array, "m"] | None = None,
        args: PyTree = None,
    ) -> Float[Array, "n"]:
        R"""Return the time derivative of the state vector $x$.

        Args:
            t: Scalar time for evaluation. This argument is unused, but required to
                comply with the [`ODESolver`][dynax.ODESolver] function signature.
            x: State vector at time `t`. This should be a 1D array of shape `(n,)`
                where `n` is the number of state dimensions.
            u: Input vector at time `t`. This should be a 1D array of shape `(m,)`
                where `m` is the number of input dimensions. If the system does not have
                inputs, this can be set to `None`.
            args: Additional arguments passed to $\mathcal{H}, \boldsymbol{J},
                \boldsymbol{R}$ and $\boldsymbol{G}$.

        Raises:
            ValueError: If a input is provided but the system does not have an input matrix.

        Returns:
            Time derivative of the state vector $\boldsymbol{x}$ as a 1D array of shape `(n,)`.

        """
        return self.state_equation(t, x, u, args)

    def state_equation(
        self,
        t: RealScalarLike,
        x: Float[Array, "n"],
        u: Float[Array, "m"] | None = None,
        args: PyTree = None,
    ) -> Float[Array, "n"]:
        R"""Compute the time derivative of the state vector $\dot{\boldsymbol{x}}$.

        Args:
            t: Scalar time for evaluation. This argument is unused, but required to
                comply with the [`ODESolver`][dynax.ODESolver] function signature.
            x: State vector at time `t`. This should be a 1D array of shape `(n,)`
                where `n` is the number of state dimensions.
            u: Input vector at time `t`. This should be a 1D array of shape `(m,)`
                where `m` is the number of input dimensions. If the system does not have
                inputs, this can be set to `None`.
            args: Additional arguments passed to $\mathcal{H}, \boldsymbol{J},
                \boldsymbol{R}$ and $\boldsymbol{G}$.

        Raises:
            ValueError: If a input is provided but the system does not have an input matrix.

        Returns:
            Time derivative of the state vector $\boldsymbol{x}$ as a 1D array of shape `(n,)`.

        """
        if self.input_matrix is None and u is not None:
            raise ValueError(
                "Input u passed to ISOPHS but the system has no input matrix."
            )
        if self.input_matrix is not None and u is None:
            raise ValueError(
                "The ISOPHS has an input matrix but no input u was provided."
            )

        sys_matrix = self.structure_matrix(x, args)

        if self.dissipation_matrix is not None:
            dissipation_matrix = self.dissipation_matrix(x, args)
            sys_matrix -= dissipation_matrix

        x_t = sys_matrix @ jax.grad(self.hamiltonian)(x, args)

        if self.input_matrix is not None:
            u = cast(Array, u)
            input_matrix = self.input_matrix(x, args)
            x_t += input_matrix @ u

        return x_t

    def output_equation(
        self,
        t: RealScalarLike,
        x: Float[Array, "n"],
        u: Float[Array, "m"] | None = None,
        args: PyTree = None,
    ) -> Float[Array, "p"]:
        R"""Compute the output $\boldsymbol{y}$.

        Args:
            t: Scalar time for evaluation. This argument is unused, but required to
                comply with the [`ODESolver`][dynax.ODESolver] function signature.
            x: State vector at time `t`. This should be a 1D array of shape `(n,)`
                where `n` is the number of state dimensions.
            u: Input vector at time `t`. This should be a 1D array of shape `(m,)`
                where `m` is the number of input dimensions. If the system does not have
                inputs, this can be set to `None`.
            args: Additional arguments passed to $\mathcal{H}, \boldsymbol{J},
                \boldsymbol{R}$ and $\boldsymbol{G}$.

        Raises:
            ValueError: If a input is provided but the system does not have an input matrix.

        Returns:
            Time derivative of the state vector $\boldsymbol{x}$ as a 1D array of shape `(n,)`.

        """
        if self.input_matrix is None:
            raise ValueError(
                "The system does not have an input matrix. However, to evaluate "
                "the output equation the system must have an input matrix."
            )

        input_matrix = self.input_matrix(x, args)
        return input_matrix.mT @ jax.grad(self.hamiltonian)(x, args)


class MatrixWrapper(eqx.Module):
    """Compatibility layer for using `klax` matrices with the call interface required by [`ISOPHS`][dynax.ISOPHS].

    Matrix-valued functions implemented in `klax` currently only support one
    input argument `x -> A`. However, the matrix-valued functions required for
    the [input-state-output port-Hamiltonian system][dynax.ISOPHS] demand the
    call signature `(x, args) -> A`. This class can wrap the `klax` matrix-valued
    functions to comply with the latter call signature.
    It supports 4 modes depending on `state_dependent` and `parameter_dependent`:

    - If both `state_dependent=parameter_dependent=True`, then the `x` and `args`
    are combined using `jax.flatten_util.ravel_pytree([x, args])` and the result
    is passed to the wrapped matrix-valued function.
    - If only `state_dependent=True`, then `x` is passed to the matrix-valued
    function.
    - If only `parameter_dependent=True`, then `args` is passed to the
    matrix-valued function.
    - If both `state_dependent=parameter_dependent=False`, then `None` is passed
    to the matrix-valued function.
    """

    matrix: Callable[[Array], Array]
    state_dependent: bool = False
    parameter_dependent: bool = False

    def __call__(
        self, x: PyTree[Array] | None, args: PyTree[Array] | None
    ) -> Float[Array, "i j"]:
        tree = []
        if self.state_dependent:
            tree.append(x)
        if self.parameter_dependent:
            tree.append(args)
        flat, _ = jax.flatten_util.ravel_pytree(tree)
        return self.matrix(flat)

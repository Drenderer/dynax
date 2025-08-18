R"""Input-State port-Hamiltonian Systems (ISPHS)."""

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float, Scalar


class ISPHS(eqx.Module):
    R"""Input-State port-Hamiltonian Systems (ISPHS).

    Implementation of a output-less port-Hamiltonian system:
    $$ \dot{x} = (J(x)-R(x))\frac{\partial \mathcal{H}}{\partial x} + B(x)u $$
    where:
        $x(t)$ is the state vector,
        $u(t)$ is the input vector,
        $\mathcal{H}(x)$ is the Hamiltonian function given by `hamiltonian`,
        $J$ is the structure matrix given by `structure_matrix`,
        $R$ is the dissipation matrix given by `dissipation_matrix`,
        and $B$ is the input matrix given by `input_matrix`.
    """

    hamiltonian: Callable[[Array], Scalar]
    structure_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]]  # noqa: F722, F821
    dissipation_matrix: (
        Callable[[Float[Array, "n"]], Float[Array, "n n"]] | None
    )  # noqa: F722, F821
    input_matrix: Callable[[Float[Array, "n"]], Float[Array, "n m"]] | None  # noqa: F722, F821

    def __init__(
        self,
        hamiltonian: Callable[[Array], Scalar],
        structure_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]],  # noqa: F722, F821
        dissipation_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]]  # noqa: F722, F821
        | None = None,
        input_matrix: Callable[[Float[Array, "n"]], Float[Array, "n m"]]
        | None = None,  # noqa: F722, F821
    ):
        R"""Initialize the ISPHS.

        Args:
            hamiltonian: Function or submodel computing the Hamiltonian $\mathcal{H}(x)$
                as a function of the state vector.
            structure_matrix: Function or submodel computing the structure matrix $J(x)$
                as a function of the state vector. From the port-Hamiltonian modeling view
                the structure matrix is required to be skew-symmetric, i.e., $J = -J^T$.
            dissipation_matrix: Function or submodel computing the dissipation matrix $R(x)$
                as a function of the state vector. From the port-Hamiltonian modeling view
                the dissipation matrix is required to be symmetric positive semi-definite,
                i.e., $R = R^T,\,R\succ0$. Alternatively, `None` can be used to indicate
                that the system does not have a dissipation matrix $R$.
            input_matrix: Function or submodel computing the input matrix $G(x)$ as a
                function of the state vector. Alternatively, `None` can be used to indicate
                that the system does not have external inputs $u$. Defaults to None.

        Tipp:
            Consider using [klax](https://drenderer.github.io/klax/api/training/#klax.fit)
            for convenient implementations of [matrix valued functions](https://drenderer.github.io/klax/api/nn/matrices/).

        """
        self.hamiltonian = hamiltonian
        self.structure_matrix = structure_matrix
        self.dissipation_matrix = dissipation_matrix
        self.input_matrix = input_matrix

    def __call__(self, t: Scalar, x: Array, u: Array | None = None) -> Array:
        """Return the time derivative of the state vector $x$.

        Args:
            t: Scalar time for evaluation. This argument is unused, but required to
                comply with the [`ODESolver`][dynax.ODESolver] function signature.
            x: State vector at time `t`. This should be a 1D array of shape `(n,)`
                where `n` is the number of state dimensions.
            u: Input vector at time `t`. This should be a 1D array of shape `(m,)`
                where `m` is the number of input dimensions. If the system does not have
                inputs, this can be set to `None`.

        Raises:
            ValueError: If a input is provided but the system does not have an input matrix.

        Returns:
            Time derivative of the state vector $x$ as a 1D array of shape `(n,)`.

        """
        structure_matrix = self.structure_matrix(x)

        if self.dissipation_matrix is not None:
            dissipation_matrix = self.dissipation_matrix(x)
            structure_matrix -= dissipation_matrix

        x_t = structure_matrix @ jax.grad(self.hamiltonian)(x)

        if self.input_matrix is not None:
            if u is None:
                raise ValueError(
                    "The ISPHS has an input matrix but no input u was provided."
                )
            input_matrix = self.input_matrix(x)
            x_t += input_matrix @ u

        return x_t

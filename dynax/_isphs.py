"""
This file implements learnable Input-State port-Hamiltonian Systems (ISPHS) with eqvloution equations:
x_t = (J - R) dH_dx + B u
"""

import equinox as eqx
import jax
from jaxtyping import Array, Scalar, Float
from typing import Callable


class ISPHS(eqx.Module):
    hamiltonian: Callable[[Array], Scalar]
    structure_matrix: Callable[[Float[Array, "s"]], Float[Array, "s s"]]  # noqa: F722, F821
    dissipation_matrix: Callable[[Float[Array, "s"]], Float[Array, "s s"]] | None  # noqa: F722, F821
    input_matrix: Callable[[Float[Array, "s"]], Float[Array, "s i"]] | None  # noqa: F722, F821

    def __init__(
        self,
        hamiltonian: Callable[[Array], Scalar],
        structure_matrix: Callable[[Float[Array, "s"]], Float[Array, "s s"]],  # noqa: F722, F821
        dissipation_matrix: Callable[[Float[Array, "s"]], Float[Array, "s s"]]  # noqa: F722, F821
        | None = None,
        input_matrix: Callable[[Float[Array, "s"]], Float[Array, "s i"]] | None = None,  # noqa: F722, F821
    ):
        self.hamiltonian = hamiltonian
        self.structure_matrix = structure_matrix
        self.dissipation_matrix = dissipation_matrix
        self.input_matrix = input_matrix

    def __call__(self, t: Scalar, x: Array, u: Array | None) -> Array:
        J = self.structure_matrix(x)

        if self.dissipation_matrix is not None:
            R = self.dissipation_matrix(x)
            J -= R

        x_t = J @ jax.grad(self.hamiltonian)(x)

        if self.input_matrix is not None:
            if u is None:
                raise ValueError(
                    "The ISPHS has an input matrix but no input u was provided."
                )
            G = self.input_matrix(x)
            x_t += G @ u

        return x_t

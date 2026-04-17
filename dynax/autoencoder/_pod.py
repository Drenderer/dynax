from typing import Literal, cast

from jax import numpy as jnp
from jaxtyping import Array, Shaped
from numpy.typing import NDArray


def get_svd(
    snapshots: Shaped[NDArray | Array, "n_batch n_time n_space"],
) -> tuple[
    Shaped[Array, "n_batch n_time n_time"],
    Shaped[Array, "n_time"],
    Shaped[Array, "n_time n_space"],
]:
    """Calculate the (stacked) svd for snapshot matrices.

    Snapshots must be 3D. Useful for calculating the modes across multiple trajectories.
    The trajectories are stacked along the time dimension before calculating the SVD.
    Afterwards the left singular vectors are reshaped to `snapshot.shape`, such that
    `snapshot == U @ jnp.diag(s) @ V`.

    Note:
        This does not calculate the full square matrices for the left and right singular
        vectors `U`, `V`. Instead, only the parts corresponding to non-zero singular
        values are kept, resulting in an output shape that depends on
        `x=min(n_batch*n_time, n_space)`.

    Args:
        snapshots (Array): Array of trajectories, shape `(n_batch, n_time, n_space)`.

    Returns:
        SVD-matrices:
        - U - Left singular matrix, shape `(n_batch, n_time, x)`
        - s - singular values, shape `(x,)`
        - V - right singular matrix, shape `(x, n_space)`

    """
    n_batch, n_time, _ = snapshots.shape
    snapshots_stacked = jnp.concatenate(snapshots, axis=0)
    U, s, V = jnp.linalg.svd(snapshots_stacked, full_matrices=False)  # noqa: N806
    U = jnp.reshape(U, (n_batch, n_time, -1))  # noqa: N806

    return U, s, V


class PODLatentSpace:
    """Proper orthogonal decomposition based latent space.

    Linear projections to and from a latent space using the right
    singular vectors from the SVD of snapshots with as modes/basis.

    When projecting into the latent space, the input is first shifted
    (optional), then scaled and finally projected onto the modes.
    By default, the scaling is computed such that the latent variables
    of the snapshots used to calculate the modes have standard deviation
    one.

    For the projection into the full order space, the latent variables
    are first multiplied with the modes, then scaled and finally shifted.
    """

    shift: Shaped[Array, "1 1 n_space"]
    scale: Shaped[Array, ""]
    modes: Shaped[Array, "n_time n_space"]
    singular_values: Shaped[Array, "n_time"]
    _num_modes: int

    def __init__(
        self,
        snapshots: Shaped[NDArray | Array, "n_batch n_time n_space"],
        num_modes: int,
        shift: float | Shaped[NDArray | Array, "#n_space"] | None = None,
        scaling: float
        | Shaped[NDArray | Array, ""]
        | Literal["normalize"]
        | None = "normalize",
    ) -> None:
        """Initialize the SVD latent space.

        Note:
            After initialization you can still change the number of modes,
            i.e., the latent space dimension.

        Args:
            snapshots: Snapshots of shape `(n_batch, n_time, n_space)`.
            num_modes: Number of modes to use in the latent space.
            shift: Array or float to shift a snapshot by, before transforming.
                By setting `shift=<special state>`, the special state is
                transformed to the origin of the latent space.
            scaling: Scalar value to scale all latent variables by.
                Can be `"normalize"`, in which case the scale is automatically
                chosen, such that the latent variables of `snapshots` have
                standard deviation one.

        """
        n_batch, n_time, n_space = snapshots.shape

        if shift is None:
            self.shift = jnp.array(0.0)
        else:
            shift = jnp.atleast_1d(shift)
            assert shift.shape == (1,) or shift.shape == (n_space,), (
                "Shift must be 1-dimensional and of shape (n_space,)."
            )
            self.shift = shift
        _snapshot = snapshots - self.shift

        left_singular_vectors, singular_values, right_singular_vectors = (
            get_svd(_snapshot)
        )

        if scaling is None:
            self.scale = jnp.array(1.0)
        elif scaling == "normalize":
            # Compute scaling such that the latent variables have std=1
            latent_variables = left_singular_vectors * singular_values
            self.scale = jnp.std(latent_variables[..., :num_modes])
        else:
            scale = jnp.array(scaling)
            assert scale.ndim == 0
            self.scale = scale

        self.modes = right_singular_vectors
        self.singular_values = singular_values
        self.num_modes = num_modes

    @property
    def num_modes(self) -> int:
        return self._num_modes

    @num_modes.setter
    def num_modes(self, value: int):
        self._num_modes = min(value, self.modes.shape[0])

    def to_latent(
        self,
        snapshot: Shaped[NDArray | Array, "... n_space"],
    ) -> Shaped[Array, "... n_modes"]:
        """Project snapshot into the latent space.

        Args:
            snapshot: Data with shape `(..., n_space)`.

        Returns:
            Array of latent variables with shape `(..., n_modes)`.

        """
        _snapshot = snapshot - self.shift
        latent_variables = (
            _snapshot @ self.modes[: self.num_modes].T / self.scale
        )
        return cast(Array, latent_variables)

    def from_latent(
        self,
        latent_variables: Shaped[NDArray | Array, "... n_modes"],
    ) -> Shaped[Array, "... n_space"]:
        """Project latent variables into the full order space.

        Args:
            latent_variables: Data with shape `(..., n_modes)`.

        Returns:
            Array with shape `(..., n_space)`.

        """
        snapshot = (
            latent_variables @ self.modes[: self.num_modes] * self.scale
            + self.shift
        )
        return cast(Array, snapshot)

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import numpy as np
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PyTree
from numpy.typing import NDArray

from ._misc import default_floating_dtype

# Single axis indexing type for numpy arrays
type Index1D = int | slice | Array | NDArray[np.integer] | NDArray[np.bool]
type Index = Index1D | tuple[Index1D, ...]


def differentiate[T: PyTree](ts: Float[Array, "k"], ys: T) -> T:
    """Approximate the derivatives of a timeseries.

    Returns:
        PyTree of derivatives of `ys` evaluated at `ts`.

    """
    coeffs = backward_hermite_coefficients(ts, ys)
    interp = CubicInterpolation(ts, coeffs)
    return jax.vmap(interp.derivative)(ts)


class Trajectory(eqx.Module):
    """Simple dataclass for storing trajectory like data.

    The fields are:
        - ts: Timestamps, (typically) expected shape (...,)
        - ys: State vectors, (typically) expected shape (..., n)
        - y_ts: State derivatives, (typically) expected shape (..., n)
        - us: Inputs/Excitations, (typically) expected shape (..., m)
    """

    _ts: Array | None
    _ys: Array | None
    _y_ts: Array | None
    _us: Array | None

    def __init__(
        self,
        ts: Any = None,
        ys: Any = None,
        *,
        y_ts: Any = None,
        us: Any = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Initialize a Trajectory and converts all arrays to jax arrays.

        Args:
            ts: Timestamps. Defaults to None.
            ys: States. Defaults to None.
            y_ts: State derivatives. Defaults to None.
            us: Input/Exciataions. Defaults to None.
            dtype: Data type of the arrays after conversion to jax arrays.
                Defaults to None.

        """
        dtype = default_floating_dtype() if dtype is None else dtype
        self._ts = None if ts is None else jnp.asarray(ts, dtype=dtype)
        self._ys = None if ys is None else jnp.asarray(ys, dtype=dtype)
        self._y_ts = None if y_ts is None else jnp.asarray(y_ts, dtype=dtype)
        self._us = None if us is None else jnp.asarray(us, dtype=dtype)

    # The following properties are only introduced for typing reasons.
    # This way any property is guaranteed of type Array.
    @property
    def ts(self) -> Array:
        if self._ts is None:
            raise ValueError(f"'ts' not set for {self!r}.")
        return self._ts

    @property
    def ys(self) -> Array:
        if self._ys is None:
            raise ValueError(f"'ys' not set for {self!r}.")
        return self._ys

    @property
    def y_ts(self) -> Array:
        if self._y_ts is None:
            raise ValueError(f"'y_ts' not set for {self!r}.")
        return self._y_ts

    @property
    def us(self) -> Array:
        if self._us is None:
            raise ValueError(f"'us' not set for {self!r}.")
        return self._us

    def __getitem__(self, index: Index) -> "Trajectory":
        return jax.tree.map(lambda x: x[index], self)

    @staticmethod
    def stack(
        trajectories: Sequence["Trajectory"],
        axis: int = 0,
        dtype: DTypeLike | None = None,
    ) -> "Trajectory":
        """Broadcasted stacking of trajectories.

        Args:
            trajectories: Sequence of trajectories to stack.
            axis: Axis along which to stack. Defaults to 0.
            dtype: Optional dtype of the resulting array.
                If not specified, the dtype will be determined via type promotion
                rules described in type-promotion. Defaults to None.

        Returns:
            Trajectory containing the stacked arrays.

        """
        return jax.tree.map(
            lambda *x: jnp.stack(x, axis=axis, dtype=dtype), *trajectories
        )

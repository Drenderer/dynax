from collections.abc import Sequence
from pathlib import Path
from typing import Literal, overload

import equinox as eqx
import jax
import numpy as np
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jaxtyping import Array, ArrayLike, Float, PyTree
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ._misc import default_floating_dtype

# Single axis indexing type for numpy arrays
type Index1D = int | slice | NDArray[np.integer] | NDArray[np.bool] | Array


def differentiate[T: PyTree](ts: Float[Array, "k"], ys: T) -> T:
    """Approximate the derivatives of a timeseries.

    Returns:
        PyTree of derivatives of `ys` evaluated at `ts`.

    """
    coeffs = backward_hermite_coefficients(ts, ys)
    interp = CubicInterpolation(ts, coeffs)
    return jax.vmap(interp.derivative)(ts)


@overload
def _as_nonsqueezing_index[T: tuple[Index1D, ...]](index: T) -> T: ...
@overload
def _as_nonsqueezing_index(index: Index1D) -> tuple[Index1D]: ...
def _as_nonsqueezing_index(index):
    """Modify an index passed to __getitem__ of a numpy array to never drop dimensions.

    Args:
        index: Index that could be passed to `np.ndarray.__getitem__`.
            Example: `(1, slice(1,3), np.array([True, False, False]))`

    Returns:
        Modified index

    """
    if not isinstance(index, tuple):
        index = (index,)

    index = tuple(
        slice(int(i), int(i) + 1)
        if (
            isinstance(i, (int,))
            or (isinstance(i, (np.ndarray, Array)) and i.ndim == 0)
        )
        else i
        for i in index
    )
    return index


class _BaseTrajectory(eqx.Module):
    """Base class for trajectories."""

    ts: NDArray
    ys: NDArray
    _y_ts: NDArray | None
    _us: NDArray | None

    def __init__(
        self,
        ts: ArrayLike,
        ys: ArrayLike,
        y_ts: ArrayLike | None = None,
        us: ArrayLike | None = None,
        dtype: DTypeLike = None,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        self.ts = np.asarray(ts, dtype=dtype)
        self.ys = np.asarray(ys, dtype=dtype)
        self._y_ts = None if y_ts is None else np.asarray(y_ts, dtype=dtype)
        self._us = None if us is None else np.asarray(us, dtype=dtype)

    # Making y_ts, and us a property enables more intuitive typing
    # since any type checker knows that traj.y_ts is of type NDArray
    # and not of type None.
    @property
    def y_ts(self) -> NDArray:
        if self._y_ts is None:
            raise ValueError("Trajectory does not contain derivatives y_ts.")
        return self._y_ts

    @y_ts.setter
    def y_ts(self, y_ts):
        self._y_ts = y_ts

    @property
    def us(self) -> NDArray:
        if self._us is None:
            raise ValueError("Trajectory does not contain inputs us.")
        return self._us

    @us.setter
    def us(self, us):
        self._us = us

    def save(self, file: Path, overwrite: bool = False):
        if not overwrite and file.exists():
            raise FileExistsError(
                "File already exists. Set overwrite=True to overwrite."
            )
        data = dict(ts=self.ts, ys=self.ys)
        if self._y_ts is not None:
            data["y_ts"] = self._y_ts
        if self._us is not None:
            data["us"] = self._us
        np.savez(file, **data, allow_pickle=False)

    @classmethod
    def load(cls, file):
        data = np.load(file, allow_pickle=False)
        return cls(**data)


class Trajectory(_BaseTrajectory):
    """Represents a single trajectory."""

    def __init__(
        self,
        ts: ArrayLike,
        ys: ArrayLike,
        y_ts: ArrayLike | None = None,
        us: ArrayLike | None = None,
        dtype: DTypeLike = None,
    ):
        """Create a Trajectory from array-like objects.

        This will convert all inputs to numpy arrays.

        Args:
            ts: Array of timestamps with `shape=(k,)`
            ys: Array of states with `shape=(k, n)`
            y_ts: Optional array of state derivatives with `shape=(k, n)`. Defaults to None.
            us: Optional array of inputs with `shape=(k, m)`. Defaults to None.
            dtype: Optionally specify the dtype for the conversion to numpy arrays.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.

        """
        super().__init__(ts, ys, y_ts, us, dtype)

    def __check_init__(self):
        """Check if the provided fields are compatible.

        Raises ValueError if:
            - The dimensionalities of `ts`, `ys`, `y_ts` or `us` are incorrect.
            - The the first dimension of `ys`, `y_ts` or `us` do not match `ts.size`
            - The the second dimension of `y_ts` does not match the second dimension of `ys`
        """
        if self.ts.ndim != 1:
            raise ValueError("Timestamps ts must be 1-dimensinonal.")
        if self.ys.ndim != 2:
            raise ValueError("States ys must be 2-dimensinonal.")
        if self._y_ts is not None and self._y_ts.ndim != 2:
            raise ValueError("States derivatives y_ts must be 2-dimensinonal.")
        if self._us is not None and self._us.ndim != 2:
            raise ValueError("Inputs us must be 2-dimensinonal.")

        time_size = self.ts.size
        for field in ["ys", "_y_ts", "_us"]:
            f = getattr(self, field)
            if f is None:
                continue
            if f.shape[0] != time_size:
                raise ValueError(
                    f"Shape missmatch between ts of size {time_size} and {field.lstrip('_')} of shape {f.shape}."
                )
        if self._y_ts is not None and self._y_ts.shape[1] != self.ys.shape[1]:
            raise ValueError(
                f"Shape missmatch between ys of shape {self.ys.shape} and y_ts of shape {self._y_ts.shape}."
            )

    def __str__(self) -> str:
        """Get string representation."""
        time_size = self.ts.size
        state_size = self.ys.shape[1]
        if self._us is not None:
            input_size = self._us.shape[1]
            return f"{self.__class__.__name__} with {time_size} {'timestamp' if time_size == 1 else 'timestamps'}, state size {state_size} and input size {input_size}."
        else:
            return f"{self.__class__.__name__} with {time_size} {'timestamp' if time_size == 1 else 'timestamps'} and state size {state_size}."

    def __getitem__(self, index: Index1D | tuple[Index1D]) -> "Trajectory":
        """Index a `Trajectory` like a numpy array along the time axis."""
        index = _as_nonsqueezing_index(index)
        new_ts = self.ts[index]
        new_ys = self.ys[index]
        new_y_ts = None if self._y_ts is None else self._y_ts[index]
        new_us = None if self._us is None else self._us[index]
        return Trajectory(new_ts, new_ys, new_y_ts, new_us)

    def __setitem__(
        self,
        index: Index1D,
        values: dict[Literal["ts", "ys", "y_ts", "us"], ArrayLike],
    ):
        """Set the values at a given index along the time axis.

        Args:
            index: 1D slice index.
            values: Dictionary of values.

        Example:
            ```python
            >>> import numpy as np
            >>> from dynax import Trajectory
            >>> traj = Trajectory(np.arange(5), np.zeros((5,3)))
            >>> traj[2] = dict(ts=2.5, ys=np.array([1,2,3]))
            >>> traj.ts
            array([0. , 1. , 2.5, 3. , 4. ], dtype=float32)
            ```

        """
        for name, value in values.items():
            try:
                getattr(self, name)[index] = value
            except ValueError as e:
                e.add_note("Cannot set value.")
                raise


class TrajectoryCollection(_BaseTrajectory):
    """Represents a collection of trajectories sharing the same timestamps."""

    def __init__(
        self,
        ts: ArrayLike,
        ys: ArrayLike,
        y_ts: ArrayLike | None = None,
        us: ArrayLike | None = None,
        dtype: DTypeLike = None,
    ):
        """Create a TrajectoryCollection from array-like objects.

        This will convert all inputs to numpy arrays.

        Args:
            ts: Array of timestamps with `shape=(k,)`
            ys: Array of states with `shape=(b, k, n)`
            y_ts: Optional array of state derivatives with `shape=(b, k, n)`. Defaults to None.
            us: Optional array of inputs with `shape=(b, k, m)`. Defaults to None.
            dtype: Optionally specify the dtype for the conversion to numpy arrays.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.

        """
        dtype = default_floating_dtype() if dtype is None else dtype
        self.ts = np.asarray(ts, dtype=dtype)
        self.ys = np.asarray(ys, dtype=dtype)
        self._y_ts = None if y_ts is None else np.asarray(y_ts, dtype=dtype)
        self._us = None if us is None else np.asarray(us, dtype=dtype)

    @classmethod
    def from_trajectories(
        cls, trajectories: Sequence[Trajectory], dtype: DTypeLike = None
    ):
        """Create a collection by stacking trajectories with the same timestamps.

        Args:
            trajectories: Sequence of trajectories.
            dtype: Optional dtype argument provided to `np.stack`.
                Defaults to None, which let's numpy decide which dtype to use.

        Raises:
            ValueError: If the trajectories timestamps differ or the sequence is empty.

        Returns:
            `TrajectoryCollection` of stacked trajectories.

        """
        if len(trajectories) == 0:
            raise ValueError("Need at least one Trajectory.")
        _ts = trajectories[0].ts
        if not all(
            [np.array_equal(_ts, traj.ts) for traj in trajectories[1:]]
        ):
            raise ValueError(
                "All trajectories must share the same timestamps."
            )
        data = dict(ts=_ts)
        for field_name in ["ys", "_y_ts", "_us"]:
            data[field_name] = np.stack(
                [getattr(traj, field_name) for traj in trajectories], axis=0
            )
        ts = _ts
        ys = np.stack([traj.ys for traj in trajectories], dtype=dtype)
        y_ts_mask = [traj._y_ts is not None for traj in trajectories]
        if all(y_ts_mask):
            y_ts = np.stack([traj.y_ts for traj in trajectories], dtype=dtype)
        elif not any(y_ts_mask):
            y_ts = None
        else:
            raise ValueError(
                "Some trajectories have y_ts while others do not. Cannot stack."
            )
        us_mask = [traj._us is not None for traj in trajectories]
        if all(us_mask):
            us = np.stack([traj.us for traj in trajectories], dtype=dtype)
        elif not any(us_mask):
            us = None
        else:
            raise ValueError(
                "Some trajectories have us while others do not. Cannot stack."
            )

        return cls(ts=ts, ys=ys, y_ts=y_ts, us=us)

    def __check_init__(self):
        """Check if the provided fields are compatible.

        Raises ValueError if:
            - The dimensionalities of `ts`, `ys`, `y_ts` or `us` are incorrect.
            - The the first dimension of `ys`, `y_ts` or `us` do not match.
            - The the second dimension of `ys`, `y_ts` or `us` do not match `ts.size`
            - The the second dimension of `y_ts` does not match the second dimension of `ys`
        """
        if self.ts.ndim != 1:
            raise ValueError("Timestamps ts must be 1-dimensinonal.")
        if self.ys.ndim != 3:
            raise ValueError("States ys must be 3-dimensinonal.")
        if self._y_ts is not None and self._y_ts.ndim != 3:
            raise ValueError("States derivatives y_ts must be 3-dimensinonal.")
        if self._us is not None and self._us.ndim != 3:
            raise ValueError("Inputs us must be 3-dimensinonal.")

        batch_size = self.ys.shape[0]
        time_size = self.ts.size
        for field in ["ys", "_y_ts", "_us"]:
            f = getattr(self, field)
            if f is None:
                continue
            if f.shape[0] != batch_size:
                raise ValueError(
                    f"Shape missmatch between batch sizes of {field} of shape {f.shape} and ys with batch size {batch_size}."
                )
            if f.shape[1] != time_size:
                raise ValueError(
                    f"Shape missmatch between ts of size {time_size} and {field.lstrip('_')} of shape {f.shape}."
                )
        if (
            self._y_ts is not None
            and self._y_ts.shape[-1] != self.ys.shape[-1]
        ):
            raise ValueError(
                f"Shape missmatch between ys of shape {self.ys.shape} and y_ts of shape {self._y_ts.shape}."
            )

    def __str__(self) -> str:
        """Get string representation."""
        batch_size = self.ys.shape[0]
        time_size = self.ts.size
        state_size = self.ys.shape[-1]
        if self._us is not None:
            input_size = self._us.shape[-1]
            return f"{self.__class__.__name__} containing {batch_size} {'trajectories' if batch_size == 1 else 'trajectory'} with {time_size} timestamps, state size {state_size} and input size {input_size}."
        else:
            return f"{self.__class__.__name__} containing {batch_size} {'trajectories' if batch_size == 1 else 'trajectory'} with {time_size} timestamps and state size {state_size}."

    def __getitem__(
        self, index: Index1D | tuple[Index1D] | tuple[Index1D, Index1D]
    ) -> "TrajectoryCollection":
        """Index a `TrajectoryCollection` like a numpy array along the batch and time axes."""
        index = _as_nonsqueezing_index(index)
        new_ts = self.ts[index[1]] if len(index) == 2 else self.ts
        new_ys = self.ys[index]
        new_y_ts = None if self._y_ts is None else self._y_ts[index]
        new_us = None if self._us is None else self._us[index]
        return TrajectoryCollection(new_ts, new_ys, new_y_ts, new_us)

    def __setitem__(
        self,
        index: Index1D | tuple[Index1D] | tuple[Index1D, Index1D],
        values: dict[Literal["ts", "ys", "y_ts", "us"], ArrayLike],
    ):
        """Set the values at a given index along the batch and/or time axes/axis.

        Args:
            index: 1D slice index.
            values: Dictionary of values.`

        """
        for name, value in values.items():
            try:
                getattr(self, name)[index] = value
            except ValueError as e:
                e.add_note("Cannot set value.")
                raise

    def get_trajectory(self, index: int):
        """Retrieve trajectory at a given index.

        Args:
            index: Index of the trajectory along the batch dimension.

        Returns:
            Trajectory.

        """
        new_ys = self.ys[index]
        new_y_ts = None if self._y_ts is None else self._y_ts[index]
        new_us = None if self._us is None else self._us[index]
        return Trajectory(ts=self.ts, ys=new_ys, y_ts=new_y_ts, us=new_us)

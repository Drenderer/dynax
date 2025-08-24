# %%
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any, Literal, TypeVarTuple, Unpack, overload

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Integer
from numpy.typing import ArrayLike, NDArray

# Single axis indexing type for numpy arrays
type Index1D = int | slice | NDArray[np.integer] | NDArray[np.bool] | Array

## Define converters

# def _as_numpy_1darray(ts: Any) -> NDArray:
#     """Similar to numpy's `atleast_1d`, but raises a ValueError if the input has more than one dimension."""
#     ts = np.atleast_1d(ts)
#     if ts.ndim != 1:
#         raise ValueError("Array can not have more than one dimensions.")
#     return ts

# def _as_numpy_2darray(x: Any) -> NDArray:
#     """Similar to numpy's `atleast_2d`, except:
#     - appends the new axis in the last dimension.
#     - raises a ValueError if the input has more than two dimensions.
#     """  # noqa: D205
#     x = np.asarray(x)
#     if x.ndim == 0:
#         return x.reshape(1, 1)
#     elif x.ndim == 1:
#         return x[:, np.newaxis]
#     elif x.ndim == 2:
#         return x
#     else:
#         raise ValueError("Array can not have more than two dimensions.")


# def _as_numpy_3darray(x: Any) -> NDArray:
#     """Similar to numpy's `atleast_3d`, except:
#     - appends the new axis/axes in the last dimension(s).
#     - raises a ValueError if the input has more than three dimensions.
#     """  # noqa: D205
#     x = np.asarray(x)
#     if x.ndim == 0:
#         return x.reshape(1, 1, 1)
#     elif x.ndim == 1:
#         return x[:, np.newaxis, np.newaxis]
#     elif x.ndim == 2:
#         return x[:, :, np.newaxis]
#     elif x.ndim == 3:
#         return x
#     else:
#         raise ValueError("Array can not have more than three dimensions.")


# def _as_numpy_2darray_or_none(x: Any) -> NDArray | None:
#     if x is None:
#         return None
#     return _as_numpy_2darray(x)


# def _as_numpy_3darray_or_none(x: Any) -> NDArray | None:
#     if x is None:
#         return None
#     return _as_numpy_3darray(x)


def _as_numpy_or_none(x: Any) -> NDArray | None:
    if x is None:
        return None
    return np.asarray(x)


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


class Trajectory(eqx.Module):
    """Represents a single trajectory."""

    ts: NDArray = eqx.field(converter=np.asarray)
    ys: NDArray = eqx.field(converter=np.asarray)
    y_ts: NDArray | None = eqx.field(converter=_as_numpy_or_none, default=None)
    us: NDArray | None = eqx.field(converter=_as_numpy_or_none, default=None)

    def __check_init__(self):
        """Check if the provided fields are compatible.

        Raises ValueError if:
            - The the first dimension of `ys`, `y_ts` or `us` do not match `ts.size`
            - The the second dimension of `y_ts` does not match the second dimension of `ys`
        """
        if self.ts.ndim != 1:
            raise ValueError("Timestamps ts must be 1-dimensinonal.")
        if self.ys.ndim != 2:
            raise ValueError("States ys must be 2-dimensinonal.")
        if self.y_ts is not None and self.y_ts.ndim != 2:
            raise ValueError("States derivatives y_ts must be 2-dimensinonal.")
        if self.us is not None and self.us.ndim != 2:
            raise ValueError("Inputs us must be 2-dimensinonal.")

        time_size = self.ts.size
        for field in ["ys", "y_ts", "us"]:
            f = getattr(self, field)
            if f is None:
                continue
            if f.shape[0] != time_size:
                raise ValueError(
                    f"Shape missmatch between ts of size {time_size} and {field} of shape {f.shape}."
                )
        if self.y_ts is not None and self.y_ts.shape[1] != self.ys.shape[1]:
            raise ValueError(
                f"Shape missmatch between ys of shape {self.ys.shape} and y_ts of shape {self.y_ts.shape}."
            )

    def __str__(self) -> str:
        """Get string representation."""
        time_size = self.ts.size
        state_size = self.ys.shape[1]
        if self.us is not None:
            input_size = self.us.shape[1]
            return f"{self.__class__.__name__} with {time_size} timestamps, state size {state_size} and input size {input_size}."
        else:
            return f"{self.__class__.__name__} with {time_size} timestamps and state size {state_size}."

    def __getitem__(self, index: Index1D) -> "Trajectory":
        """Index a `Trajectory` like a numpy array along the time axis."""
        sliced_fields = dict(
            ts=self.ts[index],
            ys=np.atleast_2d(self.ys[index]),
            y_ts=None
            if self.y_ts is None
            else np.atleast_2d(self.y_ts[index]),
            us=None if self.us is None else np.atleast_2d(self.us[index]),
        )
        return Trajectory(**sliced_fields)

    def __setitem__(
        self,
        index: Index1D,
        values: dict[Literal["ts", "ys", "y_ts", "us"], ArrayLike],
    ):
        """Set the values at a given index along the time axis.

        Args:
            index: 1D slice index.
            values: Dictionary of values. E.g.: `{"ts": 1, "ys": np.array([...]), ...}`

        """
        for name, value in values.items():
            getattr(self, name)[index] = value


class TrajectoryCollection(eqx.Module):
    """Represents a collection of trajectories sharing the same timestamps."""

    ts: NDArray = eqx.field(converter=np.asarray)
    ys: NDArray = eqx.field(converter=np.asarray)
    y_ts: NDArray | None = eqx.field(converter=_as_numpy_or_none, default=None)
    us: NDArray | None = eqx.field(converter=_as_numpy_or_none, default=None)

    @classmethod
    def from_trajectories(cls, trajectories: Sequence[Trajectory]):
        """Create a collection by stacking trajectories with the same timestamps.

        Args:
            trajectories: Sequence of trajectories.

        Raises:
            ValueError: If the trajectories timestamps differ of the sequence is empty.

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
        for field_name in ["ys", "y_ts", "us"]:
            data[field_name] = np.stack(
                [getattr(traj, field_name) for traj in trajectories], axis=0
            )
        return cls(**data)

    def __check_init__(self):
        """Check if the provided fields are compatible.

        Raises ValueError if:
            - The the first dimension of `ys`, `y_ts` or `us` do not match.
            - The the second dimension of `ys`, `y_ts` or `us` do not match `ts.size`
            - The the second dimension of `y_ts` does not match the second dimension of `ys`
        """
        if self.ts.ndim != 1:
            raise ValueError("Timestamps ts must be 1-dimensinonal.")
        if self.ys.ndim != 3:
            raise ValueError("States ys must be 3-dimensinonal.")
        if self.y_ts is not None and self.y_ts.ndim != 3:
            raise ValueError("States derivatives y_ts must be 3-dimensinonal.")
        if self.us is not None and self.us.ndim != 3:
            raise ValueError("Inputs us must be 3-dimensinonal.")

        batch_size = self.ys.shape[0]
        time_size = self.ts.size
        for field in ["ys", "y_ts", "us"]:
            f = getattr(self, field)
            if f is None:
                continue
            if f.shape[0] != batch_size:
                raise ValueError(
                    f"Shape missmatch between batch sizes of {field} of shape {f.shape} and ys with batch size {batch_size}."
                )
            if f.shape[1] != time_size:
                raise ValueError(
                    f"Shape missmatch between ts of size {time_size} and {field} of shape {f.shape}."
                )
        if self.y_ts is not None and self.y_ts.shape[-1] != self.ys.shape[-1]:
            raise ValueError(
                f"Shape missmatch between ys of shape {self.ys.shape} and y_ts of shape {self.y_ts.shape}."
            )

    def __str__(self) -> str:
        """Get string representation."""
        batch_size = self.ys.shape[0]
        time_size = self.ts.size
        state_size = self.ys.shape[-1]
        if self.us is not None:
            input_size = self.us.shape[-1]
            return f"{self.__class__.__name__} containing {batch_size} {'trajectories' if batch_size == 1 else 'trajectory'} with {time_size} timestamps, state size {state_size} and input size {input_size}."
        else:
            return f"{self.__class__.__name__} containing {batch_size} {'trajectories' if batch_size == 1 else 'trajectory'} with {time_size} timestamps and state size {state_size}."

    def __getitem__(
        self, index: Index1D | tuple[Index1D] | tuple[Index1D, Index1D]
    ) -> "TrajectoryCollection":
        """Index a `TrajectoryCollection` like a numpy array along the batch and time axes."""
        index = _as_nonsqueezing_index(index)

        sliced_fields = dict(
            ts=self.ts[index[1]] if len(index) == 2 else self.ts,
            ys=self.ys[index],
            y_ts=None if self.y_ts is None else self.y_ts[index],
            us=None if self.us is None else self.us[index],
        )
        return TrajectoryCollection(**sliced_fields)

    def __setitem__(
        self,
        index: Index1D | tuple[Index1D] | tuple[Index1D, Index1D],
        values: dict[Literal["ts", "ys", "y_ts", "us"], ArrayLike],
    ):
        """Set the values at a given index along the batch and/or time axes/axis.

        Args:
            index: 1D slice index.
            values: Dictionary of values. E.g.: `{"ts": 1, "ys": np.array([...]), ...}`

        """
        for name, value in values.items():
            getattr(self, name)[index] = value

    def approximate_state_derivative(self):
        pass

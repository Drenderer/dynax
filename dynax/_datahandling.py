from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, overload

import jax
import numpy as np
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, DTypeLike, Float, PyTree
from numpy.typing import NDArray

from ._misc import default_floating_dtype

# Single axis indexing type for numpy arrays
type Index1D = int | slice | Array | NDArray[np.integer] | NDArray[np.bool]


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
    """Modify an index passed to __getitem__ of an array to never drop dimensions.

    Args:
        index: Index that could be passed to `array.__getitem__`.
            Example: `(1, slice(1,3), jnp.array([True, False, False]))`

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


class HasDType(Protocol):
    dtype: DTypeLike  # could refine to np.dtype[Any] if you want


class ArrayField:
    """Descriptor.

      - stores value in _< name > on the instance
      - converts assigned ArrayLike -> jnp.asarray
      - raises ValueError on get if value is None
    Typing: __get__ is overloaded so that instance-access returns Array.
    """

    def __set_name__(self, owner: type[object], name: str) -> None:  # noqa: D105
        self.public_name = name
        self.private_name = "_" + name
        self.availability_name = "has_" + name

    @overload
    def __get__(
        self, instance: None, owner: type[object] | None
    ) -> "ArrayField": ...
    @overload
    def __get__(
        self, instance: HasDType, owner: type[object] | None
    ) -> Array: ...
    def __get__(
        self, instance: HasDType | None, owner: type[object] | None
    ) -> "ArrayField | Array":  # noqa: D105
        if instance is None:
            return self
        value: Array = getattr(instance, self.private_name)
        if value is None:
            raise ValueError(
                f"Field {self.public_name!r} is not defined on {instance!r}."
            )
        return value

    def __set__(self, instance: HasDType, value: ArrayLike | None) -> None:  # noqa: D105
        available = value is not None
        if available:
            value = jnp.asarray(value, dtype=instance.dtype)
        setattr(instance, self.availability_name, available)
        setattr(instance, self.private_name, value)


class BaseTrajectory:
    """Base class for objects representing trajectory related data.

    Such classes can contain
    - ts: Timestamps
    - ys: States
    - y_ts: State derivatives
    - us: Inputs

    These attributes are handled as properties to enable:
    - Automatic conversion to jax.numpy arrays upon setting values
    - Raising value errors on retrieving unset values (None-values)
        This enables intuitive typing, since obj.us is always of type
        Array.

    """

    ts: ArrayField = ArrayField()
    ys: ArrayField = ArrayField()
    y_ts: ArrayField = ArrayField()
    us: ArrayField = ArrayField()
    # has_<...> attributes for checking if certain fields are set.
    # They are set by the ArrayField Descriptor and are necessary
    # since `instance.ts is None` will raise if ts is None.
    has_ts: bool
    has_ys: bool
    has_y_ts: bool
    has_us: bool
    dtype: DTypeLike

    def __init__(
        self,
        ts: ArrayLike | None = None,
        ys: ArrayLike | None = None,
        y_ts: ArrayLike | None = None,
        us: ArrayLike | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        self.dtype = default_floating_dtype() if dtype is None else dtype
        self.ts = ts
        self.ys = ys
        self.y_ts = y_ts
        self.us = us

    def save(self, file: Path | str, overwrite: bool = False) -> None:
        file = Path(file)
        if not overwrite and file.exists():
            raise FileExistsError(
                "File already exists. Set overwrite=True to overwrite."
            )
        data = dict()
        if self.has_ts:
            data["ts"] = self.ts
        if self.has_ys:
            data["ys"] = self.ys
        if self.has_y_ts:
            data["y_ts"] = self.y_ts
        if self.has_us:
            data["us"] = self.us
        jnp.savez(file, **data, allow_pickle=False)

    @classmethod
    def load[T](cls: type[T], file) -> T:
        # Usage of np here: jnp.load() for .npz files returns numpy arrays anyways.
        data = np.load(file, allow_pickle=False)
        return cls(**data)


class Trajectory(BaseTrajectory):
    """Represents a single trajectory."""

    def __init__(
        self,
        ts: ArrayLike,
        ys: ArrayLike,
        *,
        y_ts: ArrayLike | None = None,
        us: ArrayLike | None = None,
        dtype: DTypeLike | None = None,
    ):
        """Create a Trajectory from array-like objects.

        This will convert all inputs to jax.numpy arrays.

        Args:
            ts: Array of timestamps with `shape=(k,)`
            ys: Array of states with `shape=(k, n)`
            y_ts: Optional array of state derivatives with `shape=(k, n)`. Defaults to None.
            us: Optional array of inputs with `shape=(k, m)`. Defaults to None.
            dtype: Optionally specify the dtype for the conversion to arrays.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.

        """
        super().__init__(ts=ts, ys=ys, y_ts=y_ts, us=us, dtype=dtype)
        self.__check_init__()

    def __check_init__(self) -> None:
        """Check if the provided fields are compatible.

        Raises ValueError if:
            - The dimensionalities of `ts` and `ys` are incorrect.
            - The the shapes of `y_ts` and `us` do not match the expected shapes computed from ts and ys.
            - The the second dimension of `y_ts` does not match the second dimension of `ys`
        """
        # Check ts
        if self.ts.ndim != 1:
            raise ValueError("Timestamps 'ts' must be 1-dimensinonal.")
        time_size = self.ts.size

        # Check ys
        if self.ys.ndim != 2:
            raise ValueError("States 'ys' must be 2-dimensinonal.")
        if self.ys.shape[0] != time_size:
            raise ValueError(
                f"Shape missmatch between 'ts' of size {time_size} and 'ys' of shape {self.ys.shape}."
            )
        state_size = self.ys.shape[1]

        # Check y_ts
        if self.has_y_ts and self.y_ts.shape != (time_size, state_size):
            raise ValueError(
                f"Shape missmatch for 'y_ts'. Expected shape {(time_size, state_size)!r} but got {self.y_ts.shape}."
            )

        # Check us
        if self.has_us and (
            self.us.shape[0] != time_size or self.us.ndim != 2
        ):
            raise ValueError(
                f"Shape missmatch for 'y_ts'. Expected shape {(time_size, 'any')!r} but got {self.us.shape}."
            )

    def __str__(self) -> str:
        """Get string representation."""
        time_size = self.ts.size
        state_size = self.ys.shape[1]
        if self.has_us:
            input_size = self.us.shape[1]
            return f"{self.__class__.__name__} with {time_size} {'timestamp' if time_size == 1 else 'timestamps'}, state size {state_size} and input size {input_size}."
        else:
            return f"{self.__class__.__name__} with {time_size} {'timestamp' if time_size == 1 else 'timestamps'} and state size {state_size}."

    def __getitem__(self, index: Index1D | tuple[Index1D]) -> "Trajectory":
        """Index a `Trajectory` like an array along the time axis."""
        index = _as_nonsqueezing_index(index)
        new_ts = self.ts[index]
        new_ys = self.ys[index]
        new_y_ts = self.y_ts[index] if self.has_y_ts else None
        new_us = self.us[index] if self.has_us else None
        return Trajectory(ts=new_ts, ys=new_ys, y_ts=new_y_ts, us=new_us)

    def __setitem__(
        self,
        index: Index1D,
        values: dict[Literal["ts", "ys", "y_ts", "us"], ArrayLike],
    ) -> None:
        """Set the values at a given index along the time axis.

        Args:
            index: 1D slice index.
            values: Dictionary of values.

        Example:
            ```python
            >>> import jax.numpy as jnp
            >>> from dynax import Trajectory
            >>> traj = Trajectory(jnp.arange(5), jnp.zeros((5,3)))
            >>> traj[2] = dict(ts=2.5, ys=jnp.array([1,2,3]))
            >>> traj.ts
            array([0. , 1. , 2.5, 3. , 4. ], dtype=float32)
            ```

        """
        for name, value in values.items():
            try:
                setattr(self, name, getattr(self, name).at[index].set(value))
            except ValueError as e:
                e.add_note("Cannot set value.")
                raise


class TrajectoryCollection(BaseTrajectory):
    """Represents a collection of trajectories sharing the same timestamps."""

    def __init__(
        self,
        ts: ArrayLike,
        ys: ArrayLike,
        *,
        y_ts: ArrayLike | None = None,
        us: ArrayLike | None = None,
        dtype: DTypeLike | None = None,
    ):
        """Create a TrajectoryCollection from array-like objects.

        This will convert all inputs to jax.numpy arrays.

        Args:
            ts: Array of timestamps with `shape=(k,)`
            ys: Array of states with `shape=(b, k, n)`
            y_ts: Optional array of state derivatives with `shape=(b, k, n)`. Defaults to None.
            us: Optional array of inputs with `shape=(b, k, m)`. Defaults to None.
            dtype: Optionally specify the dtype for the conversion to numpy arrays.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.

        """
        super().__init__(ts=ts, ys=ys, y_ts=y_ts, us=us, dtype=dtype)
        self.__check_init__()

    @classmethod
    def from_trajectories(
        cls, trajectories: Sequence[Trajectory], dtype: DTypeLike | None = None
    ) -> "TrajectoryCollection":
        """Create a collection by stacking trajectories with the same timestamps.

        Args:
            trajectories: Sequence of trajectories.
            dtype: Optional dtype argument provided to `jnp.stack`.
                Defaults to None, which let's jax.numpy decide which dtype to use.

        Raises:
            ValueError: If the trajectories timestamps differ or the sequence is empty.

        Returns:
            `TrajectoryCollection` of stacked trajectories.

        """
        if len(trajectories) == 0:
            raise ValueError("Need at least one Trajectory.")
        _ts = trajectories[0].ts
        if not all(
            [jnp.array_equal(_ts, traj.ts) for traj in trajectories[1:]]
        ):
            raise ValueError(
                "All trajectories must share the same timestamps."
            )
        ts = _ts
        ys = jnp.stack([traj.ys for traj in trajectories], dtype=dtype)
        y_ts_mask = [traj.has_y_ts for traj in trajectories]
        if all(y_ts_mask):
            y_ts = jnp.stack([traj.y_ts for traj in trajectories], dtype=dtype)
        elif not any(y_ts_mask):
            y_ts = None
        else:
            raise ValueError(
                "Some trajectories have y_ts while others do not. Cannot stack."
            )
        us_mask = [traj.has_us for traj in trajectories]
        if all(us_mask):
            us = jnp.stack([traj.us for traj in trajectories], dtype=dtype)
        elif not any(us_mask):
            us = None
        else:
            raise ValueError(
                "Some trajectories have us while others do not. Cannot stack."
            )

        return cls(ts=ts, ys=ys, y_ts=y_ts, us=us)

    def __check_init__(self) -> None:
        """Check if the provided fields are compatible.

        Raises ValueError if:
            - The dimensionalities of `ts` and `ys` are incorrect.
            - The the shapes of `y_ts` and `us` do not match the expected shapes computed from ts and ys.
        """
        # Check ts
        if self.ts.ndim != 1:
            raise ValueError("Timestamps 'ts' must be 1-dimensinonal.")
        time_size = self.ts.size

        # Check ys
        if self.ys.ndim != 3:
            raise ValueError("States 'ys' must be 3-dimensinonal.")
        if self.ys.shape[1] != time_size:
            raise ValueError(
                f"Shape missmatch between 'ts' of size {time_size} and 'ys' of shape {self.ys.shape}."
            )

        batch_size = self.ys.shape[0]
        state_size = self.ys.shape[2]

        # Check y_ts
        if self.has_y_ts and self.y_ts.shape != (
            batch_size,
            time_size,
            state_size,
        ):
            raise ValueError(
                f"Shape missmatch for 'y_ts'. Expected shape {(batch_size, time_size, state_size)!r} but got {self.y_ts.shape}."
            )

        # Check us
        if self.has_us and (
            self.us.shape[:2] != (batch_size, time_size) or self.us.ndim != 3
        ):
            raise ValueError(
                f"Shape missmatch for 'y_ts'. Expected shape {(batch_size, time_size, 'any')!r} but got {self.us.shape}."
            )

    def __str__(self) -> str:
        """Get string representation."""
        batch_size = self.ys.shape[0]
        time_size = self.ts.size
        state_size = self.ys.shape[-1]
        if self.has_us:
            input_size = self.us.shape[-1]
            return f"{self.__class__.__name__} containing {batch_size} {'trajectories' if batch_size == 1 else 'trajectory'} with {time_size} timestamps, state size {state_size} and input size {input_size}."
        else:
            return f"{self.__class__.__name__} containing {batch_size} {'trajectories' if batch_size == 1 else 'trajectory'} with {time_size} timestamps and state size {state_size}."

    def __getitem__(
        self, index: Index1D | tuple[Index1D] | tuple[Index1D, Index1D]
    ) -> "TrajectoryCollection":
        """Index a `TrajectoryCollection` like an array along the batch and time axes."""
        index = _as_nonsqueezing_index(index)
        new_ts = self.ts[index[1]] if len(index) == 2 else self.ts
        new_ys = self.ys[index]
        new_y_ts = self.y_ts[index] if self.has_y_ts else None
        new_us = self.us[index] if self.has_us else None
        return TrajectoryCollection(
            ts=new_ts, ys=new_ys, y_ts=new_y_ts, us=new_us
        )

    def __setitem__(
        self,
        index: Index1D | tuple[Index1D] | tuple[Index1D, Index1D],
        values: dict[Literal["ts", "ys", "y_ts", "us"], ArrayLike],
    ) -> None:
        """Set the values at a given index along the batch and/or time axes/axis.

        Args:
            index: 1D slice index.
            values: Dictionary of values.`

        """
        for name, value in values.items():
            try:
                setattr(self, name, getattr(self, name).at[index].set(value))
            except ValueError as e:
                e.add_note("Cannot set value.")
                raise

    def get_trajectory(self, index: int) -> Trajectory:
        """Retrieve trajectory at a given index.

        Args:
            index: Index of the trajectory along the batch dimension.

        Returns:
            Trajectory.

        """
        new_ys = self.ys[index]
        new_y_ts = self.y_ts[index] if self.has_y_ts else None
        new_us = self.us[index] if self.has_us else None
        return Trajectory(ts=self.ts, ys=new_ys, y_ts=new_y_ts, us=new_us)


class DerivativeCollection(BaseTrajectory):
    """Collection of state-derivative pairs with optional timestamps and inputs."""

    def __init__(
        self,
        ys: ArrayLike,
        y_ts: ArrayLike,
        *,
        ts: ArrayLike | None = None,
        us: ArrayLike | None = None,
        dtype: DTypeLike | None = None,
    ):
        """Create a Derivative collection from array-like objects.

        This will convert all inputs to jax.numpy arrays.

        Args:
            ys: Array of states with `shape=(b, n)`
            y_ts: Array of state derivatives with `shape=(b, n)`. Defaults to None.
            ts: Optional Array of timestamps with `shape=(b,)`
            us: Optional array of inputs with `shape=(b, m)`. Defaults to None.
            dtype: Optionally specify the dtype for the conversion to arrays.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.

        """
        super().__init__(ts=ts, ys=ys, y_ts=y_ts, us=us, dtype=dtype)
        self.__check_init__()

    def __check_init__(self) -> None:
        """Check if the provided fields are compatible.

        Raises ValueError if:
            - The dimensionalities of `ts`, `ys`, `y_ts` or `us` are incorrect.
            - The first dimensions of `ts`, `ys`, `y_ts` or `us` are not the same
            - The the second dimension of `y_ts` does not match the second dimension of `ys`
        """
        # Check ys
        if self.ys.ndim != 2:
            raise ValueError("States 'ys' must be 2-dimensinonal.")

        batch_size = self.ys.shape[0]
        state_size = self.ys.shape[1]

        # Check ts
        if self.has_ts and self.ts.shape != (batch_size,):
            raise ValueError(
                f"Shape missmatch for 'ts'. Expected shape {(batch_size,)!r} but got {self.ts.shape}."
            )

        # Check y_ts
        if self.has_y_ts and self.y_ts.shape != (batch_size, state_size):
            raise ValueError(
                f"Shape missmatch for 'y_ts'. Expected shape {(batch_size, state_size)!r} but got {self.y_ts.shape}."
            )

        # Check us
        if self.has_us and (self.us.shape[0] != batch_size or self.us.ndim != 2):
            raise ValueError(
                f"Shape missmatch for 'y_ts'. Expected shape {(batch_size, 'any')!r} but got {self.us.shape}."
            )

    @classmethod
    def from_trajectory_collection(
        cls, coll: TrajectoryCollection
    ) -> "DerivativeCollection":
        if not coll.has_y_ts:
            raise ValueError(
                "TrajectoryCollection must have derivatives in order to create a DerivativeCollection"
            )
        batch_size = coll.ys.shape[0]
        ts = jnp.tile(coll.ts, (batch_size, 1))
        ts = ts.reshape(-1)  # flatten
        state_size = coll.ys.shape[2]
        ys = coll.ys.reshape(-1, state_size)
        y_ts = coll.y_ts.reshape(-1, state_size)
        if coll.has_us:
            input_size = coll.us.shape[2]
            us = coll.us.reshape(-1, input_size)
        else:
            us = None
        return cls(ts=ts, ys=ys, y_ts=y_ts, us=us)

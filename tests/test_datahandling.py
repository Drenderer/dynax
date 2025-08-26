import jax
import jax.numpy as jnp
import pytest
from jax import random as jr

from dynax._datahandling import (
    DerivativeCollection,
    Trajectory,
    TrajectoryCollection,
    _as_nonsqueezing_index,
    differentiate,
)


def test_differentiate():
    ts = jnp.linspace(0, 1, 100)
    ys = (ts, jnp.stack([jnp.sin(ts), jnp.cos(ts)], axis=1))
    y_ts = (jnp.ones_like(ts), jnp.stack([jnp.cos(ts), -jnp.sin(ts)], axis=1))
    y_ts_ = differentiate(ts, ys)
    assert jax.tree.structure(y_ts) == jax.tree.structure(y_ts_)
    assert jax.tree.map(jnp.allclose, y_ts, y_ts_)


@pytest.mark.parametrize(
    "index, expected_shape",
    [
        # Single integer index (should add axis)
        (0, (1, 4, 5)),
        # Single slice (should not change shape)
        (slice(1, 3), (2, 4, 5)),
        # Single boolean array
        (jnp.array([True, False, True]), (2, 4, 5)),
        # Single 0-dim array (should add axis)
        (jnp.array(0), (1, 4, 5)),
        (jnp.array(0), (1, 4, 5)),
        # Tuple: int, slice, int
        ((1, slice(2, 4), 0), (1, 2, 1)),
        # Tuple: 0-dim array, slice, int
        ((jnp.array(0), slice(2, 4), 0), (1, 2, 1)),
        # Tuple: int, int, int
        ((1, 2, 0), (1, 1, 1)),
        # Tuple: slice, slice, slice
        ((slice(0, 2), slice(1, 3), slice(0, 2)), (2, 2, 2)),
        # Tuple: int, boolean array, int
        ((0, jnp.array([True, False, True, False]), 0), (1, 2, 1)),
        # Tuple: 0-dim jax array, numpy array, int
        ((jnp.array(0), jnp.array([1, 2]), 0), (1, 2, 1)),
        # Empty tuple:
        ((), (3, 4, 5)),
    ],
)
def test_as_nonsqueezing_index(index, expected_shape):
    arr = jnp.zeros((3, 4, 5))
    idx = _as_nonsqueezing_index(index)
    result = arr[idx]
    assert result.shape == expected_shape


# TODO: Test dtype
class TestTrajectory:
    def test_basic_init(self, getkey):
        ts = jnp.linspace(0, 1, 5)
        ys = jr.normal(getkey(), (5, 3))
        traj = Trajectory(ts=ts, ys=ys)
        assert jnp.allclose(traj.ts, ts)
        assert jnp.allclose(traj.ys, ys)
        assert not traj.has_y_ts
        assert not traj.has_us

    def test_optional_fields(self, getkey):
        ts = jnp.linspace(0, 1, 4)
        ys = jr.normal(getkey(), (4, 2))
        y_ts = jr.normal(getkey(), (4, 2))
        us = jr.normal(getkey(), (4, 1))
        traj = Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)
        assert jnp.allclose(traj.y_ts, y_ts)
        assert jnp.allclose(traj.us, us)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
    def test_dtype(self, dtype):
        traj = Trajectory(ts=[1, 2, 3], ys=jnp.zeros((3, 2)), dtype=dtype)
        assert traj.ts.dtype == dtype
        assert traj.ys.dtype == dtype

    @pytest.mark.parametrize(
        "ts, ys, y_ts, us",
        [
            (jnp.zeros((2, 2)), jnp.zeros((2, 2)), None, None),
            (jnp.zeros(3), jnp.zeros((3,)), None, None),
            (jnp.zeros(3), jnp.zeros((3, 2)), jnp.zeros((3,)), None),
            (jnp.zeros(3), jnp.zeros((3, 2)), None, jnp.zeros((3,))),
            (jnp.zeros(3), jnp.zeros((2, 2)), None, None),
            (jnp.zeros(3), jnp.zeros((3, 2)), jnp.zeros((3, 3)), None),
        ],
    )
    def test_check_init_invalid(self, ts, ys, y_ts, us):
        with pytest.raises(ValueError):
            Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)

    def test_getitem(self):
        ts = jnp.arange(5)
        ys = jnp.zeros((5, 2))
        y_ts = jnp.zeros((5, 2))
        us = jnp.zeros((5, 1))
        traj = Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)
        sub = traj[2]
        assert sub.ts.shape == (1,)
        assert sub.ys.shape == (1, 2)
        assert sub.y_ts.shape == (1, 2)
        assert sub.us.shape == (1, 1)

    def test_setitem(self):
        ts = jnp.arange(3)
        ys = jnp.zeros((3, 2))
        traj = Trajectory(ts=ts, ys=ys)
        traj[1] = {"ys": jnp.ones((2,))}
        assert jnp.allclose(traj.ys[1], 1)

    def test_save_and_load(self, tmp_path, getkey):
        ts = jnp.arange(5)
        ys = jr.normal(getkey(), (5, 3))
        y_ts = jr.normal(getkey(), (5, 3))
        us = jr.normal(getkey(), (5, 2))
        traj = Trajectory(ts, ys, y_ts=y_ts, us=us)
        file = tmp_path / "traj.npz"
        traj.save(file)
        loaded = Trajectory.load(file)
        assert jnp.allclose(loaded.ts, ts)
        assert jnp.allclose(loaded.ys, ys)
        assert jnp.allclose(loaded.y_ts, y_ts)
        assert jnp.allclose(loaded.us, us)

    def test_save_and_load_partial_fields(self, tmp_path, getkey):
        ts = jnp.arange(3)
        ys = jr.normal(getkey(), (3, 2))
        # Only ys, no y_ts or us
        traj = Trajectory(ts, ys)
        file = tmp_path / "traj.npz"
        traj.save(file)
        loaded = Trajectory.load(file)
        assert jnp.allclose(loaded.ts, ts)
        assert jnp.allclose(loaded.ys, ys)
        assert not loaded.has_y_ts
        assert not loaded.has_us

    def test_save_overwrite_protection(self, tmp_path):
        ts = jnp.arange(3)
        ys = jnp.zeros((3, 1))
        traj = Trajectory(ts, ys)
        file = tmp_path / "traj.npz"
        traj.save(file)
        with pytest.raises(FileExistsError):
            traj.save(file)
        # Should succeed with overwrite=True
        traj.save(file, overwrite=True)


class TestTrajectoryCollection:
    def test_from_trajectories(self):
        ts = jnp.linspace(0, 1, 4)
        ys1 = jnp.ones((4, 2))
        ys2 = jnp.zeros((4, 2))
        t1 = Trajectory(ts=ts, ys=ys1)
        t2 = Trajectory(ts=ts, ys=ys2)
        coll = TrajectoryCollection.from_trajectories([t1, t2])
        assert coll.ys.shape == (2, 4, 2)
        assert jnp.allclose(coll.ts, ts)

    def test_from_trajectories_mismatch(self):
        ts1 = jnp.linspace(0, 1, 3)
        ts2 = jnp.linspace(0, 2, 3)
        t1 = Trajectory(ts=ts1, ys=jnp.ones((3, 2)))
        t2 = Trajectory(ts=ts2, ys=jnp.zeros((3, 2)))
        with pytest.raises(
            ValueError,
            match="All trajectories must share the same timestamps.",
        ):
            TrajectoryCollection.from_trajectories([t1, t2])

    def test_check_init_valid(self):
        ts = jnp.linspace(0, 1, 2)
        ys = jnp.zeros((3, 2, 2))
        y_ts = jnp.zeros((3, 2, 2))
        us = jnp.zeros((3, 2, 1))
        coll = TrajectoryCollection(ts=ts, ys=ys, y_ts=y_ts, us=us)
        coll.__check_init__()  # Should not raise

    @pytest.mark.parametrize(
        "ts,ys,y_ts,us",
        [
            (jnp.zeros((2, 2)), jnp.zeros((2, 2)), None, None),  # ts shape
            (jnp.zeros(3), jnp.zeros((3, 1)), None, None),  # ys shape
            (
                jnp.zeros(3),
                jnp.zeros((1, 3, 2)),
                jnp.zeros((3,)),
                None,
            ),  # y_ts shape
            (
                jnp.zeros(3),
                jnp.zeros((1, 3, 2)),
                None,
                jnp.zeros(
                    (
                        3,
                        4,
                    )
                ),
            ),  # us shape
            (
                jnp.zeros(3),
                jnp.zeros((1, 2, 2)),
                None,
                None,
            ),  # time doesn't match
            (
                jnp.zeros(3),
                jnp.zeros((1, 3, 2)),
                jnp.zeros((1, 3, 3)),
                None,
            ),  # state doesn't match
        ],
    )
    def test_check_init_invalid(self, ts, ys, y_ts, us):
        with pytest.raises(ValueError):
            TrajectoryCollection(ts=ts, ys=ys, y_ts=y_ts, us=us)

    def test_getitem(self):
        ts = jnp.arange(4)
        ys = jnp.zeros((10, 4, 2))
        y_ts = jnp.zeros((10, 4, 2))
        us = jnp.zeros((10, 4, 3))
        coll = TrajectoryCollection(ts=ts, ys=ys, y_ts=y_ts, us=us)
        sub = coll[1, 2]
        assert jnp.allclose(sub.ts, ts[2])
        assert sub.ys.shape == (1, 1, 2)
        assert sub.y_ts.shape == (1, 1, 2)
        assert sub.us.shape == (1, 1, 3)

    def test_gettrajectory(self):
        ts = jnp.arange(4)
        ys = jnp.zeros((10, 4, 2))
        y_ts = jnp.zeros((10, 4, 2))
        us = jnp.zeros((10, 4, 3))
        coll = TrajectoryCollection(ts=ts, ys=ys, y_ts=y_ts, us=us)
        traj = coll.get_trajectory(0)
        assert isinstance(traj, Trajectory)
        assert jnp.allclose(traj.ts, coll.ts)
        assert jnp.allclose(traj.ys, coll.ys[0])
        assert jnp.allclose(traj.y_ts, coll.y_ts[0])
        assert jnp.allclose(traj.us, coll.us[0])

    def test_setitem(self):
        ts = jnp.arange(2)
        ys = jnp.zeros((2, 2, 2))
        coll = TrajectoryCollection(ts=ts, ys=ys)
        coll[(1, 1)] = {"ys": jnp.ones((2,))}
        assert jnp.allclose(coll.ys[1, 1], 1)

    def test_save_and_load(self, tmp_path, getkey):
        ts = jnp.arange(4)
        ys = jr.normal(getkey(), (2, 4, 3))
        y_ts = jr.normal(getkey(), (2, 4, 3))
        us = jr.normal(getkey(), (2, 4, 2))
        dset = TrajectoryCollection(ts, ys, y_ts=y_ts, us=us)
        file = tmp_path / "dset.npz"
        dset.save(file)
        loaded = TrajectoryCollection.load(file)
        assert jnp.allclose(loaded.ts, ts)
        assert jnp.allclose(loaded.ys, ys)
        assert jnp.allclose(loaded.y_ts, y_ts)
        assert jnp.allclose(loaded.us, us)

    def test_save_and_load_partial_fields(self, tmp_path, getkey):
        ts = jnp.arange(3)
        ys_col = jr.normal(getkey(), (2, 3, 2))
        dset = TrajectoryCollection(ts, ys_col)
        file = tmp_path / "dset.npz"
        dset.save(file)
        loaded = TrajectoryCollection.load(file)
        assert jnp.allclose(loaded.ts, ts)
        assert jnp.allclose(loaded.ys, ys_col)
        assert not loaded.has_y_ts
        assert not loaded.has_us


class TestDerivativeCollection:
    def test_from_trajectory_collection(self, getkey):
        ts = jnp.arange(3)
        ys = jr.normal(getkey(), (2, 3, 2))
        dset = TrajectoryCollection(ts, ys)
        with pytest.raises(ValueError):
            DerivativeCollection.from_trajectory_collection(dset)
        y_ts = jr.normal(getkey(), (2, 3, 2))
        us = jr.normal(getkey(), (2, 3, 1))
        dset = TrajectoryCollection(ts, ys, y_ts=y_ts, us=us)
        deriv = DerivativeCollection.from_trajectory_collection(dset)
        assert deriv.ts.shape == (6,)

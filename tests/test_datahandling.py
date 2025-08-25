import jax.numpy as jnp
import numpy as np
import pytest

from dynax._datahandling import (
    Trajectory,
    TrajectoryCollection,
    _as_nonsqueezing_index,
)


@pytest.mark.parametrize(
    "index, expected_shape",
    [
        # Single integer index (should add axis)
        (0, (1, 4, 5)),
        # Single slice (should not change shape)
        (slice(1, 3), (2, 4, 5)),
        # Single boolean array
        (np.array([True, False, True]), (2, 4, 5)),
        # Single 0-dim array (should add axis)
        (np.array(0), (1, 4, 5)),
        (jnp.array(0), (1, 4, 5)),
        # Tuple: int, slice, int
        ((1, slice(2, 4), 0), (1, 2, 1)),
        # Tuple: 0-dim array, slice, int
        ((np.array(0), slice(2, 4), 0), (1, 2, 1)),
        # Tuple: int, int, int
        ((1, 2, 0), (1, 1, 1)),
        # Tuple: slice, slice, slice
        ((slice(0, 2), slice(1, 3), slice(0, 2)), (2, 2, 2)),
        # Tuple: int, boolean array, int
        ((0, np.array([True, False, True, False]), 0), (1, 2, 1)),
        # Tuple: 0-dim jax array, numpy array, int
        ((jnp.array(0), np.array([1, 2]), 0), (1, 2, 1)),
        # Empty tuple:
        ((), (3, 4, 5)),
    ],
)
def test_as_nonsqueezing_index(index, expected_shape):
    arr = np.zeros((3, 4, 5))
    idx = _as_nonsqueezing_index(index)
    result = arr[idx]
    assert result.shape == expected_shape


class TestTrajectory:
    def test_basic_init(self):
        ts = np.linspace(0, 1, 5)
        ys = np.random.randn(5, 3)
        traj = Trajectory(ts=ts, ys=ys)
        assert np.allclose(traj.ts, ts)
        assert np.allclose(traj.ys, ys)
        assert traj._y_ts is None
        assert traj._us is None

    def test_optional_fields(self):
        ts = np.linspace(0, 1, 4)
        ys = np.random.randn(4, 2)
        y_ts = np.random.randn(4, 2)
        us = np.random.randn(4, 1)
        traj = Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)
        assert np.allclose(traj.y_ts, y_ts)
        assert np.allclose(traj.us, us)

    @pytest.mark.parametrize(
        "ts, ys, y_ts, us",
        [
            (np.zeros((2, 2)), np.zeros((2, 2)), None, None),
            (np.zeros(3), np.zeros((3,)), None, None),
            (np.zeros(3), np.zeros((3, 2)), np.zeros((3,)), None),
            (np.zeros(3), np.zeros((3, 2)), None, np.zeros((3,))),
            (np.zeros(3), np.zeros((2, 2)), None, None),
            (np.zeros(3), np.zeros((3, 2)), np.zeros((3, 3)), None),
        ],
    )
    def test_check_init_invalid(self, ts, ys, y_ts, us):
        with pytest.raises(ValueError):
            Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)

    def test_getitem(self):
        ts = np.arange(5)
        ys = np.zeros((5, 2))
        y_ts = np.zeros((5, 2))
        us = np.zeros((5, 1))
        traj = Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)
        sub = traj[2]
        assert sub.ts.shape == (1,)
        assert sub.ys.shape == (1, 2)
        assert sub.y_ts.shape == (1, 2)
        assert sub.us.shape == (1, 1)

    def test_setitem(self):
        ts = np.arange(3)
        ys = np.zeros((3, 2))
        traj = Trajectory(ts=ts, ys=ys)
        traj[1] = {"ys": np.ones((1, 2))}
        assert np.allclose(traj.ys[1], 1)


class TestTrajectoryCollection:
    def test_from_trajectories(self):
        ts = np.linspace(0, 1, 4)
        ys1 = np.ones((4, 2))
        ys2 = np.zeros((4, 2))
        t1 = Trajectory(ts=ts, ys=ys1)
        t2 = Trajectory(ts=ts, ys=ys2)
        coll = TrajectoryCollection.from_trajectories([t1, t2])
        assert coll.ys.shape == (2, 4, 2)
        assert np.allclose(coll.ts, ts)

    def test_from_trajectories_mismatch(self):
        ts1 = np.linspace(0, 1, 3)
        ts2 = np.linspace(0, 2, 3)
        t1 = Trajectory(ts=ts1, ys=np.ones((3, 2)))
        t2 = Trajectory(ts=ts2, ys=np.zeros((3, 2)))
        with pytest.raises(
            ValueError,
            match="All trajectories must share the same timestamps.",
        ):
            TrajectoryCollection.from_trajectories([t1, t2])

    def test_check_init_valid(self):
        ts = np.linspace(0, 1, 2)
        ys = np.zeros((3, 2, 2))
        y_ts = np.zeros((3, 2, 2))
        us = np.zeros((3, 2, 1))
        coll = TrajectoryCollection(ts=ts, ys=ys, y_ts=y_ts, us=us)
        coll.__check_init__()  # Should not raise

    @pytest.mark.parametrize(
        "ts,ys,y_ts,us",
        [
            (np.zeros((2, 2)), np.zeros((2, 2)), None, None),  # ts shape
            (np.zeros(3), np.zeros((3, 1)), None, None), # ys shape
            (np.zeros(3), np.zeros((1, 3, 2)), np.zeros((3,)), None), # y_ts shape
            (np.zeros(3), np.zeros((1, 3, 2)), None, np.zeros((3,4,))), # us shape
            (np.zeros(3), np.zeros((1, 2, 2)), None, None), # time doesn't match
            (np.zeros(3), np.zeros((1, 3, 2)), np.zeros((1, 3, 3)), None), # state doesn't match
        ],
    )
    def test_check_init_invalid(self, ts, ys, y_ts, us):
        with pytest.raises(ValueError):
            TrajectoryCollection(ts=ts, ys=ys, y_ts=y_ts, us=us)

    def test_getitem(self):
        ts = np.arange(4)
        ys = np.zeros((10, 4, 2))
        y_ts = np.zeros((10, 4, 2))
        us = np.zeros((10, 4, 3))
        coll = TrajectoryCollection(ts=ts, ys=ys, y_ts=y_ts, us=us)
        sub = coll[1, 2]
        assert np.allclose(sub.ts, ts[2])
        assert sub.ys.shape == (1, 1, 2)
        assert sub.y_ts.shape == (1, 1, 2)
        assert sub.us.shape == (1, 1, 3)

    def test_setitem(self):
        ts = np.arange(2)
        ys = np.zeros((2, 2, 2))
        coll = TrajectoryCollection(ts=ts, ys=ys)
        coll[(1, 1)] = {"ys": np.ones((1, 2))}
        assert np.allclose(coll.ys[1, 1], 1)

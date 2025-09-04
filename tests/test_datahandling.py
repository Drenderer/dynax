import jax
import jax.numpy as jnp
import pytest
from jax import random as jr

from dynax._datahandling import (
    Trajectory,
    differentiate,
)


def test_differentiate():
    ts = jnp.linspace(0, 1, 1000, dtype=jax.numpy.float64)
    ys = (ts, jnp.stack([jnp.sin(ts), jnp.cos(ts)], axis=1))
    y_ts = (jnp.ones_like(ts), jnp.stack([jnp.cos(ts), -jnp.sin(ts)], axis=1))
    y_ts_ = differentiate(ts, ys)
    assert jax.tree.structure(y_ts) == jax.tree.structure(y_ts_)
    assert all(
        jax.tree.map(
            lambda x, y: jnp.allclose(x, y, rtol=1e-3, atol=1e-3), y_ts, y_ts_
        )
    )


class TestTrajectory:
    def test_init_and_fields(self):
        ts = jnp.linspace(0, 1, 5)
        ys = jnp.arange(10).reshape(5, 2)
        y_ts = jnp.arange(10, 20).reshape(5, 2)
        us = jnp.arange(5).reshape(5, 1)
        traj = Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)
        assert traj.ts.shape == (5,)
        assert traj.ys.shape == (5, 2)
        assert traj.y_ts.shape == (5, 2)
        assert traj.us.shape == (5, 1)
        assert jnp.allclose(traj.ts, ts)
        assert jnp.allclose(traj.ys, ys)
        assert jnp.allclose(traj.y_ts, y_ts)
        assert jnp.allclose(traj.us, us)

    def test_init_with_none(self):
        traj = Trajectory()
        assert traj._ts is None
        assert traj._ys is None
        assert traj._y_ts is None
        assert traj._us is None

    def test_dtype(self):
        ts = [0, 1, 2]
        ys = [[1, 2], [3, 4], [5, 6]]
        traj = Trajectory(ts=ts, ys=ys, dtype=jnp.float32)
        assert traj.ts.dtype == jnp.float32
        assert traj.ys.dtype == jnp.float32

    def test_getitem(self):
        ts = jnp.linspace(0, 1, 4)
        ys = jnp.arange(8).reshape(4, 2)
        y_ts = jnp.arange(8, 16).reshape(4, 2)
        us = jnp.arange(4).reshape(4, 1)
        traj = Trajectory(ts=ts, ys=ys, y_ts=y_ts, us=us)
        sub = traj[1]
        assert jnp.allclose(sub.ts, ts[1])
        assert jnp.allclose(sub.ys, ys[1])
        assert jnp.allclose(sub.y_ts, y_ts[1])
        assert jnp.allclose(sub.us, us[1])

    def test_stack(self):
        ts = jnp.linspace(0, 1, 3)
        ys1 = jnp.ones((3, 2))
        ys2 = jnp.zeros((3, 2))
        t1 = Trajectory(ts=ts, ys=ys1)
        t2 = Trajectory(ts=ts, ys=ys2)
        stacked = Trajectory.stack([t1, t2], axis=0)
        assert stacked.ts.shape == (2, 3)
        assert stacked.ys.shape == (2, 3, 2)
        assert jnp.allclose(stacked.ys[0], ys1)
        assert jnp.allclose(stacked.ys[1], ys2)

    def test_stack_with_none_fields(self):
        ts = jnp.linspace(0, 1, 2)
        t1 = Trajectory(ts=ts)
        t2 = Trajectory(ts=ts)
        stacked = Trajectory.stack([t1, t2], axis=0)
        assert stacked.ts.shape == (2, 2)
        assert stacked._ys is None

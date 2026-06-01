import jax.numpy as jnp

from dynax import concat_pytree, slice_pytree


def test_concat_pytree_concatenates_arrays_and_keeps_non_arrays():
    a = {
        "x": jnp.array([1.0, 2.0]),
        "nested": {"y": jnp.array([[1, 2], [3, 4]])},
        "tag": "keep_a",
    }
    b = {
        "x": jnp.array([3.0]),
        "nested": {"y": jnp.array([[5, 6]])},
        "tag": "ignored_b",
    }
    axes = {"x": 0, "nested": 0, "tag": None}

    out = concat_pytree(a, b, axes)

    assert jnp.allclose(out["x"], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(
        out["nested"]["y"], jnp.array([[1, 2], [3, 4], [5, 6]])
    )
    assert out["tag"] == "keep_a"


def test_slice_pytree_slices_arrays_and_keeps_non_arrays():
    a = {
        "x": jnp.array([10, 20, 30, 40]),
        "nested": {"y": jnp.array([[1, 2], [3, 4], [5, 6]])},
        "tag": "keep_a",
    }
    start = {"x": 1, "nested": 0, "tag": None}
    stop = {"x": 3, "nested": {"y": 2}, "tag": None}
    axes = {"x": 0, "nested": 0, "tag": None}

    out = slice_pytree(a, start=start, stop=stop, axes=axes)

    assert jnp.allclose(out["x"], jnp.array([20, 30]))
    assert jnp.allclose(out["nested"]["y"], jnp.array([[1, 2], [3, 4]]))
    assert out["tag"] == "keep_a"

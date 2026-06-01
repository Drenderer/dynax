from typing import Any

import jax
from jax import numpy as jnp
from jaxtyping import Array, PyTree


def concat_pytree(
    a: PyTree[Any, "T"],  # type: ignore
    b: PyTree[Any, "T"],  # type: ignore
    axes: PyTree[int, "T ..."] = 0,  # type: ignore
) -> PyTree[Any, "T"]:  # type: ignore
    """Concatenate arrays from two PyTrees along given axes.

    For each Array leaf, concatenates `b` onto `a` along `axis`,
    producing an augmented state. Non-Array leaves are taken from `a`.

    Args:
        a: Original PyTree.
        b: Second PyTree with the same structure as `a`.
            Each Array leaf will be concatenated onto the corresponding
            leaf of `a`.
        axes: PyTree of concatenation axes, matching the structure of `a`.
            A scalar `0` is broadcast to all leaves.

    Returns:
        Augmented state PyTree with the same structure as `x`, where each
        Array leaf has size `x.shape[axis] + x_aux.shape[axis]` along
        `axis`.

    """
    axes = jax.tree.broadcast(axes, a, is_leaf=lambda x: x is None)
    return jax.tree.map(
        lambda a, b, axis: jnp.concat([a, b], axis=axis)
        if isinstance(a, Array)
        else a,
        a,
        b,
        axes,
    )


def slice_pytree(
    a: PyTree[Any, "T"],  # type: ignore
    start: PyTree[int | None, "T ..."] = 0,  # type: ignore
    stop: PyTree[int | None, "T ..."] = None,  # type: ignore
    axes: PyTree[int, "T ..."] = 0,  # type: ignore
) -> PyTree[Any, "T"]:  # type: ignore
    """Slice arrays in a PyTree along given axes.

    For each Array leaf, slices from `start` to `stop` along `axis`.
    Non-Array leaves are passed through unchanged.

    Args:
        a: Input PyTree.
        start: PyTree of start indices, matching the structure of `a`.
            A scalar `0` is broadcast to all leaves.
        stop: PyTree of stop indices, matching the structure of `a`.
            `None` slices to the end of the axis. A scalar is broadcast
            to all leaves.
        axes: PyTree of axes along which to slice, matching the structure
            of `a`. A scalar `0` is broadcast to all leaves.

    Returns:
        PyTree with the same structure as `a`, where each Array leaf has
        been sliced to `a[..., start:stop, ...]` along `axis`.

    """
    start = jax.tree.broadcast(start, a, is_leaf=lambda x: x is None)
    stop = jax.tree.broadcast(stop, a, is_leaf=lambda x: x is None)
    axes = jax.tree.broadcast(axes, a, is_leaf=lambda x: x is None)
    return jax.tree.map(
        lambda a, start, stop, axis: jax.lax.slice_in_dim(
            a, start, stop, axis=axis
        )
        if isinstance(a, Array)
        else a,
        a,
        start,
        stop,
        axes,
    )

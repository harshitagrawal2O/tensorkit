"""The inverse of NumPy broadcasting -- Milestone 2.

Forward, NumPy stretches a ``(3,)`` operand against a ``(4, 3)`` one for free. Backward, a
gradient arriving with shape ``(4, 3)`` has to fold back to ``(3,)``, and the fold is a **sum**,
not a mean: broadcasting duplicated the value, so each copy contributes its own gradient.

Get this wrong and gradients are off by exactly the broadcast factor. Nothing errors. The model
trains, just badly. That is why it has its own module and its own property tests.

Concepts: ``docs/concepts/broadcasting-backward.md``.
Tests: ``tests/test_broadcasting.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["unbroadcast", "broadcast_shapes_or_raise"]


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape``, undoing whatever broadcasting the forward pass did.

    Args:
        grad: The incoming gradient, shaped as the broadcast result.
        shape: The original operand shape the gradient must be folded back to.

    Returns:
        An array of exactly ``shape``.

    Algorithm (two distinct cases -- both are needed, and skipping either is a real bug):
      1. **Rank promotion.** NumPy prepends length-1 axes to the lower-rank operand. Sum over
         the leading ``grad.ndim - len(shape)`` axes to drop them entirely.
      2. **Size-1 stretching.** For each remaining axis where the original extent was 1 and the
         broadcast extent is greater than 1, sum with ``keepdims=True`` so the axis survives
         with extent 1.

    Invariants (``SPEC.md`` section 3.2):
        I-UB-SHAPE: the result's shape equals ``shape``, exactly, always.
        I-UB-SUMPRESERVE: ``result.sum() == grad.sum()`` to floating tolerance. The gradient of
            a duplicated value is the *sum* over its copies. If you ever reach for ``mean``
            here, stop and re-derive it.

    Complexity: O(size of grad).

    Both steps are plain sums, which is the whole content of the adjoint identity
    ``<broadcast(x), g> == <x, unbroadcast(g)>``: broadcasting is a linear map that copies, and
    the adjoint of a copy is an add. ``tests/test_broadcasting.py`` asserts that identity
    directly, which is a stronger statement than sum preservation and implies it.

    Tests: tests/test_broadcasting.py::test_unbroadcast_shape_property
    """
    if grad.shape == shape:
        return grad

    promoted = grad.ndim - len(shape)
    if promoted < 0:
        raise ValueError(
            f"cannot unbroadcast a gradient of shape {grad.shape} to {shape}: the target has "
            f"more axes than the gradient, so NumPy could never have broadcast it that way"
        )

    # 1. Rank promotion: the leading axes were inserted, so they sum away entirely.
    if promoted:
        grad = grad.sum(axis=tuple(range(promoted)))

    # 2. Size-1 stretching: keepdims so the axis survives with extent 1, as the operand had it.
    stretched = tuple(
        axis
        for axis, (got, want) in enumerate(zip(grad.shape, shape, strict=True))
        if want == 1 and got != 1
    )
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)

    if grad.shape != shape:
        raise ValueError(
            f"unbroadcast produced shape {grad.shape}, not the requested {shape}. The operand "
            f"and the gradient were never broadcast-compatible (I-UB-SHAPE)."
        )
    return grad


def broadcast_shapes_or_raise(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Return the broadcast shape of ``a`` and ``b``, or raise with a readable message.

    ``np.broadcast_shapes`` already does this. The reason to wrap it is the error message: a
    shape mismatch deep in a network is far easier to debug when the exception names both
    operand shapes and the operation, rather than surfacing as a NumPy error three frames down.

    Raises:
        ValueError: if the shapes are not broadcast-compatible.

    Tests: tests/test_broadcasting.py::test_incompatible_shapes_raise_readably
    """
    try:
        return tuple(np.broadcast_shapes(a, b))
    except ValueError as exc:
        raise ValueError(
            f"operands could not be broadcast together: shapes {a} and {b}. NumPy aligns "
            f"shapes from the right; every pair of extents must be equal or one of them 1."
        ) from exc

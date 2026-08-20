"""Differentiable primitives and their local gradient rules -- Milestone 2.

Every gradient rule in TensorKit lives here. That is deliberate: ``gradcheck`` sweeps
:data:`PRIMITIVES` and asserts each entry against numerical differentiation, so a rule that is
not in this registry is a rule nobody checked.

The mental model is the **vector-Jacobian product**. A primitive never builds its Jacobian --
for ``matmul`` on ``(512, 512)`` operands that would be a 68-billion-element object. It answers
one question instead: given ``v = d(loss)/d(output)``, what is ``v @ J``? For every primitive
here that product has a closed form costing about as much as the forward pass, which is the
entire reason reverse mode is affordable.

Concepts: ``docs/concepts/autodiff.md``, ``docs/concepts/vjp.md``.
Tests: ``tests/test_ops.py``, ``tests/test_gradcheck.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from tensorkit.broadcasting import broadcast_shapes_or_raise, unbroadcast

if TYPE_CHECKING:
    from tensorkit.tensor import BackwardFn, Tensor

__all__ = [
    "Primitive",
    "PRIMITIVES",
    "register",
    "add",
    "sub",
    "mul",
    "div",
    "power",
    "neg",
    "matmul",
    "sum_",
    "mean",
    "max_",
    "exp",
    "log",
    "sqrt",
    "tanh",
    "relu",
    "abs_",
    "reshape",
    "transpose",
    "getitem",
    "concat",
    "softmax",
    "log_softmax",
    "gelu",
]


class Primitive(NamedTuple):
    """A differentiable operation registered for the gradcheck sweep.

    Attributes:
        name: Operator name as it appears in ``Tensor._op``.
        forward: The forward function over Tensors.
        arity: Number of Tensor inputs.
        kink_free: False for operations with a non-differentiable point (relu, abs, max).
            Gradcheck perturbs these away from the kink and says so in the failure message,
            rather than reporting a spurious mismatch at a point where no gradient exists.
        milestone: Which milestone introduces it, so ``make test-tensorkit-mN`` can filter.
    """

    name: str
    forward: Callable[..., Tensor]
    arity: int
    kink_free: bool
    milestone: int


#: Every registered primitive, keyed by name. Populated by :func:`register`.
PRIMITIVES: dict[str, Primitive] = {}


def register(
    name: str, *, arity: int, kink_free: bool = True, milestone: int = 2
) -> Callable[[Callable[..., Tensor]], Callable[..., Tensor]]:
    """Decorator adding a primitive to :data:`PRIMITIVES`.

    GIVEN -- registration plumbing, not an algorithm. Decorate every new differentiable op
    with it; ``tests/test_gradcheck.py::test_every_primitive_is_checked`` fails if a public
    op in this module is missing from the registry, which is what stops the sweep quietly
    shrinking as the library grows.
    """

    def decorator(fn: Callable[..., Tensor]) -> Callable[..., Tensor]:
        PRIMITIVES[name] = Primitive(name, fn, arity, kink_free, milestone)
        return fn

    return decorator


def _node(
    data: np.ndarray,
    parents: tuple[Tensor, ...],
    op: str,
    make_backward: Callable[[Tensor], BackwardFn],
) -> Tensor:
    """Wrap a forward result as a tape node.

    The import is deferred because :mod:`tensorkit.tensor` imports this module at module scope:
    the rules live here, the node type lives there, and one of the two edges has to be lazy.
    Doing it in exactly one function keeps the rest of the module free of import bookkeeping.
    """
    from tensorkit.tensor import Tensor

    return Tensor.from_op(data, parents, op, make_backward)


def _reduced_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    """Return the axes a reduction collapses, normalised to non-negative and sorted.

    ``None`` means every axis, which is what makes ``sum()`` and ``sum(axis=(0, 1, ...))``
    share one backward rule instead of two.
    """
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(sorted(ax % ndim for ax in axes))


def _restore_reduced(grad: np.ndarray, axes: tuple[int, ...], keepdims: bool) -> np.ndarray:
    """Put the axes a reduction removed back, so the gradient can broadcast over them again.

    With ``keepdims=True`` they were never removed and this is the identity. With
    ``keepdims=False`` the gradient arrives one rank per reduced axis too small, and
    broadcasting it without reinserting them aligns the *wrong* axes -- silently, whenever the
    remaining extents happen to be compatible. This is the single most common shape bug in a
    hand-rolled engine, which is why it is one named function used by every reduction.
    """
    if keepdims or not axes:
        return grad
    return np.expand_dims(grad, axes)


# ---------------------------------------------------------------------------
# Binary elementwise ops
# ---------------------------------------------------------------------------


@register("add", arity=2)
def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise ``a + b`` with NumPy broadcasting.

    VJP: ``da = unbroadcast(g, a.shape)``, ``db = unbroadcast(g, b.shape)``.

    The local derivative is 1 for both operands, so all the work is the unbroadcast. If both
    shapes are already equal this is the identity and the bug hides; the tests deliberately
    use ragged shapes.

    Tests: tests/test_ops.py::test_add_broadcast_backward
    """
    broadcast_shapes_or_raise(a.shape, b.shape)
    data = a.data + b.data

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None:
                return
            if a.requires_grad:
                a.accumulate_grad(unbroadcast(g, a.shape))
            if b.requires_grad:
                b.accumulate_grad(unbroadcast(g, b.shape))

        return _backward

    return _node(data, (a, b), "add", _rule)


@register("sub", arity=2)
def sub(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise ``a - b``.

    VJP: ``da = unbroadcast(g, a.shape)``, ``db = unbroadcast(-g, b.shape)``.

    The right operand's gradient is negated *and* unbroadcast. Doing only one of the two is a
    sign error or a shape error respectively -- and the sign error is the quiet one.
    """
    broadcast_shapes_or_raise(a.shape, b.shape)
    data = a.data - b.data

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None:
                return
            if a.requires_grad:
                a.accumulate_grad(unbroadcast(g, a.shape))
            if b.requires_grad:
                b.accumulate_grad(unbroadcast(-g, b.shape))

        return _backward

    return _node(data, (a, b), "sub", _rule)


@register("mul", arity=2)
def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise ``a * b``.

    VJP: ``da = unbroadcast(g * b, a.shape)``, ``db = unbroadcast(g * a, b.shape)``.

    Multiply **before** unbroadcasting. Unbroadcasting first reduces the gradient to the
    operand's shape and then multiplies by a differently-shaped operand, which either raises
    or -- worse -- broadcasts back to something plausible and wrong.

    Tests: tests/test_ops.py::test_mul_broadcast_backward
    """
    broadcast_shapes_or_raise(a.shape, b.shape)
    data = a.data * b.data

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None:
                return
            if a.requires_grad:
                a.accumulate_grad(unbroadcast(g * b.data, a.shape))
            if b.requires_grad:
                b.accumulate_grad(unbroadcast(g * a.data, b.shape))

        return _backward

    return _node(data, (a, b), "mul", _rule)


@register("div", arity=2)
def div(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise ``a / b``.

    VJP: ``da = unbroadcast(g / b, a.shape)``, ``db = unbroadcast(-g * a / b**2, b.shape)``.

    Two terms. Dropping the second is the planted bug in
    ``tests/test_gradcheck.py::test_gradcheck_catches_a_planted_bug``, and it is a good bug
    to have seen: the forward pass is fine, the network still trains, and the denominator's
    gradient is simply absent.

    Tests: tests/test_ops.py::test_div_backward_has_both_terms
    """
    broadcast_shapes_or_raise(a.shape, b.shape)
    data = a.data / b.data

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None:
                return
            if a.requires_grad:
                a.accumulate_grad(unbroadcast(g / b.data, a.shape))
            if b.requires_grad:
                b.accumulate_grad(unbroadcast(-g * a.data / (b.data * b.data), b.shape))

        return _backward

    return _node(data, (a, b), "div", _rule)


@register("pow", arity=1)
def power(a: Tensor, exponent: float) -> Tensor:
    """Elementwise ``a ** exponent`` for constant ``exponent``.

    VJP: ``da = g * exponent * a ** (exponent - 1)``.

    A Tensor exponent would need the ``a ** x * ln a`` term and a positivity domain check, so
    it is rejected rather than approximated -- see :meth:`tensorkit.scalar.Value.__pow__`.
    """
    try:
        power_of = float(exponent)
    except TypeError as exc:
        raise TypeError(
            f"the exponent of Tensor ** exponent must be a constant int or float, not "
            f"{type(exponent).__name__}: a Tensor exponent needs the a**x * ln(a) term, "
            f"which no rule here implements."
        ) from exc

    data = a.data**power_of

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g * power_of * a.data ** (power_of - 1.0))

        return _backward

    return _node(data, (a,), "pow", _rule)


@register("neg", arity=1)
def neg(a: Tensor) -> Tensor:
    """Elementwise ``-a``. VJP: ``da = -g``."""
    data = -a.data

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(-g)

        return _backward

    return _node(data, (a,), "neg", _rule)


@register("matmul", arity=2)
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiply, with batching over the leading dimensions.

    VJP: ``da = g @ b.swapaxes(-1, -2)``, ``db = a.swapaxes(-1, -2) @ g``.

    Derive it once from ``C[i,j] = sum_k A[i,k] B[k,j]`` and it stops being something to
    memorise. Two things are easy to get wrong:

    * **Batch dimensions.** ``(B, H, T, d) @ (B, H, d, T)`` broadcasts over ``(B, H)``. If an
      operand had a batch extent of 1 that got stretched, its gradient must be unbroadcast
      back. NanoLM's attention hits this on day one.
    * **Vector promotion.** NumPy promotes a 1-D operand to a matrix and then removes the
      added axis from the result. The backward pass has to undo that promotion, not just
      transpose.

    Complexity: forward O(B * n * m * p); backward is two more matmuls of the same order --
    the "backward costs about 2x forward" rule of thumb, from here.

    Implementation: promote both operands to at least 2-D and insert the matching axis into
    the incoming gradient, so one rule covers vector-matrix, matrix-vector, matrix-matrix and
    every batched shape. The promoted gradient is unbroadcast to the *promoted* operand shape
    (that undoes any stretched batch axis) and only then reshaped back to the operand's own
    shape (that undoes the promotion).

    Tests: tests/test_ops.py::test_matmul_batched_backward
    """
    try:
        data = a.data @ b.data
    except ValueError as exc:
        raise ValueError(
            f"matmul: shapes {a.shape} and {b.shape} do not line up. The last axis of the "
            f"left operand must match the second-to-last of the right one, and any leading "
            f"batch axes must broadcast."
        ) from exc

    a_is_vector = a.data.ndim == 1
    b_is_vector = b.data.ndim == 1

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None:
                return
            lhs = a.data[np.newaxis, :] if a_is_vector else a.data
            rhs = b.data[:, np.newaxis] if b_is_vector else b.data
            # Right operand first: for a 1-D @ 1-D product the gradient is 0-d, and inserting
            # the row axis at -2 is only meaningful once the column axis exists.
            if b_is_vector:
                g = np.expand_dims(g, -1)
            if a_is_vector:
                g = np.expand_dims(g, -2)
            if a.requires_grad:
                da = g @ np.swapaxes(rhs, -1, -2)
                a.accumulate_grad(unbroadcast(da, lhs.shape).reshape(a.shape))
            if b.requires_grad:
                db = np.swapaxes(lhs, -1, -2) @ g
                b.accumulate_grad(unbroadcast(db, rhs.shape).reshape(b.shape))

        return _backward

    return _node(data, (a, b), "matmul", _rule)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@register("sum", arity=1)
def sum_(a: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
    """Sum over ``axis``.

    VJP: broadcast ``g`` back over the reduced axes.

    With ``keepdims=False`` the reduced axis is gone from ``g``, so it must be reinserted
    (``np.expand_dims``) before broadcasting. Sum and broadcast are duals: the gradient of a
    sum is a broadcast, and the gradient of a broadcast is a sum. If that symmetry is clear,
    :mod:`tensorkit.broadcasting` stops feeling like a special case.

    Tests: tests/test_ops.py::test_sum_backward_keepdims_false
    """
    # The keepdims branch is spelled out rather than passed through: NumPy's stubs type the
    # return differently for each literal, and a plain bool matches no overload.
    total = np.sum(a.data, axis=axis, keepdims=True) if keepdims else np.sum(a.data, axis=axis)
    data = np.asarray(total)
    axes = _reduced_axes(axis, a.data.ndim)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(np.broadcast_to(_restore_reduced(g, axes, keepdims), a.shape))

        return _backward

    return _node(data, (a,), "sum", _rule)


@register("mean", arity=1)
def mean(a: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
    """Mean over ``axis``. VJP: ``sum``'s, scaled by ``1 / n_reduced``.

    ``n_reduced`` is the product of the collapsed extents, computed from the shape rather than
    from ``a.size // out.size`` so that a zero-length axis cannot turn into a division by zero.
    """
    average = np.mean(a.data, axis=axis, keepdims=True) if keepdims else np.mean(a.data, axis=axis)
    data = np.asarray(average)
    axes = _reduced_axes(axis, a.data.ndim)
    count = 1
    for ax in axes:
        count *= a.shape[ax]

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            spread = np.broadcast_to(_restore_reduced(g, axes, keepdims), a.shape)
            a.accumulate_grad(spread / count)

        return _backward

    return _node(data, (a,), "mean", _rule)


@register("max", arity=1, kink_free=False)
def max_(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Maximum over ``axis``.

    VJP: route ``g`` to the argmax position, zero elsewhere. Ties go to the first occurrence --
    a documented choice. Splitting the gradient across tied positions is also defensible; what
    is not defensible is routing the full gradient to *every* tied position, which multiplies
    the gradient by the tie count.

    ``np.argmax`` returns the first maximum, so building the routing mask from it -- rather
    than from ``a.data == out.data``, which marks every tie -- is what implements the choice.
    """
    peak = np.max(a.data, axis=axis, keepdims=True) if keepdims else np.max(a.data, axis=axis)
    data = np.asarray(peak)
    axes = _reduced_axes(axis, a.data.ndim)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            mask = np.zeros(a.shape, dtype=a.dtype)
            if axis is None:
                mask[np.unravel_index(int(np.argmax(a.data)), a.shape)] = 1.0
            else:
                winners = np.expand_dims(np.argmax(a.data, axis=axis), axis)
                np.put_along_axis(mask, winners, 1.0, axis)
            spread = np.broadcast_to(_restore_reduced(g, axes, keepdims), a.shape)
            a.accumulate_grad(mask * spread)

        return _backward

    return _node(data, (a,), "max", _rule)


# ---------------------------------------------------------------------------
# Unary elementwise ops
# ---------------------------------------------------------------------------


@register("exp", arity=1)
def exp(a: Tensor) -> Tensor:
    """Elementwise ``e ** a``. VJP: ``da = g * out`` -- the output is its own derivative."""
    data = np.exp(a.data)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g * out.data)

        return _backward

    return _node(data, (a,), "exp", _rule)


@register("log", arity=1)
def log(a: Tensor) -> Tensor:
    """Elementwise natural log. VJP: ``da = g / a``.

    Non-positive input raises instead of returning ``-inf``/``nan``. NumPy would only emit a
    RuntimeWarning and hand back nan, which then contaminates every gradient downstream and
    surfaces as a dead network several layers from the actual mistake.
    """
    if bool(np.any(a.data <= 0)):
        offenders = int(np.count_nonzero(a.data <= 0))
        raise ValueError(
            f"log() is undefined for non-positive values: {offenders} of {a.data.size} "
            f"elements are <= 0 (minimum {float(np.min(a.data))!r}). The domain is x > 0; "
            f"returning nan here would poison the whole backward pass silently."
        )
    data = np.log(a.data)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g / a.data)

        return _backward

    return _node(data, (a,), "log", _rule)


@register("sqrt", arity=1)
def sqrt(a: Tensor) -> Tensor:
    """Elementwise square root. VJP: ``da = g * 0.5 / out``.

    Negative input raises. Zero is allowed but the gradient there is infinite -- the function
    is continuous at 0 and its derivative is not, which is why every caller that can reach 0
    (variance in a normalisation layer, for one) adds an epsilon first.
    """
    if bool(np.any(a.data < 0)):
        offenders = int(np.count_nonzero(a.data < 0))
        raise ValueError(
            f"sqrt() is undefined for negative values: {offenders} of {a.data.size} elements "
            f"are < 0 (minimum {float(np.min(a.data))!r})."
        )
    data = np.sqrt(a.data)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g * 0.5 / out.data)

        return _backward

    return _node(data, (a,), "sqrt", _rule)


@register("tanh", arity=1)
def tanh(a: Tensor) -> Tensor:
    """Hyperbolic tangent, elementwise. VJP: ``da = g * (1 - out ** 2)``.

    Uses the output rather than recomputing: the identity is exact and the value is to hand.
    """
    data = np.tanh(a.data)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g * (1.0 - out.data * out.data))

        return _backward

    return _node(data, (a,), "tanh", _rule)


@register("relu", arity=1, kink_free=False)
def relu(a: Tensor) -> Tensor:
    """Elementwise ``max(a, 0)``. VJP: ``da = g * (a > 0)``.

    The subgradient at exactly 0 is taken to be 0, matching :meth:`Value.relu` and PyTorch.
    Gradcheck probes this op away from the kink because a central difference straddling 0
    measures the *average* of the two one-sided slopes, which is 0.5 and matches nothing.
    """
    data = np.maximum(a.data, 0.0)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g * (a.data > 0.0))

        return _backward

    return _node(data, (a,), "relu", _rule)


@register("abs", arity=1, kink_free=False)
def abs_(a: Tensor) -> Tensor:
    """Elementwise absolute value. VJP: ``da = g * sign(a)``.

    ``np.sign(0)`` is 0, which is the same subgradient convention ``relu`` uses at its kink.
    """
    data = np.abs(a.data)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g * np.sign(a.data))

        return _backward

    return _node(data, (a,), "abs", _rule)


# ---------------------------------------------------------------------------
# Shape ops -- no arithmetic, but every one of them still needs a rule
# ---------------------------------------------------------------------------


@register("reshape", arity=1)
def reshape(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape ``a``. VJP: reshape the gradient back to the input's shape.

    A reshape moves no data, so its Jacobian is a permutation of the identity and the VJP is
    the inverse reshape. ``-1`` is resolved by NumPy in the forward pass; the backward pass
    uses ``a.shape`` and so never has to resolve it again.
    """
    data = a.data.reshape(shape)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(g.reshape(a.shape))

        return _backward

    return _node(data, (a,), "reshape", _rule)


@register("transpose", arity=1)
def transpose(a: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
    """Permute the axes of ``a``. VJP: apply the **inverse** permutation to the gradient.

    The inverse is ``argsort(axes)``, not ``axes`` itself. The two coincide for a 2-D swap and
    for any self-inverse permutation, which is exactly why using the wrong one passes the 2-D
    test and then quietly transposes the gradient the wrong way for 3-D and above.
    """
    perm = tuple(reversed(range(a.data.ndim))) if axes is None else tuple(axes)
    data = np.transpose(a.data, perm)
    inverse = tuple(int(i) for i in np.argsort(perm))

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            a.accumulate_grad(np.transpose(g, inverse))

        return _backward

    return _node(data, (a,), "transpose", _rule)


@register("getitem", arity=1)
def getitem(a: Tensor, index: Any) -> Tensor:
    """Index or slice ``a``. VJP: scatter the gradient into zeros at the same index.

    ``np.add.at``, never ``grad[index] = g``. Fancy indexing may name the same element twice
    and plain assignment keeps only the last write, so a repeated index loses every
    contribution but one -- the embedding-table bug, where a token appearing twice in a batch
    learns from one occurrence.

    The forward result is copied rather than returned as a view: a view would alias the input's
    buffer, and two tape nodes sharing storage make in-place mutation undetectable.
    """
    data = np.array(a.data[index], copy=True)

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None or not a.requires_grad:
                return
            scatter = np.zeros(a.shape, dtype=a.dtype)
            np.add.at(scatter, index, g)
            a.accumulate_grad(scatter)

        return _backward

    return _node(data, (a,), "getitem", _rule)


@register("concat", arity=2)
def concat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Concatenate along ``axis``. VJP: split the gradient at the same offsets.

    Concatenation is a linear map that copies each operand into a disjoint slice, so its
    adjoint reads each operand's slice straight back out. Accumulating (rather than assigning)
    into each operand's gradient matters when the same tensor is concatenated with itself.

    ``arity`` is recorded as 2 because that is what the gradcheck case exercises; the operation
    itself takes any number of operands.
    """
    if not tensors:
        raise ValueError("concat() needs at least one tensor; got an empty sequence")

    parents = tuple(tensors)
    data = np.concatenate([t.data for t in parents], axis=axis)
    offsets = np.cumsum([t.shape[axis] for t in parents])[:-1]

    def _rule(out: Tensor) -> BackwardFn:
        def _backward() -> None:
            g = out.grad
            if g is None:
                return
            for tensor, piece in zip(parents, np.split(g, offsets, axis=axis), strict=True):
                if tensor.requires_grad:
                    tensor.accumulate_grad(piece)

        return _backward

    return _node(data, parents, "concat", _rule)


# ---------------------------------------------------------------------------
# Composite ops -- built from primitives, but worth their own gradient rule
# ---------------------------------------------------------------------------


@register("softmax", arity=1, milestone=4)
def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax over ``axis``.

    Forward: subtract the per-slice max before exponentiating. ``exp(1000)`` overflows to
    ``inf`` and ``inf / inf`` is ``nan``; subtracting the max is mathematically the identity
    (the constant cancels) and keeps every exponent at most 0.

    VJP: with ``s`` the output, ``da = s * (g - sum(g * s, axis, keepdims=True))``. The full
    Jacobian ``diag(s) - s s^T`` is never materialised -- for a 50k-token vocabulary it would
    be 2.5 billion entries per row. This closed form is why softmax is affordable.

    Tests: tests/test_ops.py::test_softmax_stability, ::test_softmax_backward
    """
    raise NotImplementedError("Milestone 4")


@register("log_softmax", arity=1, milestone=6)
def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    """``log(softmax(a))``, computed as ``a - logsumexp(a)``.

    Never as ``log(softmax(a))``: softmax underflows to exactly 0 for sufficiently negative
    logits and ``log(0)`` is ``-inf``, which propagates ``nan`` gradients through the whole
    batch. The fused form has no such point.

    VJP: ``da = g - softmax(a) * sum(g, axis, keepdims=True)``.

    Tests: tests/test_ops.py::test_log_softmax_extreme_logits
    """
    raise NotImplementedError("Milestone 6")


@register("gelu", arity=1, milestone=4)
def gelu(a: Tensor, approximate: bool = True) -> Tensor:
    """Gaussian Error Linear Unit.

    Two forms, both required: the exact ``x * 0.5 * (1 + erf(x / sqrt(2)))`` and the tanh
    approximation ``0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))``. A test asserts they agree
    to 1e-3 -- which is the interesting number, because it says the approximation is worth
    using and quantifies what it costs.

    Note ``np.math.erf`` does not exist; ``math.erf`` is scalar-only. Vectorise with
    ``np.vectorize`` for the reference and use the tanh form on the hot path.
    """
    raise NotImplementedError("Milestone 4")


def _dtype_guard(x: np.ndarray) -> np.dtype[np.floating[Any]]:
    """Reject integer arrays at the tape boundary.

    An integer tensor produces an integer gradient, which truncates to zero and looks exactly
    like a vanishing gradient. Fail loudly at construction instead.

    Bools and object arrays are rejected for the same reason. Complex is rejected too: reverse
    mode over complex numbers needs the Wirtinger calculus, and silently treating it as two
    reals gives conjugated gradients.
    """
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(
            f"tensors must hold a float dtype, got {x.dtype}. An integer or boolean tensor "
            f"produces an integer gradient, which truncates every update below 1 to zero -- "
            f"indistinguishable from a vanishing gradient, and it fails at the far end of the "
            f"network from the mistake. Cast with .astype(np.float32) or .astype(np.float64) "
            f"if the values really are meant to be differentiable."
        )
    return cast("np.dtype[np.floating[Any]]", x.dtype)

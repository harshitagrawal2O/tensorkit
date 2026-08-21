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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from tensorkit.tensor import Tensor

__all__ = ["Primitive", "PRIMITIVES", "register", "add", "mul", "div", "matmul", "sum_", "softmax"]


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
    raise NotImplementedError("Milestone 2")


@register("mul", arity=2)
def mul(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise ``a * b``.

    VJP: ``da = unbroadcast(g * b, a.shape)``, ``db = unbroadcast(g * a, b.shape)``.

    Multiply **before** unbroadcasting. Unbroadcasting first reduces the gradient to the
    operand's shape and then multiplies by a differently-shaped operand, which either raises
    or -- worse -- broadcasts back to something plausible and wrong.

    Tests: tests/test_ops.py::test_mul_broadcast_backward
    """
    raise NotImplementedError("Milestone 2")


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
    raise NotImplementedError("Milestone 2")


@register("pow", arity=1)
def power(a: Tensor, exponent: float) -> Tensor:
    """Elementwise ``a ** exponent`` for constant ``exponent``.

    VJP: ``da = g * exponent * a ** (exponent - 1)``.
    """
    raise NotImplementedError("Milestone 2")


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

    Tests: tests/test_ops.py::test_matmul_batched_backward
    """
    raise NotImplementedError("Milestone 2")


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
    raise NotImplementedError("Milestone 2")


@register("mean", arity=1)
def mean(a: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
    """Mean over ``axis``. VJP: ``sum``'s, scaled by ``1 / n_reduced``."""
    raise NotImplementedError("Milestone 2")


@register("max", arity=1, kink_free=False)
def max_(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Maximum over ``axis``.

    VJP: route ``g`` to the argmax position, zero elsewhere. Ties go to the first occurrence --
    a documented choice. Splitting the gradient across tied positions is also defensible; what
    is not defensible is routing the full gradient to *every* tied position, which multiplies
    the gradient by the tie count.
    """
    raise NotImplementedError("Milestone 2")


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
    """
    raise NotImplementedError("Milestone 2")

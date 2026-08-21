"""The tensor tape node -- Milestone 2.

``Tensor`` is ``Value`` with three complications, and all three are where the bugs live:

1. **Shape.** Every gradient must match its value's shape exactly, including for 0-d arrays.
2. **Broadcasting.** The forward pass stretches operands for free; the backward pass has to
   fold the gradient back down, and the fold is a sum (see :mod:`tensorkit.broadcasting`).
3. **Memory.** Retaining every intermediate gradient makes peak memory O(activations). Only
   leaves keep theirs.

Concepts: ``docs/concepts/autodiff.md``, ``docs/concepts/broadcasting-backward.md``.
Tests: ``tests/test_tensor_autograd.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["Tensor"]

ArrayLike = np.ndarray | list[Any] | tuple[Any, ...] | float | int

#: ``_backward`` closures are stored per instance; this alias documents their type.
BackwardFn = Callable[[], None]


class Tensor:
    """An n-dimensional tape node.

    Attributes:
        data: Dense ``float32`` or ``float64``, C-contiguous.
        grad: Same shape and dtype as ``data``, or None if no gradient has arrived yet.
        requires_grad: Whether this tensor participates in the backward pass.
        _prev: The tensors this one was produced from. Empty for a leaf.
        _backward: Closure reading ``self.grad`` and accumulating into ``_prev`` gradients.
        _op: Operator name, for graph rendering and error messages.
        _retain_grad: Whether a non-leaf should keep its gradient after the backward pass.

    Invariants (``SPEC.md`` section 3.1): I-SHAPE, I-LEAF, I-ACCUM, I-ACYCLIC, I-ONCE, I-READY,
    I-SEED, I-NOGRAD. Each has a test named after it.
    """

    __slots__ = ("data", "grad", "requires_grad", "_prev", "_backward", "_op", "_retain_grad")

    # Bare annotations, not assignments: they document the types for mypy without creating
    # class attributes, which __slots__ would reject.
    data: np.ndarray
    grad: np.ndarray | None
    requires_grad: bool
    _prev: tuple[Tensor, ...]
    _backward: BackwardFn
    _op: str
    _retain_grad: bool

    def __init__(
        self,
        data: ArrayLike,
        *,
        requires_grad: bool = False,
        dtype: np.dtype[Any] | type[Any] | None = None,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        """Wrap ``data`` as a tape node.

        ``dtype`` defaults to ``float32``. Gradient checking runs in ``float64`` because
        central differences in ``float32`` have error floors around 1e-3 -- far above any
        tolerance worth asserting (see ``docs/concepts/gradcheck.md``).
        """
        raise NotImplementedError("Milestone 2")

    # -- introspection -------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        raise NotImplementedError("Milestone 2")

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        raise NotImplementedError("Milestone 2")

    @property
    def dtype(self) -> np.dtype[Any]:
        """Element dtype."""
        raise NotImplementedError("Milestone 2")

    @property
    def is_leaf(self) -> bool:
        """True when this tensor was not produced by a recorded operation (``_prev == ()``)."""
        raise NotImplementedError("Milestone 2")

    def __repr__(self) -> str:
        """Show shape, dtype, op, and whether a gradient is present."""
        raise NotImplementedError("Milestone 2")

    # -- gradient plumbing ---------------------------------------------------

    def accumulate_grad(self, incoming: np.ndarray) -> None:
        """Add ``incoming`` into ``self.grad``, allocating on first arrival.

        This is the single place gradients are written, so I-ACCUM and I-SHAPE are enforced
        once rather than in every backward closure.

        Invariants:
            I-ACCUM: ``+=``, never ``=``. A tensor consumed k times receives k contributions
                and their sum is the total derivative.
            I-SHAPE: ``incoming.shape == self.data.shape``. A mismatch means an unbroadcast is
                missing upstream, and the assertion here names the op so you find it fast --
                far better than NumPy broadcasting the mismatch into a plausible wrong answer.

        Tests: tests/test_tensor_autograd.py::test_accumulate_grad_rejects_shape_mismatch
        """
        raise NotImplementedError("Milestone 2")

    def zero_grad(self) -> None:
        """Set ``grad`` to None.

        None rather than ``zeros_like``, so "no gradient yet" and "gradient is exactly zero"
        stay distinguishable -- which is what makes I-ACCUM testable (I-MOD-ZERO).
        """
        raise NotImplementedError("Milestone 2")

    def retain_grad(self) -> None:
        """Keep this non-leaf tensor's gradient after the backward pass. Debugging only."""
        raise NotImplementedError("Milestone 2")

    def backward(self, grad: np.ndarray | float | None = None) -> None:
        """Run the reverse pass from this tensor. Delegates to :func:`tensorkit.autograd.backward`.

        Invariant I-SEED: a non-scalar tensor requires an explicit ``grad``.

        Tests: tests/test_tensor_autograd.py::test_backward_requires_seed_for_non_scalar
        """
        raise NotImplementedError("Milestone 2")

    def detach(self) -> Tensor:
        """Return a new leaf sharing this tensor's data with no history.

        Shares the buffer -- it does not copy. Mutating either view is visible in both, which is
        exactly why in-place mutation of a recorded tensor is forbidden (I-ACYCLIC).
        """
        raise NotImplementedError("Milestone 2")

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying array, detached from the graph."""
        raise NotImplementedError("Milestone 2")

    def item(self) -> float:
        """Return the single element of a size-1 tensor. Raises otherwise."""
        raise NotImplementedError("Milestone 2")

    # -- elementwise ops -----------------------------------------------------
    # Each delegates to tensorkit.ops, which owns the gradient rules. Keeping the rules in one
    # module means gradcheck has one list to sweep and no rule can be quietly forgotten.

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise add, with broadcasting. Backward unbroadcasts to each operand's shape."""
        raise NotImplementedError("Milestone 2")

    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise multiply, with broadcasting."""
        raise NotImplementedError("Milestone 2")

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise subtract, with broadcasting."""
        raise NotImplementedError("Milestone 2")

    def __truediv__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise divide.

        The quotient rule has two terms; dropping the second is the planted bug in
        ``tests/test_gradcheck.py::test_gradcheck_catches_a_planted_bug``.
        """
        raise NotImplementedError("Milestone 2")

    def __pow__(self, exponent: float) -> Tensor:
        """Elementwise power with a constant exponent."""
        raise NotImplementedError("Milestone 2")

    def __neg__(self) -> Tensor:
        """Elementwise negation."""
        raise NotImplementedError("Milestone 2")

    def __radd__(self, other: ArrayLike) -> Tensor:
        """Reflected add."""
        raise NotImplementedError("Milestone 2")

    def __rmul__(self, other: ArrayLike) -> Tensor:
        """Reflected multiply."""
        raise NotImplementedError("Milestone 2")

    def __rsub__(self, other: ArrayLike) -> Tensor:
        """Reflected subtract."""
        raise NotImplementedError("Milestone 2")

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        """Reflected divide."""
        raise NotImplementedError("Milestone 2")

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiply.

        Backward: ``dA = dC @ B.T`` and ``dB = A.T @ dC``. Batched inputs broadcast over the
        leading dimensions, so both results need unbroadcasting back to the operand shapes --
        the case a 2-D-only implementation silently gets wrong once NanoLM's ``(B, H, T, d)``
        attention tensors show up.

        Tests: tests/test_tensor_autograd.py::test_matmul_batched_backward
        """
        raise NotImplementedError("Milestone 2")

    def exp(self) -> Tensor:
        """Elementwise ``e ** x``. Local gradient: the output."""
        raise NotImplementedError("Milestone 2")

    def log(self) -> Tensor:
        """Elementwise natural log. Local gradient: ``1 / x``."""
        raise NotImplementedError("Milestone 2")

    def sqrt(self) -> Tensor:
        """Elementwise square root. Local gradient: ``0.5 / out`` -- singular at 0."""
        raise NotImplementedError("Milestone 2")

    def tanh(self) -> Tensor:
        """Hyperbolic tangent, elementwise."""
        raise NotImplementedError("Milestone 2")

    def relu(self) -> Tensor:
        """Elementwise ReLU."""
        raise NotImplementedError("Milestone 2")

    def abs(self) -> Tensor:
        """Elementwise absolute value. Kinked at 0; gradcheck must probe away from it."""
        raise NotImplementedError("Milestone 2")

    # -- reductions ----------------------------------------------------------

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Sum over ``axis``.

        Backward broadcasts the incoming gradient back over the reduced axes. With
        ``keepdims=False`` the axis has been *removed*, so the gradient must be reshaped to
        reinsert it before broadcasting. Forgetting that reinsertion is the most common shape
        bug in a hand-rolled engine, and it produces an exception rather than a wrong number
        only about half the time.

        Tests: tests/test_tensor_autograd.py::test_sum_backward_reinserts_reduced_axis
        """
        raise NotImplementedError("Milestone 2")

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Mean over ``axis``. Backward is ``sum``'s divided by the number of elements reduced."""
        raise NotImplementedError("Milestone 2")

    def max(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Maximum over ``axis``.

        Backward routes the whole gradient to the argmax position and zero elsewhere. Ties are
        broken deterministically (first occurrence) -- a documented choice, since routing to
        every tied position instead scales the gradient by the tie count.
        """
        raise NotImplementedError("Milestone 2")

    # -- shape ops -----------------------------------------------------------

    def reshape(self, *shape: int) -> Tensor:
        """Reshape. Backward reshapes the gradient back to the input's shape."""
        raise NotImplementedError("Milestone 2")

    def transpose(self, *axes: int) -> Tensor:
        """Permute axes.

        Backward applies the **inverse** permutation -- not the same permutation, which happens
        to coincide only for a 2-D swap and diverges silently for 3-D and above.
        """
        raise NotImplementedError("Milestone 2")

    @property
    def T(self) -> Tensor:  # noqa: N802 - matches the NumPy/PyTorch spelling
        """Transpose of the last two axes."""
        raise NotImplementedError("Milestone 2")

    def __getitem__(self, index: Any) -> Tensor:
        """Index or slice.

        Backward scatters the gradient into a zero array at the same index -- with ``+=``, not
        ``=``. Fancy indexing may repeat an index, and NumPy's ``arr[idx] = v`` keeps only the
        last write while ``np.add.at`` accumulates. That difference is the whole bug.

        Tests: tests/test_tensor_autograd.py::test_getitem_backward_accumulates_repeated_indices
        """
        raise NotImplementedError("Milestone 2")

    def masked_fill(self, mask: np.ndarray, value: float) -> Tensor:
        """Return a copy with ``value`` where ``mask`` is True.

        Used by NanoLM's causal attention (its I-ATT-CAUSAL). Backward zeroes the gradient at
        masked positions: a filled constant has no dependence on the input.
        """
        raise NotImplementedError("Milestone 2")

    @staticmethod
    def concat(tensors: list[Tensor], axis: int = 0) -> Tensor:
        """Concatenate along ``axis``. Backward splits the gradient back at the same offsets."""
        raise NotImplementedError("Milestone 2")

    # -- constructors --------------------------------------------------------

    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
        """A zero-filled tensor."""
        raise NotImplementedError("Milestone 2")

    @staticmethod
    def ones(*shape: int, requires_grad: bool = False) -> Tensor:
        """A one-filled tensor."""
        raise NotImplementedError("Milestone 2")

    @staticmethod
    def randn(*shape: int, requires_grad: bool = False, seed: int | None = None) -> Tensor:
        """Standard-normal samples. ``seed`` makes a test reproducible."""
        raise NotImplementedError("Milestone 2")

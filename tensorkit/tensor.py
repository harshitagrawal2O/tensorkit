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

from tensorkit import autograd, ops
from tensorkit.ops import _dtype_guard

__all__ = ["Tensor"]

ArrayLike = np.ndarray | list[Any] | tuple[Any, ...] | float | int

#: ``_backward`` closures are stored per instance; this alias documents their type.
BackwardFn = Callable[[], None]

#: What a Python literal becomes when no dtype is asked for. An ndarray argument keeps its own
#: dtype instead: it already carries a deliberate choice by the caller, and silently narrowing
#: a float64 array to float32 would put every gradcheck below its tolerance.
DEFAULT_DTYPE = np.float32


def _noop() -> None:
    """Default ``_backward`` for leaves: a leaf has no inputs to propagate to."""
    return None


class _TapeArray(np.ndarray[Any, np.dtype[Any]]):
    """The array type ``Tensor.data`` holds: an ndarray that can be frozen against writes.

    NumPy has no write barrier and no version counter, so an engine cannot otherwise notice
    that a recorded tensor's values changed after its gradient rule closed over them
    (I-ACYCLIC). Marking the array read-only through ``flags.writeable`` would raise a NumPy
    ``ValueError`` about a read-only destination, which says nothing about tapes; this raises a
    ``RuntimeError`` that names the actual rule.

    It intercepts item assignment -- ``t.data[i] = v``, which is how in-place mutation is
    written in practice. In-place *ufuncs* (``t.data -= lr * g``) go through the C API and are
    deliberately still allowed: that is the optimiser's update path, and it happens between
    passes when no closure is waiting on the old values.
    """

    _frozen: bool

    def __array_finalize__(self, obj: Any) -> None:
        """Give every view and every derived array its own frozen flag.

        Views inherit the flag because they alias the same memory. Arrays built by
        :class:`Tensor` go through ``.view()`` and are unfrozen there, so an operation's result
        never arrives pre-frozen just because one of its operands was.
        """
        self._frozen = bool(getattr(obj, "_frozen", False))

    def freeze(self) -> None:
        """Refuse further item assignment: this buffer is now recorded on the tape."""
        self._frozen = True

    def thaw(self) -> None:
        """Allow item assignment again. Used when a fresh Tensor adopts a view of a buffer."""
        self._frozen = False

    def __setitem__(self, key: Any, value: Any) -> None:
        """Reject item assignment while this array is recorded on the tape."""
        if self._frozen:
            raise RuntimeError(
                "in-place mutation of a tensor that is already recorded as an input to an "
                "operation (I-ACYCLIC). Its backward closure captured the values as they were "
                "at the forward pass, so the gradient rule it will evaluate no longer matches "
                "the data. Build a new tensor instead, or call .detach() first if you really "
                "mean to step outside the graph."
            )
        super().__setitem__(key, value)


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

        The default applies to Python literals. An ``ndarray`` argument keeps the dtype it
        arrived with, so ``Tensor(x_float64)`` stays in float64 without every caller having to
        repeat itself -- and an integer array is refused outright rather than quietly cast
        (:func:`tensorkit.ops._dtype_guard`), because an integer tensor's gradient truncates to
        zero and reads as a vanishing gradient.
        """
        array = np.asarray(data)
        if dtype is not None:
            array = array.astype(np.dtype(dtype), copy=False)
        elif not isinstance(data, np.ndarray):
            array = array.astype(DEFAULT_DTYPE, copy=False)
        _dtype_guard(array)

        if not array.flags.c_contiguous:
            # 0-d arrays are always contiguous, so this branch never sees one -- which matters,
            # because np.ascontiguousarray promotes a 0-d array to 1-d and I-SHAPE forbids that.
            array = np.ascontiguousarray(array)

        # A fresh view, so the frozen flag belongs to this tensor and not to whatever array the
        # caller handed in (or to the operand an op's result happened to be derived from).
        buffer = array.view(_TapeArray)
        buffer.thaw()

        self.data = buffer
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._prev = _children
        self._backward = _noop
        self._op = _op
        self._retain_grad = False

    @classmethod
    def from_op(
        cls,
        data: np.ndarray,
        parents: tuple[Tensor, ...],
        op: str,
        make_backward: Callable[[Tensor], BackwardFn],
    ) -> Tensor:
        """Build the tape node for one primitive's forward result.

        Every rule in :mod:`tensorkit.ops` ends here, so the recording policy is written once:

        * Record only when gradients are enabled **and** some operand requires one (I-NOGRAD).
          A node with no differentiable input is a constant, and putting it on the tape costs
          memory to compute a gradient nobody will read.
        * ``make_backward`` is a factory rather than a closure because the rule needs to read
          ``out.grad``, and ``out`` does not exist until this method has built it.
        * Freeze each operand's buffer. The rule has just captured those values; changing them
          afterwards would leave the recorded derivative describing a function that no longer
          matches the data (I-ACYCLIC).

        Args:
            data: The forward result.
            parents: The operands, in argument order.
            op: The operator name, for ``repr`` and error messages.
            make_backward: Given the output node, returns its ``_backward`` closure.

        Returns:
            The output tensor, recorded on the tape unless recording is off.
        """
        recording = autograd.is_grad_enabled() and any(p.requires_grad for p in parents)
        out = cls(
            data,
            requires_grad=recording,
            dtype=data.dtype,
            _children=parents if recording else (),
            _op=op,
        )
        if recording:
            out._backward = make_backward(out)
            for parent in parents:
                buffer = parent.data
                if isinstance(buffer, _TapeArray):
                    buffer.freeze()
        return out

    # -- introspection -------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return int(self.data.ndim)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Element dtype."""
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        """True when this tensor was not produced by a recorded operation (``_prev == ()``)."""
        return not self._prev

    @property
    def retains_grad(self) -> bool:
        """True when this node keeps its gradient past the backward pass.

        Leaves always do (that is the point of a parameter); a non-leaf only does after
        :meth:`retain_grad`. :func:`tensorkit.autograd.backward` reads this to decide what to
        free, which is what keeps peak memory O(parameters) rather than O(activations).
        """
        return self._retain_grad

    def __repr__(self) -> str:
        """Show shape, dtype, op, and whether a gradient is present."""
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"op={self._op or 'leaf'!r}, requires_grad={self.requires_grad}, "
            f"grad={'set' if self.grad is not None else 'none'})"
        )

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

        The first arrival is *copied* rather than adopted: rules legitimately hand over
        read-only broadcast views and shared buffers, and adopting one would alias two
        tensors' gradients together the moment the second contribution arrived.

        Tests: tests/test_tensor_autograd.py::test_accumulate_grad_rejects_shape_mismatch
        """
        if incoming.shape != self.data.shape:
            raise ValueError(
                f"gradient shape {incoming.shape} does not match the tensor's shape "
                f"{self.data.shape} (op {self._op or 'leaf'!r}). An unbroadcast is missing "
                f"upstream: the gradient of an operand must always have the operand's own "
                f"shape (I-SHAPE). Letting NumPy broadcast it here would produce a "
                f"plausible-looking wrong answer instead of this message."
            )
        if self.grad is None:
            self.grad = np.array(incoming, dtype=self.data.dtype, copy=True)
        else:
            self.grad += incoming

    def zero_grad(self) -> None:
        """Set ``grad`` to None.

        None rather than ``zeros_like``, so "no gradient yet" and "gradient is exactly zero"
        stay distinguishable -- which is what makes I-ACCUM testable (I-MOD-ZERO).
        """
        self.grad = None

    def retain_grad(self) -> None:
        """Keep this non-leaf tensor's gradient after the backward pass. Debugging only."""
        self._retain_grad = True

    def backward(self, grad: np.ndarray | float | None = None) -> None:
        """Run the reverse pass from this tensor. Delegates to :func:`tensorkit.autograd.backward`.

        Invariant I-SEED: a non-scalar tensor requires an explicit ``grad``.

        Tests: tests/test_tensor_autograd.py::test_backward_requires_seed_for_non_scalar
        """
        autograd.backward(self, grad)

    def detach(self) -> Tensor:
        """Return a new leaf sharing this tensor's data with no history.

        Shares the buffer -- it does not copy. Mutating either view is visible in both, which is
        exactly why in-place mutation of a recorded tensor is forbidden (I-ACYCLIC).

        The result is a genuine leaf: ``_prev`` is empty, so nothing downstream of it can reach
        this tensor's history and the whole subgraph above it is free to be collected. It does
        carry one closure, though. The engine calls ``_backward`` on every node in the walk,
        leaves included, and this one reports a gradient of **exactly zero** back to the source.

        That zero is the derivative of a stop-gradient, and reporting it rather than staying
        silent is what makes ``.grad`` say which of two very different things happened:

        * ``grad is None`` -- the tensor was never in the graph at all. Look at
          ``requires_grad``, at ``no_grad()``, or at whether you stepped out to NumPy.
        * ``grad`` is all zeros -- the tensor *was* in the graph, along a path someone cut.
          Look for the ``detach()``.

        Conflating the two costs an afternoon, and ``gradcheck`` relies on the distinction: a
        severed path has to surface as a wrong number it can measure, not as a missing one.
        """
        source = self
        out = Tensor(source.data, requires_grad=False, dtype=source.data.dtype)

        def _backward() -> None:
            if source.requires_grad:
                source.accumulate_grad(np.zeros(source.shape, dtype=source.dtype))

        out._backward = _backward
        return out

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying array, detached from the graph."""
        return np.array(self.data, dtype=self.data.dtype, copy=True)

    def item(self) -> float:
        """Return the single element of a size-1 tensor. Raises otherwise."""
        if self.data.size != 1:
            raise ValueError(
                f"item() needs a tensor with exactly one element, but this one has shape "
                f"{self.shape} ({self.data.size} elements). Reduce it first, or use .numpy()."
            )
        return float(self.data.reshape(-1)[0])

    # -- elementwise ops -----------------------------------------------------
    # Each delegates to tensorkit.ops, which owns the gradient rules. Keeping the rules in one
    # module means gradcheck has one list to sweep and no rule can be quietly forgotten.

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise add, with broadcasting. Backward unbroadcasts to each operand's shape."""
        return ops.add(self, self._as_tensor(other))

    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise multiply, with broadcasting."""
        return ops.mul(self, self._as_tensor(other))

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise subtract, with broadcasting."""
        return ops.sub(self, self._as_tensor(other))

    def __truediv__(self, other: Tensor | ArrayLike) -> Tensor:
        """Elementwise divide.

        The quotient rule has two terms; dropping the second is the planted bug in
        ``tests/test_gradcheck.py::test_gradcheck_catches_a_planted_bug``.
        """
        return ops.div(self, self._as_tensor(other))

    def __pow__(self, exponent: float) -> Tensor:
        """Elementwise power with a constant exponent."""
        return ops.power(self, exponent)

    def __neg__(self) -> Tensor:
        """Elementwise negation."""
        return ops.neg(self)

    def __radd__(self, other: ArrayLike) -> Tensor:
        """Reflected add."""
        return ops.add(self._as_tensor(other), self)

    def __rmul__(self, other: ArrayLike) -> Tensor:
        """Reflected multiply."""
        return ops.mul(self._as_tensor(other), self)

    def __rsub__(self, other: ArrayLike) -> Tensor:
        """Reflected subtract."""
        return ops.sub(self._as_tensor(other), self)

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        """Reflected divide."""
        return ops.div(self._as_tensor(other), self)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiply.

        Backward: ``dA = dC @ B.T`` and ``dB = A.T @ dC``. Batched inputs broadcast over the
        leading dimensions, so both results need unbroadcasting back to the operand shapes --
        the case a 2-D-only implementation silently gets wrong once NanoLM's ``(B, H, T, d)``
        attention tensors show up.

        Tests: tests/test_tensor_autograd.py::test_matmul_batched_backward
        """
        return ops.matmul(self, self._as_tensor(other))

    def exp(self) -> Tensor:
        """Elementwise ``e ** x``. Local gradient: the output."""
        return ops.exp(self)

    def log(self) -> Tensor:
        """Elementwise natural log. Local gradient: ``1 / x``."""
        return ops.log(self)

    def sqrt(self) -> Tensor:
        """Elementwise square root. Local gradient: ``0.5 / out`` -- singular at 0."""
        return ops.sqrt(self)

    def tanh(self) -> Tensor:
        """Hyperbolic tangent, elementwise."""
        return ops.tanh(self)

    def relu(self) -> Tensor:
        """Elementwise ReLU."""
        return ops.relu(self)

    def abs(self) -> Tensor:
        """Elementwise absolute value. Kinked at 0; gradcheck must probe away from it."""
        return ops.abs_(self)

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
        return ops.sum_(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Mean over ``axis``. Backward is ``sum``'s divided by the number of elements reduced."""
        return ops.mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Maximum over ``axis``.

        Backward routes the whole gradient to the argmax position and zero elsewhere. Ties are
        broken deterministically (first occurrence) -- a documented choice, since routing to
        every tied position instead scales the gradient by the tie count.
        """
        return ops.max_(self, axis=axis, keepdims=keepdims)

    # -- shape ops -----------------------------------------------------------

    def reshape(self, *shape: int) -> Tensor:
        """Reshape. Backward reshapes the gradient back to the input's shape."""
        return ops.reshape(self, shape)

    def transpose(self, *axes: int) -> Tensor:
        """Permute axes.

        Backward applies the **inverse** permutation -- not the same permutation, which happens
        to coincide only for a 2-D swap and diverges silently for 3-D and above.
        """
        return ops.transpose(self, axes if axes else None)

    @property
    def T(self) -> Tensor:  # noqa: N802 - matches the NumPy/PyTorch spelling
        """Transpose of the last two axes."""
        if self.ndim < 2:
            return ops.transpose(self, None)
        axes = list(range(self.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return ops.transpose(self, tuple(axes))

    def __getitem__(self, index: Any) -> Tensor:
        """Index or slice.

        Backward scatters the gradient into a zero array at the same index -- with ``+=``, not
        ``=``. Fancy indexing may repeat an index, and NumPy's ``arr[idx] = v`` keeps only the
        last write while ``np.add.at`` accumulates. That difference is the whole bug.

        Tests: tests/test_tensor_autograd.py::test_getitem_backward_accumulates_repeated_indices
        """
        return ops.getitem(self, index)

    def masked_fill(self, mask: np.ndarray, value: float) -> Tensor:
        """Return a copy with ``value`` where ``mask`` is True.

        Used by NanoLM's causal attention (its I-ATT-CAUSAL). Backward zeroes the gradient at
        masked positions: a filled constant has no dependence on the input.

        The rule lives here rather than in :mod:`tensorkit.ops` on purpose: everything in that
        module is swept by gradcheck, and a fill has no gradcheck case (its mask argument is
        not a differentiable input). Registering it would make the coverage test claim a check
        that does not exist.
        """
        selector = np.asarray(mask, dtype=bool)
        data = np.where(selector, np.asarray(value, dtype=self.dtype), self.data)

        def _rule(out: Tensor) -> BackwardFn:
            def _backward() -> None:
                g = out.grad
                if g is None or not self.requires_grad:
                    return
                self.accumulate_grad(g * ~selector)

            return _backward

        return Tensor.from_op(data, (self,), "masked_fill", _rule)

    @staticmethod
    def concat(tensors: list[Tensor], axis: int = 0) -> Tensor:
        """Concatenate along ``axis``. Backward splits the gradient back at the same offsets."""
        return ops.concat(tensors, axis=axis)

    # -- constructors --------------------------------------------------------

    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
        """A zero-filled tensor."""
        return Tensor(np.zeros(shape, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape: int, requires_grad: bool = False) -> Tensor:
        """A one-filled tensor."""
        return Tensor(np.ones(shape, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape: int, requires_grad: bool = False, seed: int | None = None) -> Tensor:
        """Standard-normal samples. ``seed`` makes a test reproducible."""
        rng = np.random.default_rng(seed)
        return Tensor(rng.standard_normal(shape).astype(DEFAULT_DTYPE), requires_grad=requires_grad)

    # -- internals -----------------------------------------------------------

    def _as_tensor(self, other: Tensor | ArrayLike) -> Tensor:
        """Coerce the right-hand side of an operator to a Tensor.

        A bare number or array becomes a constant leaf in *this* tensor's dtype, so ``x * 2``
        does not silently promote a float32 graph to float64, and an integer literal is a
        legitimate constant rather than an integer tensor the dtype guard would refuse.
        """
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.dtype), dtype=self.dtype)

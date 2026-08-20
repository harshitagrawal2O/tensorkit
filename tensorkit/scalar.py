"""Scalar reverse-mode automatic differentiation -- Milestone 1.

This module exists for one reason: the whole of reverse-mode autodiff fits in a scalar
implementation you can hold in your head, and every bug you will hit in ``tensor.py`` has a
smaller, more legible version here. Build it first, understand it completely, then generalise.

Concepts: ``docs/concepts/autodiff.md``.
Tests: ``tests/test_scalar_autograd.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import cast

from tensorkit import autograd

__all__ = ["Value"]


class Value:
    """A scalar node in the computational graph.

    Attributes:
        data: The scalar produced by the forward pass.
        grad: d(output) / d(self), accumulated during the backward pass. Starts at 0.0.
        _prev: The Values this node was computed from. Empty tuple for a leaf.
        _backward: A closure that reads ``self.grad`` and accumulates into ``_prev`` grads.
        _op: The operator that produced this node, for graph rendering and error messages.

    Invariants (``SPEC.md`` section 3.1):
        I-ACCUM: ``_backward`` accumulates with ``+=``; it never assigns.
        I-ACYCLIC: the graph reachable through ``_prev`` is a DAG.
        I-ONCE: one ``backward()`` call runs each reachable ``_backward`` exactly once.
        I-READY: when a node's ``_backward`` runs, every consumer of it has already run.
    """

    __slots__ = ("data", "grad", "_prev", "_backward", "_op")

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        """Create a Value. Leaves pass no children."""
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: tuple[Value, ...] = _children
        self._backward: Callable[[], None] = _noop
        self._op: str = _op

    def __repr__(self) -> str:
        """Show the value, its gradient, and the op that produced it."""
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self._op or 'leaf'!r})"

    # -- primitives ----------------------------------------------------------

    def __add__(self, other: Value | float) -> Value:
        """Return ``self + other``.

        Local gradient: d(a+b)/da = 1 and d(a+b)/db = 1, so the closure adds ``out.grad`` to
        both inputs. Add, never assign: ``a + a`` must give ``a.grad == 2``, and an assignment
        silently gives 1 (I-ACCUM).

        Tests: tests/test_scalar_autograd.py::test_add_backward
        """
        rhs = Value._coerce(other)
        out = Value(self.data + rhs.data, (self, rhs), "+")

        def _backward() -> None:
            self.grad += out.grad
            rhs.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        """Return ``self * other``.

        Local gradient: d(ab)/da = b and d(ab)/db = a. Capture the operand *values* in the
        closure. Reading ``other.data`` at closure-call time is fine here because ``.data`` is
        immutable by convention; reading ``other.grad`` would not be.

        Tests: tests/test_scalar_autograd.py::test_mul_backward
        """
        rhs = Value._coerce(other)
        out = Value(self.data * rhs.data, (self, rhs), "*")

        def _backward() -> None:
            self.grad += rhs.data * out.grad
            rhs.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> Value:
        """Return ``self ** other`` for a constant exponent.

        Only int/float exponents are in scope. A Value exponent needs the
        ``d/dx a**x = a**x * ln a`` term; raise a clear ``TypeError`` for one rather than
        computing something plausible and wrong.

        Tests: tests/test_scalar_autograd.py::test_pow_backward
        """
        try:
            exponent = float(other)
        except TypeError as exc:
            # A Value has no __float__, which is exactly the case being rejected.
            raise TypeError(
                f"the exponent of Value ** exponent must be a constant int or float, not "
                f"{type(other).__name__}. A Value exponent also needs the "
                f"d/dx (a ** x) = a ** x * ln(a) term, which this milestone does not "
                f"implement -- and a silently missing term is worse than a refusal."
            ) from exc
        out = Value(self.data**exponent, (self,), f"**{exponent:g}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1.0) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """Return ``max(self, 0)``.

        The derivative at exactly 0 is undefined. Pick a convention (0 is conventional),
        document it, and note in ``docs/concepts/gradcheck.md`` why the numerical check must
        probe away from the kink.

        Convention here: the subgradient at 0 is **0**, so ``relu`` is treated as the map
        ``x -> x * (x > 0)``. Any value in [0, 1] is a valid subgradient; 0 is what PyTorch
        uses, and picking the same one keeps cross-checks against it meaningful.

        Tests: tests/test_scalar_autograd.py::test_relu_backward
        """
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Return ``e ** self``. Local gradient: the output itself.

        Tests: tests/test_scalar_autograd.py::test_exp_backward
        """
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Return the natural log. Local gradient: ``1 / self``.

        ``log`` of a non-positive value must raise rather than return ``-inf``/``nan`` and
        poison the entire backward pass with silent NaNs that surface ten layers away.

        Tests: tests/test_scalar_autograd.py::test_log_backward
        """
        if self.data <= 0.0:
            raise ValueError(
                f"log() is undefined for {self.data!r}: the domain is x > 0. Returning -inf or "
                f"nan here would propagate through the first multiplication and destroy every "
                f"gradient in the batch, far away from the mistake."
            )
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Hyperbolic tangent. Local gradient: ``1 - out ** 2``, where ``out`` is the result.

        Use the output rather than recomputing ``math.tanh``: the identity is exact and the
        value is already to hand.

        Tests: tests/test_scalar_autograd.py::test_tanh_backward
        """
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - out.data * out.data) * out.grad

        out._backward = _backward
        return out

    # -- the graph walk ------------------------------------------------------

    def topological_order(self) -> list[Value]:
        """Return every node reachable from self, with parents before children.

        Must be **iterative**. A recursive DFS blows the Python stack at a few thousand nodes
        and a 10,000-node chain is a perfectly ordinary network. The test asserts exactly that.

        Complexity: O(V + E) time, O(V) space.

        The algorithm itself lives in :func:`tensorkit.autograd.topological_order`, which is
        shared with :class:`tensorkit.tensor.Tensor`: one sort, one set of tests, one place for
        the cycle check to be right.

        Tests: tests/test_scalar_autograd.py::test_topological_order_is_iterative
        """
        # The shared walk is typed against the structural GraphNode protocol; every node it can
        # reach from a Value is a Value, because no operator here mixes node types.
        return cast(list["Value"], autograd.topological_order(self))

    def backward(self) -> None:
        """Populate ``.grad`` on every node reachable from self.

        Algorithm:
          1. Linearise the graph with :meth:`topological_order`.
          2. Seed ``self.grad = 1.0`` -- that is d(self)/d(self).
          3. Walk the order in **reverse**, calling each node's ``_backward`` exactly once.

        Why reverse topological order rather than a recursive descent: a node with two consumers
        must not propagate until both consumers have contributed, or it propagates a partial
        gradient (I-READY). Reverse topological order is precisely the schedule that guarantees
        that, and as a bonus it visits each node once (I-ONCE) rather than once per path -- the
        difference between linear and exponential on a chain of diamonds.

        This does **not** zero gradients first. Calling ``backward()`` twice accumulates, which
        is what gradient accumulation across microbatches needs. Document it; do not prevent it.

        What a second call *does* reset is the intermediates: every non-leaf gradient is set to
        0.0 before the walk, so each pass contributes exactly one copy of the derivative to the
        leaves. Without that, a shared intermediate would carry the previous pass's total into
        this one and the leaves would grow faster than linearly in the number of calls.

        Tests: tests/test_scalar_autograd.py::test_backward_visits_each_node_once
        """
        autograd.backward(self)

    # -- derived operators ---------------------------------------------------
    # All expressible via the primitives above. Implement them that way rather than adding new
    # backward rules: fewer gradient rules means fewer places to be wrong.

    def __neg__(self) -> Value:
        """Return ``-self``."""
        return self * -1.0

    def __radd__(self, other: float) -> Value:
        """Return ``other + self``."""
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        """Return ``self - other``."""
        return self + (-Value._coerce(other))

    def __rsub__(self, other: float) -> Value:
        """Return ``other - self``."""
        return (-self) + other

    def __rmul__(self, other: float) -> Value:
        """Return ``other * self``."""
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        """Return ``self / other``."""
        return self * Value._coerce(other) ** -1.0

    def __rtruediv__(self, other: float) -> Value:
        """Return ``other / self``."""
        return Value._coerce(other) * self**-1.0

    def __iter__(self) -> Iterator[Value]:
        """Values are scalars and are deliberately not iterable."""
        raise TypeError("Value is a scalar and is not iterable")

    @staticmethod
    def _coerce(other: Value | float) -> Value:
        """Wrap a bare number as a leaf Value; pass a Value through unchanged."""
        return other if isinstance(other, Value) else Value(float(other))


def _noop() -> None:
    """Default ``_backward`` for leaves: a leaf has no inputs to propagate to."""
    return None

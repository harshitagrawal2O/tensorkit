"""Scalar reverse-mode automatic differentiation -- Milestone 1.

This module exists for one reason: the whole of reverse-mode autodiff fits in a scalar
implementation you can hold in your head, and every bug you will hit in ``tensor.py`` has a
smaller, more legible version here. Build it first, understand it completely, then generalise.

Concepts: ``docs/concepts/autodiff.md``.
Tests: ``tests/test_scalar_autograd.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

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
        raise NotImplementedError("Milestone 1")

    def __mul__(self, other: Value | float) -> Value:
        """Return ``self * other``.

        Local gradient: d(ab)/da = b and d(ab)/db = a. Capture the operand *values* in the
        closure. Reading ``other.data`` at closure-call time is fine here because ``.data`` is
        immutable by convention; reading ``other.grad`` would not be.

        Tests: tests/test_scalar_autograd.py::test_mul_backward
        """
        raise NotImplementedError("Milestone 1")

    def __pow__(self, other: float) -> Value:
        """Return ``self ** other`` for a constant exponent.

        Only int/float exponents are in scope. A Value exponent needs the
        ``d/dx a**x = a**x * ln a`` term; raise a clear ``TypeError`` for one rather than
        computing something plausible and wrong.

        Tests: tests/test_scalar_autograd.py::test_pow_backward
        """
        raise NotImplementedError("Milestone 1")

    def relu(self) -> Value:
        """Return ``max(self, 0)``.

        The derivative at exactly 0 is undefined. Pick a convention (0 is conventional),
        document it, and note in ``docs/concepts/gradcheck.md`` why the numerical check must
        probe away from the kink.

        Tests: tests/test_scalar_autograd.py::test_relu_backward
        """
        raise NotImplementedError("Milestone 1")

    def exp(self) -> Value:
        """Return ``e ** self``. Local gradient: the output itself.

        Tests: tests/test_scalar_autograd.py::test_exp_backward
        """
        raise NotImplementedError("Milestone 1")

    def log(self) -> Value:
        """Return the natural log. Local gradient: ``1 / self``.

        ``log`` of a non-positive value must raise rather than return ``-inf``/``nan`` and
        poison the entire backward pass with silent NaNs that surface ten layers away.

        Tests: tests/test_scalar_autograd.py::test_log_backward
        """
        raise NotImplementedError("Milestone 1")

    def tanh(self) -> Value:
        """Hyperbolic tangent. Local gradient: ``1 - out ** 2``, where ``out`` is the result.

        Use the output rather than recomputing ``math.tanh``: the identity is exact and the
        value is already to hand.

        Tests: tests/test_scalar_autograd.py::test_tanh_backward
        """
        raise NotImplementedError("Milestone 1")

    # -- the graph walk ------------------------------------------------------

    def topological_order(self) -> list[Value]:
        """Return every node reachable from self, with parents before children.

        Must be **iterative**. A recursive DFS blows the Python stack at a few thousand nodes
        and a 10,000-node chain is a perfectly ordinary network. The test asserts exactly that.

        Complexity: O(V + E) time, O(V) space.

        Tests: tests/test_scalar_autograd.py::test_topological_order_is_iterative
        """
        raise NotImplementedError("Milestone 1")

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

        Tests: tests/test_scalar_autograd.py::test_backward_visits_each_node_once
        """
        raise NotImplementedError("Milestone 1")

    # -- derived operators ---------------------------------------------------
    # All expressible via the primitives above. Implement them that way rather than adding new
    # backward rules: fewer gradient rules means fewer places to be wrong.

    def __neg__(self) -> Value:
        """Return ``-self``."""
        raise NotImplementedError("Milestone 1")

    def __radd__(self, other: float) -> Value:
        """Return ``other + self``."""
        raise NotImplementedError("Milestone 1")

    def __sub__(self, other: Value | float) -> Value:
        """Return ``self - other``."""
        raise NotImplementedError("Milestone 1")

    def __rsub__(self, other: float) -> Value:
        """Return ``other - self``."""
        raise NotImplementedError("Milestone 1")

    def __rmul__(self, other: float) -> Value:
        """Return ``other * self``."""
        raise NotImplementedError("Milestone 1")

    def __truediv__(self, other: Value | float) -> Value:
        """Return ``self / other``."""
        raise NotImplementedError("Milestone 1")

    def __rtruediv__(self, other: float) -> Value:
        """Return ``other / self``."""
        raise NotImplementedError("Milestone 1")

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

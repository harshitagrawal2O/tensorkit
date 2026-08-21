"""The graph walk and the gradient-recording context -- Milestones 1 and 2.

``scalar.Value`` and ``tensor.Tensor`` are two node types over the same graph algorithm. That
algorithm lives here so it is written, tested, and reasoned about exactly once.

Concepts: ``docs/concepts/autodiff.md``.
Tests: ``tests/test_scalar_autograd.py``, ``tests/test_tensor_autograd.py``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = ["GraphNode", "topological_order", "backward", "no_grad", "is_grad_enabled"]


@runtime_checkable
class GraphNode(Protocol):
    """The structural contract every tape node satisfies.

    Both :class:`tensorkit.scalar.Value` and :class:`tensorkit.tensor.Tensor` satisfy it, which
    is what lets one topological sort serve both.
    """

    _prev: tuple[GraphNode, ...]
    _op: str

    def _backward(self) -> None:
        """Read this node's gradient and accumulate into each input's gradient."""
        ...


# ---------------------------------------------------------------------------
# no_grad -- GIVEN. Thread-local because a process pool worker (03-inferserve)
# may hold several loops, and a global flag would leak across them.
# ---------------------------------------------------------------------------

_state = threading.local()


def is_grad_enabled() -> bool:
    """Return whether operations should currently record onto the tape."""
    return bool(getattr(_state, "grad_enabled", True))


@contextmanager
def no_grad() -> Iterator[None]:
    """Disable tape recording inside the block.

    Every op still computes its forward value; none of them records ``_prev`` or ``_backward``.
    Two places need this and both are load-bearing:

    * **Evaluation.** Building a graph you never call ``backward()`` on retains every
      activation until the next garbage collection -- an easy way to OOM during validation.
    * **The optimiser step.** ``p.data -= lr * p.grad`` written as tape operations extends the
      graph every step, so the tape grows without bound across training (I-OPT-NOGRAD).

    Nesting restores the previous state rather than unconditionally re-enabling, so a
    ``no_grad`` inside a ``no_grad`` does not silently turn recording back on.

    Tests: tests/test_tensor_autograd.py::test_no_grad_leaves_the_tape_empty
    """
    previous = is_grad_enabled()
    _state.grad_enabled = False
    try:
        yield
    finally:
        _state.grad_enabled = previous


# ---------------------------------------------------------------------------
# The graph walk -- YOURS.
# ---------------------------------------------------------------------------


def topological_order(root: GraphNode) -> list[GraphNode]:
    """Return every node reachable from ``root``, parents before children.

    Args:
        root: The node to walk back from -- normally a scalar loss.

    Returns:
        A list in topological order. ``root`` is last.

    Requirements:
      * **Iterative.** Use an explicit stack with an explicit colour/visited marking. Recursion
        dies at ``sys.getrecursionlimit()``, which a 30-layer network with a few hundred ops per
        layer passes comfortably.
      * **Each node once.** Deduplicate by ``id()``, not by equality -- two distinct tensors can
        hold equal data and must remain distinct nodes.
      * **Cycle detection.** A cycle means an op consumed its own output (I-ACYCLIC). Raise a
        named error identifying the node rather than looping forever; the failure mode without
        this is a hang, which is much harder to debug than an exception.

    Complexity: O(V + E) time, O(V) space.

    Tests: tests/test_scalar_autograd.py::test_topological_order_puts_parents_before_children
    """
    raise NotImplementedError("Milestone 1")


def backward(root: GraphNode, grad: np.ndarray | float | None = None) -> None:
    """Run the reverse pass from ``root``.

    Args:
        root: The node to differentiate.
        grad: The seed gradient, ``d(final) / d(root)``. Required when ``root`` is not a scalar.

    Algorithm: topologically sort, seed ``root.grad``, walk in reverse calling ``_backward``.

    Invariants (``SPEC.md`` section 3.1):
        I-SEED: a non-scalar root without an explicit ``grad`` raises. Implicitly summing a
            non-scalar output is a silent semantic choice, and the caller who wanted a mean
            instead gets wrong gradients with no warning.
        I-ONCE: each reachable ``_backward`` is called exactly once.
        I-READY: no ``_backward`` runs before all of that node's consumers have.
        I-LEAF: non-leaf gradients are released after their ``_backward`` runs, unless the node
            requested retention. Holding every intermediate gradient makes peak memory O(graph)
            instead of O(parameters).

    Raises:
        RuntimeError: if ``root`` is non-scalar and ``grad`` is None (I-SEED).

    Tests: tests/test_tensor_autograd.py::test_backward_requires_seed_for_non_scalar
    """
    raise NotImplementedError("Milestone 1")

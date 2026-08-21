"""The graph walk and the gradient-recording context -- Milestones 1 and 2.

``scalar.Value`` and ``tensor.Tensor`` are two node types over the same graph algorithm. That
algorithm lives here so it is written, tested, and reasoned about exactly once.

Concepts: ``docs/concepts/autodiff.md``.
Tests: ``tests/test_autograd.py``.
"""

# ruff: noqa: SLF001
# The tape protocol -- ``_prev``, ``_op``, ``_backward`` -- is private on the node types on
# purpose: nothing outside the engine should walk a tensor's history. This module *is* the
# engine, and it is the one place that has to reach through that protocol, so the private-member
# rule is disabled here rather than papered over with a public alias that would widen the API
# for every caller just to satisfy a linter.

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast, runtime_checkable

import numpy as np

__all__ = [
    "GraphNode",
    "TensorNode",
    "ScalarNode",
    "CyclicGraphError",
    "topological_order",
    "backward",
    "no_grad",
    "is_grad_enabled",
]


@runtime_checkable
class GraphNode(Protocol):
    """The structural contract every tape node satisfies.

    Both :class:`tensorkit.scalar.Value` and :class:`tensorkit.tensor.Tensor` satisfy it, which
    is what lets one topological sort serve both.

    ``_prev`` is declared read-only (a property rather than a mutable attribute) for a typing
    reason worth knowing: a *settable* protocol member is invariant, so ``Value``'s
    ``tuple[Value, ...]`` would not satisfy ``tuple[GraphNode, ...]`` and neither node type
    would type-check against this protocol. Read-only members are covariant, and read-only is
    the truth anyway -- the walk only ever reads a node's inputs.
    """

    _op: str

    @property
    def _prev(self) -> tuple[GraphNode, ...]:
        """The nodes this one was computed from. Empty for a leaf."""
        ...

    def _backward(self) -> None:
        """Read this node's gradient and accumulate into each input's gradient."""
        ...


@runtime_checkable
class TensorNode(Protocol):
    """A tape node whose gradient is an array -- :class:`tensorkit.tensor.Tensor`.

    The seeding rule (I-SEED) and the release rule (I-LEAF) are both shape-aware, so
    :func:`backward` needs more of the node than :class:`GraphNode` promises. Declaring the
    extra surface as its own protocol keeps the driver honest about what it touches, and
    ``isinstance`` against it is what distinguishes a tensor graph from a scalar one --
    :class:`tensorkit.scalar.Value` has ``data`` and ``grad`` but no ``accumulate_grad``.
    """

    data: np.ndarray
    grad: np.ndarray | None

    @property
    def is_leaf(self) -> bool:
        """True when this node was not produced by a recorded operation."""
        ...

    @property
    def retains_grad(self) -> bool:
        """True when a non-leaf was asked to keep its gradient past the backward pass."""
        ...

    def accumulate_grad(self, incoming: np.ndarray) -> None:
        """Add ``incoming`` into this node's gradient."""
        ...


@runtime_checkable
class ScalarNode(Protocol):
    """A tape node whose gradient is a plain float -- :class:`tensorkit.scalar.Value`."""

    grad: float


class CyclicGraphError(RuntimeError):
    """Raised when the graph reachable through ``_prev`` contains a cycle (I-ACYCLIC).

    A cycle means an operation consumed its own output, which reverse mode cannot schedule:
    there is no order in which every consumer runs before its input. Without this error the
    sort loops forever, and a hang is far harder to diagnose than an exception that names the
    offending node.
    """


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

    Tests: tests/test_autograd.py::test_no_grad_leaves_tape_empty
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

#: Depth-first colours. A node is OPEN while it sits on the current DFS path, which is exactly
#: the condition that makes re-entering it a cycle rather than a diamond.
_OPEN = 0
_DONE = 1


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

    Implementation note: this is the standard two-phase iterative post-order. Each node is
    pushed twice -- once to expand its inputs, once to emit it after they are all done -- which
    is what makes the emission order a true post-order and therefore a topological order.
    Keying the colour map by ``id()`` is safe here because every node in it stays reachable
    from ``root`` (and so alive) for the whole walk, so no id can be recycled underneath us.

    Tests: tests/test_autograd.py::test_topological_order_parents_before_children
    """
    order: list[GraphNode] = []
    colour: dict[int, int] = {}
    stack: list[tuple[GraphNode, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        key = id(node)

        if expanded:
            colour[key] = _DONE
            order.append(node)
            continue

        if key in colour:
            # Already emitted, or already scheduled by another consumer: this duplicate entry
            # is a no-op. Skipping it is what keeps a diamond linear instead of exponential.
            continue

        colour[key] = _OPEN
        stack.append((node, True))
        for parent in node._prev:
            state = colour.get(id(parent))
            if state == _OPEN:
                raise CyclicGraphError(
                    f"cycle detected in the autograd graph: {parent!r} is reachable from its "
                    f"own consumer {node!r} (op {node._op!r}). An operation may not take its "
                    f"own output as an input -- see I-ACYCLIC."
                )
            if state is None:
                stack.append((parent, False))

    return order


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

    A graph is homogeneous -- all :class:`~tensorkit.tensor.Tensor` or all
    :class:`~tensorkit.scalar.Value`, because the two node types have no operator that mixes
    them -- so the node kind is decided once from ``root`` rather than re-tested per node.

    The two kinds differ only in how a repeated ``backward()`` stays honest. Tensors release
    each intermediate gradient as soon as it has propagated, so the next pass starts from
    ``None``. Values have no release step (their gradients are floats and cost nothing to
    keep, and being able to read them afterwards is the point of the scalar engine), so the
    intermediates are reset to 0.0 up front instead. Either way a second ``backward()`` adds
    exactly one more copy of the gradient to the leaves, which is what microbatch accumulation
    relies on.

    Tests: tests/test_autograd.py::test_backward_requires_seed_for_non_scalar
    """
    order = topological_order(root)

    if isinstance(root, TensorNode):
        root.accumulate_grad(_tensor_seed(root, grad))
        for node in reversed(order):
            node._backward()
            tensor_node = cast(TensorNode, node)
            if not tensor_node.is_leaf and not tensor_node.retains_grad:
                # I-LEAF: the node has propagated, so nothing will read this again.
                tensor_node.grad = None
        return

    scalar_root = cast(ScalarNode, root)
    for node in order:
        if node._prev:
            # Every non-leaf, the root included: its gradient belongs to this pass alone.
            cast(ScalarNode, node).grad = 0.0
    scalar_root.grad += 1.0 if grad is None else float(cast(float, grad))
    for node in reversed(order):
        node._backward()


def _tensor_seed(root: TensorNode, grad: np.ndarray | float | None) -> np.ndarray:
    """Return the seed gradient for a tensor root, enforcing I-SEED.

    ``d(root)/d(root)`` is ones. Anything else has to be supplied by the caller, because the
    engine cannot know which scalar function of a non-scalar output was meant.
    """
    if grad is None:
        if root.data.size != 1:
            raise RuntimeError(
                f"backward() on a non-scalar tensor of shape {root.data.shape} needs an "
                f"explicit gradient: pass backward(grad) with an array of that shape. "
                f"Implicitly summing to a scalar would be a silent semantic choice (I-SEED)."
            )
        return np.ones(root.data.shape, dtype=root.data.dtype)

    seed = np.asarray(grad, dtype=root.data.dtype)
    if seed.shape != root.data.shape:
        if seed.size != root.data.size:
            raise ValueError(
                f"seed gradient of shape {seed.shape} cannot be used for a root of shape "
                f"{root.data.shape}: the seed must match the output elementwise (I-SHAPE)."
            )
        seed = seed.reshape(root.data.shape)
    return np.array(seed, dtype=root.data.dtype, copy=True)

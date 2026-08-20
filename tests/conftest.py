"""Shared fixtures for the TensorKit suite.

Two conventions the whole suite depends on:

* **float64 everywhere.** Gradient checking in float32 has a relative error floor around 1e-3,
  so a float32 test either asserts a tolerance so loose it catches nothing or fails on correct
  code. Every fixture here produces float64.
* **Seeded RNG.** A test that fails one run in twenty is worse than no test, because it teaches
  you to re-run rather than to look.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator. Same numbers every run, on every machine."""
    return np.random.default_rng(20250820)


@pytest.fixture
def f64(rng):
    """Build a float64 Tensor with requires_grad set, of any shape."""
    from tensorkit.tensor import Tensor

    def _make(*shape: int, scale: float = 1.0, offset: float = 0.0):
        data = rng.standard_normal(shape) * scale + offset
        return Tensor(data.astype(np.float64), requires_grad=True, dtype=np.float64)

    return _make


@pytest.fixture
def positive_f64(rng):
    """A strictly positive float64 Tensor, for log/sqrt/div denominators.

    Kept away from zero on purpose: ``log`` and ``sqrt`` are singular there and ``div`` blows
    up, so a test that happens to sample near zero fails for reasons that have nothing to do
    with the gradient rule under test.
    """
    from tensorkit.tensor import Tensor

    def _make(*shape: int):
        data = rng.uniform(0.5, 2.5, size=shape)
        return Tensor(data.astype(np.float64), requires_grad=True, dtype=np.float64)

    return _make


@pytest.fixture
def count_backward_calls():
    """Wrap every node's ``_backward`` in a counter, keyed by ``id(node)``.

    This is how I-ONCE is asserted: not by inspecting the topological sort, but by observing
    that each closure actually fired exactly once. A sort can be right while the walk that
    consumes it is wrong.
    """

    def _instrument(root) -> dict[int, int]:
        counts: dict[int, int] = {}
        seen: set[int] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            counts[id(node)] = 0

            def make(n, original):
                def wrapped() -> None:
                    counts[id(n)] += 1
                    original()

                return wrapped

            node._backward = make(node, node._backward)
            stack.extend(node._prev)
        return counts

    return _instrument

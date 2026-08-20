"""Activation functions -- Milestone 4.

Concepts: ``docs/concepts/activations.md``.
Tests: ``tests/test_layers.py``.
"""

from __future__ import annotations

from tensorkit.nn.module import Module
from tensorkit.tensor import Tensor

__all__ = ["ReLU", "GELU", "Softmax", "Tanh", "Sigmoid"]


class ReLU(Module):
    """``max(x, 0)``.

    Gradient is 1 where ``x > 0`` and 0 elsewhere; at exactly 0 pick a convention (0) and
    document it. Note the strict inequality: using ``x >= 0`` passes gradient through a dead
    unit and changes the dynamics of dying-ReLU in a way that is hard to notice.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Return ``max(x, 0)``."""
        raise NotImplementedError("Milestone 4")


class GELU(Module):
    """Gaussian Error Linear Unit.

    Args:
        approximate: Use the tanh approximation instead of the exact erf form.

    Both forms must exist and a test asserts they agree to 1e-3. GELU is smooth everywhere,
    which is why transformers prefer it to ReLU: no kink means no discontinuity in the
    gradient, which matters when the residual stream sums many such paths.

    Tests: tests/test_layers.py::test_gelu_exact_vs_approximate
    """

    def __init__(self, approximate: bool = True) -> None:
        """Select the exact or approximate form."""
        raise NotImplementedError("Milestone 4")

    def forward(self, x: Tensor) -> Tensor:
        """Return GELU(x)."""
        raise NotImplementedError("Milestone 4")


class Softmax(Module):
    """Softmax over ``axis``, numerically stable.

    Subtract the per-slice maximum before exponentiating. ``softmax([1000, 1001])`` must be
    finite -- a test asserts exactly that, because it is the case the naive implementation
    turns into ``nan``.

    Tests: tests/test_layers.py::test_softmax_stability
    """

    def __init__(self, axis: int = -1) -> None:
        """Store the reduction axis."""
        raise NotImplementedError("Milestone 4")

    def forward(self, x: Tensor) -> Tensor:
        """Return softmax(x) along ``axis``."""
        raise NotImplementedError("Milestone 4")


class Tanh(Module):
    """Hyperbolic tangent. Gradient ``1 - tanh(x)**2``."""

    def forward(self, x: Tensor) -> Tensor:
        """Return tanh(x)."""
        raise NotImplementedError("Milestone 4")


class Sigmoid(Module):
    """Logistic sigmoid. Gradient ``s * (1 - s)``.

    Compute it as ``exp(x) / (1 + exp(x))`` for negative ``x`` and ``1 / (1 + exp(-x))`` for
    positive ``x``. A single branch overflows on one side or the other -- ``exp(800)`` is
    ``inf`` -- and the sign-split form is exact across the whole range.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Return sigmoid(x)."""
        raise NotImplementedError("Milestone 4")

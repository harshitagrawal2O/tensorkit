"""Stochastic gradient descent -- Milestone 6.

Tests: ``tests/test_optim.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tensorkit.nn.module import Parameter
from tensorkit.optim.optimizer import Optimizer

__all__ = ["SGD"]


class SGD(Optimizer):
    """SGD with optional momentum, Nesterov acceleration, and weight decay.

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate.
        momentum: Momentum coefficient; 0 disables it.
        weight_decay: L2 penalty added to the gradient.
        nesterov: Use Nesterov momentum. Requires ``momentum > 0``.

    Plain SGD::

        p -= lr * g

    With momentum, an exponentially weighted average of past gradients::

        v = momentum * v + g
        p -= lr * v

    Momentum damps oscillation across a ravine's steep axis while accumulating along its
    shallow one -- the geometry that makes it help. Nesterov evaluates the gradient at the
    *look-ahead* point ``p - lr * momentum * v``, so it can correct an overshoot within the
    same step instead of the next one.

    Note this is the PyTorch convention (``v = mu*v + g``, no ``(1-mu)`` damping). The
    alternative convention scales the effective learning rate by ``1/(1-mu)``, so a run ported
    between the two silently changes learning rate by 10x at ``mu=0.9``. State the convention.

    Invariants:
        ``momentum=0`` is bitwise plain gradient descent -- a regression guard.
        Weight decay is added to the gradient *before* momentum, so decay accumulates in the
        velocity too. That is the coupling AdamW exists to break.

    Tests: tests/test_optim.py::test_sgd_zero_momentum_is_plain_gd
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        """Validate the hyperparameters and initialise the base class."""
        raise NotImplementedError("Milestone 6")

    def _step_group(self, group: dict[str, Any]) -> None:
        """Apply the SGD update to every parameter in ``group``."""
        raise NotImplementedError("Milestone 6")

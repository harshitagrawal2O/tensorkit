"""Loss functions -- Milestone 6.

Both losses here are trivial mathematically and non-trivial numerically. That is the lesson:
the formula on the whiteboard and the formula you implement are not the same formula.

Concepts: ``docs/concepts/cross-entropy-stability.md``.
Tests: ``tests/test_losses.py``.
"""

from __future__ import annotations

import numpy as np

from tensorkit.nn.module import Module
from tensorkit.tensor import Tensor

__all__ = ["MSELoss", "CrossEntropyLoss"]


class MSELoss(Module):
    """Mean squared error.

    Args:
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    The reduction is not cosmetic: ``mean`` divides the gradient by the number of elements,
    so switching from ``mean`` to ``sum`` changes the effective learning rate by the batch
    size. Tests assert both and assert the ratio between them.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Store the reduction mode."""
        raise NotImplementedError("Milestone 6")

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return the mean squared error between ``prediction`` and ``target``.

        VJP: ``d/dp = 2 * (p - t) / n`` for ``mean``.

        ``target`` must not require gradients -- it is data, not a parameter. A test asserts
        no gradient flows into it, because a target on the tape silently makes the loss
        differentiable with respect to the labels.

        Tests: tests/test_losses.py::test_mse_reduction_modes
        """
        raise NotImplementedError("Milestone 6")


class CrossEntropyLoss(Module):
    """Softmax cross-entropy over logits, fused with log-softmax.

    Takes **logits**, not probabilities. Passing a softmax output into this is a real bug that
    trains anyway, just badly: applying softmax twice flattens the distribution and the
    gradients shrink accordingly.

    Args:
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        ignore_index: Target value to skip entirely -- padding positions in NanoLM. Ignored
            positions must not contribute to the loss *or* to the denominator of the mean;
            dividing by the padded length instead of the real length quietly scales the loss.
        label_smoothing: Optional smoothing in [0, 1).
    """

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        """Store the reduction mode, the ignore index, and the smoothing factor."""
        raise NotImplementedError("Milestone 6")

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return cross-entropy of ``logits`` against integer class ``targets``.

        Args:
            logits: ``(N, C)`` or ``(N, T, C)`` unnormalised scores.
            targets: ``(N,)`` or ``(N, T)`` integer class indices -- not one-hot.

        **Why fused.** The naive composition ``-log(softmax(x)[target])`` overflows for large
        logits and underflows to ``log(0) = -inf`` for very negative ones. Written as
        ``logsumexp(x) - x[target]`` with the max subtracted inside logsumexp, both hazards
        disappear and the result is exact for logits of any magnitude. This is a genuine
        end-to-end failure mode, not a theoretical one: a model that diverges produces large
        logits, and the unfused loss then returns NaN and destroys weights that were still
        recoverable.

        **The gradient is the reason to fuse.** Composing softmax's Jacobian with NLL's
        gradient and simplifying collapses everything to ``(softmax(logits) - onehot) / N``.
        One subtraction. No Jacobian, no cancellation, and a closed form the test can check
        against the unfused composition to 1e-9.

        Invariants:
            Finite loss and finite gradients for logits of magnitude 1e4.
            Gradient equals ``(softmax(logits) - onehot) / N`` to 1e-9.
            ``ignore_index`` positions contribute nothing to the loss or to the mean's divisor.

        Tests: tests/test_losses.py::test_cross_entropy_extreme_logits,
               tests/test_losses.py::test_cross_entropy_gradient_closed_form
        """
        raise NotImplementedError("Milestone 6")


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """Stable ``log(sum(exp(x)))`` along ``axis``.

    ``m + log(sum(exp(x - m)))`` where ``m = max(x)``. The shift is algebraically the identity
    -- ``exp(m)`` factors out of the sum and its log cancels -- and it guarantees the largest
    exponent is exactly 0, so nothing overflows and at least one term is exactly 1, so the sum
    never underflows to zero either.

    Tests: tests/test_losses.py::test_logsumexp_matches_naive_on_safe_input
    """
    raise NotImplementedError("Milestone 6")

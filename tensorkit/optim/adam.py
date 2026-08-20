"""Adam and AdamW -- Milestone 6.

Concepts: ``docs/concepts/adam-bias-correction.md``.
Tests: ``tests/test_optim.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tensorkit.nn.module import Parameter
from tensorkit.optim.optimizer import Optimizer

__all__ = ["Adam", "AdamW"]


class Adam(Optimizer):
    """Adaptive moment estimation.

    Args:
        params: Parameters or parameter groups.
        lr: Step size.
        betas: ``(beta1, beta2)`` decay rates for the first and second moment estimates.
        eps: Added to the denominator for numerical stability.
        weight_decay: L2 penalty added to the gradient (coupled). Prefer :class:`AdamW`.

    The update::

        m = b1 * m + (1 - b1) * g            # first moment  -- a smoothed gradient
        v = b2 * v + (1 - b2) * g * g        # second moment -- a smoothed squared gradient
        m_hat = m / (1 - b1 ** t)            # bias correction
        v_hat = v / (1 - b2 ** t)
        p -= lr * m_hat / (sqrt(v_hat) + eps)

    **Why the bias correction is not optional.** ``m`` and ``v`` start at zero, so early
    estimates are biased toward zero -- badly. At ``t=1`` with ``b2=0.999``, ``v`` is
    ``0.001 * g^2``: a thousand times too small. Without correction, ``sqrt(v)`` is ~32x too
    small and the first step is ~32x too large. The correction ``1 - b2^t`` exactly cancels
    that: at ``t=1`` it divides by 0.001, recovering ``g^2``. As ``t`` grows the factor tends
    to 1 and the correction fades out on its own.

    Invariants (``SPEC.md`` section 3.4):
        I-OPT-BIAS: ``t`` is **1** on the first update, not 0. At ``t=0`` both correction terms
            are zero and the update divides by zero. A companion test asserts the *uncorrected*
            version fails, so the correction is demonstrably load-bearing rather than cargo.
        I-OPT-STATE: ``m`` and ``v`` are per-parameter, lazily created, shaped like the parameter.
        ``eps`` is added **outside** the square root -- ``sqrt(v) + eps``, not ``sqrt(v + eps)``.
            The two differ meaningfully when ``v`` is tiny, which is exactly when eps matters.

    Tests: tests/test_optim.py::test_adam_first_step_matches_reference,
           tests/test_optim.py::test_adam_without_bias_correction_fails
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Validate the hyperparameters and initialise the base class."""
        raise NotImplementedError("Milestone 6")

    def _step_group(self, group: dict[str, Any]) -> None:
        """Apply the Adam update to every parameter in ``group``."""
        raise NotImplementedError("Milestone 6")


class AdamW(Adam):
    """Adam with **decoupled** weight decay.

    Adam's ``weight_decay`` adds ``wd * p`` to the gradient, so the decay then passes through
    the adaptive ``1/sqrt(v)`` scaling. Parameters with large historical gradients get a small
    ``1/sqrt(v)`` and therefore receive *less* decay -- the opposite of the intent, and the
    reason L2 in Adam does not regularise the way L2 in SGD does.

    AdamW applies ``p -= lr * wd * p`` separately, outside the adaptive scaling::

        p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)

    Same one-line difference that made AdamW the default for transformer training.

    Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019.

    Tests: tests/test_optim.py::test_adamw_decay_is_decoupled
    """

    def _step_group(self, group: dict[str, Any]) -> None:
        """Apply the AdamW update to every parameter in ``group``."""
        raise NotImplementedError("Milestone 6")

"""Dropout -- Milestone 4.

Tests: ``tests/test_layers.py``.
"""

from __future__ import annotations

from tensorkit.nn.module import Module
from tensorkit.tensor import Tensor

__all__ = ["Dropout"]


class Dropout(Module):
    """Inverted dropout.

    Args:
        p: Probability of zeroing an element, in [0, 1).
        seed: Optional seed for reproducible masks in tests.

    **Inverted** means the scaling by ``1 / (1 - p)`` happens at *training* time, so that
    ``eval()`` is exactly the identity -- no scaling, no mask, no branch on the wire. The
    alternative (scale at test time) makes inference depend on a training hyperparameter,
    which is a landmine the moment a checkpoint outlives the config that produced it.

    Invariants:
        ``eval()`` is bitwise the identity.
        In ``train()`` the output expectation equals the input: over 10,000 samples the mean
        ratio is 1.0 within tolerance. This is the test that catches a missing ``1/(1-p)``,
        which otherwise shows up only as a train/eval gap nobody can explain.
        ``p = 0`` is the identity in both modes.
        The mask is resampled every forward call -- a cached mask is a real bug that makes
        dropout behave like a fixed pruning.
        Backward zeroes the gradient at dropped positions and applies the same scale.

    Tests: tests/test_layers.py::test_dropout_expectation_preserved
    """

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        """Store ``p`` and create the RNG."""
        raise NotImplementedError("Milestone 4")

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout in training mode; return ``x`` unchanged in eval mode."""
        raise NotImplementedError("Milestone 4")

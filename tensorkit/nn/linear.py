"""Fully connected layer -- Milestone 4.

Tests: ``tests/test_layers.py``.
"""

from __future__ import annotations

from tensorkit.nn.module import Module
from tensorkit.tensor import Tensor

__all__ = ["Linear"]


class Linear(Module):
    """``y = x @ W + b``.

    Args:
        in_features: Size of the last input axis.
        out_features: Size of the last output axis.
        bias: Whether to learn an additive bias.
        seed: Optional seed for reproducible initialisation.

    Weight shape is ``(in_features, out_features)`` so the forward pass is ``x @ W`` with no
    transpose. PyTorch stores ``(out, in)`` and transposes on every call; either convention is
    fine as long as it is stated, and stating it is the point -- a silent convention mismatch
    is how a ported checkpoint produces garbage.

    Initialisation is Kaiming uniform on the weight (these layers are followed by ReLU-family
    activations far more often than not) and zeros on the bias.

    Invariants:
        Accepts ``(..., in_features)`` -- any number of leading batch axes. NanoLM passes
        ``(B, T, d_model)`` and the layer must not care.
        The bias broadcasts over every leading axis and its gradient unbroadcasts back to
        ``(out_features,)``, which is where an unbroadcast bug shows up first.

    Tests: tests/test_layers.py::test_linear_accepts_arbitrary_leading_axes
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, seed: int | None = None
    ) -> None:
        """Create and initialise the weight and optional bias."""
        raise NotImplementedError("Milestone 4")

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x @ W + b``."""
        raise NotImplementedError("Milestone 4")

    def __repr__(self) -> str:
        """Show in/out features and whether a bias is present."""
        raise NotImplementedError("Milestone 4")

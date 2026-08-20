"""Convolution and pooling -- Milestone 7.

A direct six-nested-loop convolution is correct and unusably slow in Python. im2col trades
memory for speed: unfold every receptive field into a row, so the whole convolution becomes one
matrix multiply, which NumPy hands to BLAS. The unfolded matrix is ``k*k`` times larger than the
input -- that is the trade, and stating it is part of the deliverable.

Concepts: ``docs/concepts/convolution-im2col.md``.
Tests: ``tests/test_conv.py``.
"""

from __future__ import annotations

import numpy as np

from tensorkit.nn.module import Module
from tensorkit.tensor import Tensor

__all__ = ["Conv2d", "MaxPool2d", "Flatten", "im2col", "col2im"]


def im2col(
    x: np.ndarray,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Unfold ``(N, C, H, W)`` into ``(N * H_out * W_out, C * kh * kw)``.

    Each row is one flattened receptive field, so ``cols @ weight_matrix`` is the convolution.

    Two implementations are worth knowing and the tests accept either:
      * ``np.lib.stride_tricks.as_strided`` -- zero-copy view, fastest, and unforgiving: wrong
        strides read arbitrary memory instead of raising.
      * Explicit fancy indexing with precomputed index arrays -- slower, allocates, and far
        easier to get right. Start here, then optimise once the test passes.

    Invariants:
        Output shape is exactly ``(N * H_out * W_out, C * kh * kw)`` (I-CONV-COL).
        Padding contributes zeros, and those zeros must not receive gradient in col2im.

    Complexity: O(N * C * kh * kw * H_out * W_out) time and the same in memory -- the memory
    figure is the one to quote when asked what im2col costs.
    """
    raise NotImplementedError("Milestone 7")


def col2im(
    cols: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Fold ``cols`` back into ``(N, C, H, W)`` -- the backward of :func:`im2col`.

    **This is a scatter-add, not a scatter-assign.** When ``stride < kernel``, receptive fields
    overlap, so one input pixel appears in several rows of ``cols`` and its gradient is the sum
    over all of them. ``out[idx] = vals`` keeps whichever write landed last; ``np.add.at`` (or
    an equivalent accumulate) is required.

    The bug is invisible at ``stride == kernel`` -- no overlap, nothing to accumulate -- which
    is exactly why ``tests/test_conv.py::test_col2im_accumulates_overlaps`` uses ``stride=1``
    with a 3x3 kernel (I-CONV-COL).
    """
    raise NotImplementedError("Milestone 7")


class Conv2d(Module):
    """2-D convolution via im2col.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Int or ``(kh, kw)``.
        stride: Int or ``(sh, sw)``.
        padding: Int or ``(ph, pw)``, zero padding.
        dilation: Int or ``(dh, dw)``.
        bias: Whether to learn a per-output-channel bias.
        seed: Optional seed for reproducible initialisation.

    Note this is cross-correlation, not mathematical convolution -- no kernel flip. Every deep
    learning framework does the same, because a learned kernel absorbs the flip. Knowing that
    the name is wrong is a good interview answer.

    Invariants:
        I-CONV-SHAPE: ``H_out = (H + 2p - d*(k-1) - 1) // s + 1``, asserted against a table.
        Passes float64 gradcheck on a small input.
        Weight shape ``(out_channels, in_channels, kh, kw)``; bias ``(out_channels,)``.

    Tests: tests/test_conv.py::test_conv_output_shape_table
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        """Normalise the int-or-pair arguments and initialise the weight and bias."""
        raise NotImplementedError("Milestone 7")

    def forward(self, x: Tensor) -> Tensor:
        """Convolve ``(N, C_in, H, W)`` to ``(N, C_out, H_out, W_out)``."""
        raise NotImplementedError("Milestone 7")

    @staticmethod
    def output_shape(
        h: int,
        w: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> tuple[int, int]:
        """Return ``(H_out, W_out)`` for the given geometry (I-CONV-SHAPE)."""
        raise NotImplementedError("Milestone 7")


class MaxPool2d(Module):
    """Max pooling over non-overlapping (or overlapping) windows.

    Backward routes the incoming gradient to the **argmax position only**, zero elsewhere.
    Ties break deterministically on the first occurrence. Routing to every tied position
    multiplies the gradient by the tie count -- rare with float inputs, common the moment a
    ReLU has zeroed a whole window.

    Tests: tests/test_conv.py::test_maxpool_backward_routes_to_argmax
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        """Normalise the arguments. ``stride`` defaults to ``kernel_size``."""
        raise NotImplementedError("Milestone 7")

    def forward(self, x: Tensor) -> Tensor:
        """Pool ``(N, C, H, W)`` down to ``(N, C, H_out, W_out)``."""
        raise NotImplementedError("Milestone 7")


class Flatten(Module):
    """Flatten every axis from ``start_dim`` onwards. Backward reshapes back."""

    def __init__(self, start_dim: int = 1) -> None:
        """Store the first axis to flatten."""
        raise NotImplementedError("Milestone 4")

    def forward(self, x: Tensor) -> Tensor:
        """Return the flattened tensor."""
        raise NotImplementedError("Milestone 4")

"""Embedding lookup -- Milestone 4.

Tests: ``tests/test_layers.py``.
"""

from __future__ import annotations

from tensorkit.nn.module import Module
from tensorkit.tensor import Tensor

__all__ = ["Embedding"]


class Embedding(Module):
    """Integer indices to dense vectors.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim: Vector width.
        padding_idx: Optional index whose row stays fixed at zero and receives no gradient.
        seed: Optional seed for reproducible initialisation.

    Forward is a gather: ``weight[indices]``. Backward is the interesting half -- a
    **scatter-add** into a zero array of the weight's shape. Indices repeat constantly (every
    common token appears many times in a batch), and ``grad[idx] = value`` keeps only the last
    write while ``np.add.at(grad, idx, value)`` accumulates. Using assignment gives frequent
    tokens a gradient from exactly one occurrence, which is a slow, quiet degradation rather
    than a crash.

    The backward is also where the "sparse gradient" idea comes from: only the rows that were
    looked up have non-zero gradient, so a dense update over a 50k x 768 table wastes almost
    all of its work. Dense is correct and is what we do; the sparse variant is an INTERVIEW.md
    question.

    Invariants:
        Gradient accumulates over repeated indices.
        ``padding_idx``'s row is zero after initialisation and its gradient stays zero.
        An out-of-range index raises with the offending value, not an opaque IndexError.

    Tests: tests/test_layers.py::test_embedding_backward_accumulates_repeats
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Create and initialise the embedding table."""
        raise NotImplementedError("Milestone 4")

    def forward(self, indices: Tensor) -> Tensor:
        """Return ``weight[indices]``; shape ``(*indices.shape, embedding_dim)``."""
        raise NotImplementedError("Milestone 4")

"""TensorKit: a reverse-mode automatic differentiation engine and neural network library.

NumPy only. If you find yourself reaching for ``import torch`` in this package, the design has
gone wrong -- see ``SPEC.md`` section 1 and ``scripts/check_dependency_purity.py``.

Public surface (frozen at Milestone 6; 02-nanolm depends on it -- see ``SPEC.md`` section 6)::

    from tensorkit import Tensor, no_grad
    from tensorkit.nn import Module, Linear, LayerNorm, Softmax, Dropout, Embedding, Sequential
    from tensorkit.optim import Adam
    from tensorkit.losses import CrossEntropyLoss
"""

from __future__ import annotations

from tensorkit.autograd import backward, no_grad, topological_order
from tensorkit.scalar import Value
from tensorkit.tensor import Tensor

__version__ = "0.1.0"

__all__ = ["Tensor", "Value", "backward", "no_grad", "topological_order", "__version__"]

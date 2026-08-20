"""Neural network layers -- Milestones 4, 5, and 7.

Frozen public surface once M5 lands; 02-nanolm imports from here (``SPEC.md`` section 6).
"""

from __future__ import annotations

from tensorkit.nn.activations import GELU, ReLU, Sigmoid, Softmax, Tanh
from tensorkit.nn.container import ModuleList, Sequential
from tensorkit.nn.conv import Conv2d, Flatten, MaxPool2d
from tensorkit.nn.dropout import Dropout
from tensorkit.nn.embedding import Embedding
from tensorkit.nn.linear import Linear
from tensorkit.nn.module import Module, Parameter
from tensorkit.nn.norm import BatchNorm1d, BatchNorm2d, LayerNorm

__all__ = [
    "Module",
    "Parameter",
    "Sequential",
    "ModuleList",
    "Linear",
    "ReLU",
    "GELU",
    "Softmax",
    "Tanh",
    "Sigmoid",
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "MaxPool2d",
    "Flatten",
    "Dropout",
    "Embedding",
]

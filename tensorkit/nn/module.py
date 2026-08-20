"""The Module base class -- Milestone 4.

``Module`` owns three things: parameter discovery, mode propagation, and serialisation. Each is
small; each has a failure mode that costs an afternoon.

Concepts: ``docs/concepts/module-system.md``.
Tests: ``tests/test_module.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from tensorkit.tensor import Tensor

__all__ = ["Module", "Parameter"]


class Parameter(Tensor):
    """A Tensor that is a learnable parameter.

    The only difference from a Tensor is intent, made checkable: ``requires_grad`` defaults to
    True and :meth:`Module.parameters` collects exactly these. Buffers -- BatchNorm's running
    statistics -- are plain Tensors and are deliberately invisible to the optimiser
    (I-MOD-BUFFER).
    """

    def __init__(self, data: np.ndarray) -> None:
        """Wrap an array as a learnable parameter."""
        raise NotImplementedError("Milestone 4")


class Module:
    """Base class for every layer and container.

    Subclasses implement :meth:`forward`. ``__call__`` dispatches to it, so a layer is used as
    ``layer(x)`` and a subclass never overrides ``__call__`` -- that hook is where shape
    checking and future instrumentation live.

    Invariants (``SPEC.md`` section 3.3):
        I-MOD-UNIQUE: :meth:`parameters` yields each parameter object exactly once.
        I-MOD-MODE: :meth:`train`/:meth:`eval` propagate to every submodule recursively.
        I-MOD-ZERO: :meth:`zero_grad` sets ``.grad = None``, not ``zeros_like``.
        I-MOD-BUFFER: buffers appear in :meth:`state_dict` but never in :meth:`parameters`.
    """

    def __init__(self) -> None:
        """Initialise the parameter, buffer, and submodule registries and the training flag."""
        raise NotImplementedError("Milestone 4")

    # -- the forward hook ----------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the layer's output. Every subclass overrides this."""
        raise NotImplementedError("subclasses must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """Dispatch to :meth:`forward`. Do not override -- override ``forward`` instead."""
        raise NotImplementedError("Milestone 4")

    # -- registration --------------------------------------------------------

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Register a learnable parameter under ``name``."""
        raise NotImplementedError("Milestone 4")

    def register_buffer(self, name: str, tensor: Tensor | None) -> None:
        """Register a non-learnable persistent tensor under ``name``.

        Buffers are saved and loaded with the model but never handed to the optimiser. Getting
        this wrong means the optimiser applies momentum to BatchNorm's running mean, which is
        a wonderfully confusing bug: the model trains, and evaluation drifts.
        """
        raise NotImplementedError("Milestone 4")

    def add_module(self, name: str, module: Module | None) -> None:
        """Register a submodule under ``name``."""
        raise NotImplementedError("Milestone 4")

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-register Parameters, Tensors, and Modules assigned as attributes.

        This is the ergonomic bit that makes ``self.weight = Parameter(...)`` work without an
        explicit registration call. Requires care: ``__init__`` must set up the registries
        *before* any other attribute assignment, or the first assignment recurses into an
        attribute that does not exist yet.

        Tests: tests/test_module.py::test_attribute_assignment_registers_parameters
        """
        raise NotImplementedError("Milestone 4")

    # -- traversal -----------------------------------------------------------

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Yield every learnable parameter, each exactly once.

        Deduplicate by ``id()``. Weight tying -- the same ``Linear`` referenced from two
        attributes, or NanoLM's embedding shared with its LM head -- otherwise yields the same
        object twice, and the optimiser then applies its update twice per step. That looks like
        a learning rate that is mysteriously 2x too high on exactly one tensor (I-MOD-UNIQUE).

        Tests: tests/test_module.py::test_weight_tying_yields_parameter_once
        """
        raise NotImplementedError("Milestone 4")

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Yield ``(dotted_name, parameter)`` pairs.

        Under weight tying one object has several names. Yield the first encountered and note
        the aliasing rather than yielding it twice -- consistency with :meth:`parameters` is
        what matters.
        """
        raise NotImplementedError("Milestone 4")

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Yield every registered buffer."""
        raise NotImplementedError("Milestone 4")

    def modules(self) -> Iterator[Module]:
        """Yield self and every submodule, depth-first."""
        raise NotImplementedError("Milestone 4")

    # -- mode ----------------------------------------------------------------

    def train(self, mode: bool = True) -> Module:
        """Set training mode recursively. Returns self so calls chain.

        Only ``Dropout`` and ``BatchNorm`` read this flag, which is exactly why forgetting to
        call ``eval()`` is so hard to spot: the model still produces plausible predictions,
        just noisier and with the wrong normalisation statistics (I-MOD-MODE).
        """
        raise NotImplementedError("Milestone 4")

    def eval(self) -> Module:
        """Set evaluation mode recursively. Equivalent to ``train(False)``."""
        raise NotImplementedError("Milestone 4")

    @property
    def training(self) -> bool:
        """Whether this module is in training mode."""
        raise NotImplementedError("Milestone 4")

    # -- gradients -----------------------------------------------------------

    def zero_grad(self) -> None:
        """Set every parameter's ``.grad`` to None (I-MOD-ZERO)."""
        raise NotImplementedError("Milestone 4")

    # -- serialisation -------------------------------------------------------

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Return a flat ``{dotted_name: array}`` mapping of parameters **and** buffers.

        Arrays, not Tensors: a checkpoint should not carry graph structure, and
        ``np.savez(**state_dict)`` should just work.
        """
        raise NotImplementedError("Milestone 4")

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        """Load parameters and buffers from a state dict.

        With ``strict=True``, missing or unexpected keys raise, and the error lists both sets.
        Silently ignoring a missing key gives you a model with a randomly-initialised layer and
        a mystery about why the reloaded checkpoint scores worse than the one you saved.

        Tests: tests/test_module.py::test_state_dict_roundtrip
        """
        raise NotImplementedError("Milestone 4")

    def __repr__(self) -> str:
        """Return an indented tree of submodules, the way PyTorch prints a model."""
        raise NotImplementedError("Milestone 4")

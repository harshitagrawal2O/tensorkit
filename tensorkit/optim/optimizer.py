"""Optimiser base class -- Milestone 6.

Tests: ``tests/test_optim.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tensorkit.nn.module import Parameter

__all__ = ["Optimizer"]


class Optimizer:
    """Base class holding parameter groups and per-parameter state.

    Args:
        params: The parameters to optimise, or a list of parameter-group dicts.
        defaults: Default hyperparameters, overridable per group.

    Parameter groups exist so different parts of a model can take different hyperparameters --
    no weight decay on biases and normalisation gains, a lower learning rate on embeddings.
    Applying weight decay to a LayerNorm gain drags it toward zero, which scales the layer's
    output toward zero; it is a real and commonly-shipped bug.

    Invariants (``SPEC.md`` section 3.4):
        I-OPT-STATE: state is keyed by ``id(param)``, created lazily, shaped like the parameter.
        I-OPT-NOGRAD: :meth:`step` runs under ``no_grad()``. Without it the update itself is
            recorded onto the tape and the graph grows every step until memory runs out.
        A parameter whose ``.grad`` is None is skipped, not treated as zero -- treating it as
        zero still applies weight decay and momentum to a parameter that took no part in the
        forward pass.
    """

    def __init__(self, params: Iterable[Parameter], defaults: dict[str, Any]) -> None:
        """Normalise ``params`` into parameter groups and store the defaults."""
        raise NotImplementedError("Milestone 6")

    def zero_grad(self) -> None:
        """Set every parameter's ``.grad`` to None."""
        raise NotImplementedError("Milestone 6")

    def step(self) -> None:
        """Apply one update. Subclasses implement :meth:`_step_group`."""
        raise NotImplementedError("Milestone 6")

    def _step_group(self, group: dict[str, Any]) -> None:
        """Update every parameter in one group. Implemented by subclasses."""
        raise NotImplementedError("subclasses must implement _step_group()")

    def state_dict(self) -> dict[str, Any]:
        """Return optimiser state for checkpointing.

        Resuming training without the optimiser state restarts momentum and Adam's moment
        estimates from zero, which produces a visible loss spike at the resume point. That
        spike is the symptom to recognise.
        """
        raise NotImplementedError("Milestone 6")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimiser state saved by :meth:`state_dict`."""
        raise NotImplementedError("Milestone 6")

    @staticmethod
    def clip_grad_norm(params: Iterable[Parameter], max_norm: float) -> float:
        """Scale gradients so their **global** L2 norm is at most ``max_norm``.

        Returns:
            The total norm before clipping -- log it; a sudden spike is the earliest warning
            that a training run is about to diverge.

        Global, not per-tensor. Per-tensor clipping rescales each tensor independently and so
        changes the *direction* of the overall update; global clipping only changes its length.
        Same name, different algorithm, different training dynamics (NanoLM's I-TRAIN-CLIP).

        Tests: tests/test_optim.py::test_clip_grad_norm_is_global
        """
        raise NotImplementedError("Milestone 6")

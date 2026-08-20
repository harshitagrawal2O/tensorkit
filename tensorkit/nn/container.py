"""Module containers -- Milestone 4.

Tests: ``tests/test_module.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

from tensorkit.nn.module import Module
from tensorkit.tensor import Tensor

__all__ = ["Sequential", "ModuleList"]


class Sequential(Module):
    """Chain modules, calling each on the previous one's output.

    Children must be registered as submodules, not stored in a plain list: a plain list is
    invisible to :meth:`Module.parameters`, so the optimiser silently trains nothing and the
    loss curve is flat. That failure is the reason ``ModuleList`` exists at all.

    Tests: tests/test_module.py::test_sequential_registers_children
    """

    def __init__(self, *modules: Module) -> None:
        """Register each module in order."""
        raise NotImplementedError("Milestone 4")

    def forward(self, x: Tensor) -> Tensor:
        """Apply each module in order."""
        raise NotImplementedError("Milestone 4")

    def __getitem__(self, index: int) -> Module:
        """Return the module at ``index``."""
        raise NotImplementedError("Milestone 4")

    def __len__(self) -> int:
        """Number of chained modules."""
        raise NotImplementedError("Milestone 4")


class ModuleList(Module):
    """A list of modules that is visible to parameter discovery.

    No ``forward``: the caller decides how to use the children. NanoLM's transformer stack is
    the motivating case -- N blocks applied in a loop with a residual around each.
    """

    def __init__(self, modules: list[Module] | None = None) -> None:
        """Register each module under its index."""
        raise NotImplementedError("Milestone 4")

    def append(self, module: Module) -> ModuleList:
        """Register and append a module. Returns self so calls chain."""
        raise NotImplementedError("Milestone 4")

    def __getitem__(self, index: int) -> Module:
        """Return the module at ``index``."""
        raise NotImplementedError("Milestone 4")

    def __iter__(self) -> Iterator[Module]:
        """Iterate the modules in registration order."""
        raise NotImplementedError("Milestone 4")

    def __len__(self) -> int:
        """Number of modules held."""
        raise NotImplementedError("Milestone 4")

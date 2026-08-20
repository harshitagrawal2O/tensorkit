"""The gradcheck fixture table — instructor-owned, part of the rubric.

Each entry says how to exercise one primitive: the shapes, whether the inputs must be kept
positive (``log``, ``sqrt``, division denominators) or away from a kink (``relu``, ``abs``,
``max``), and what to call it. The sweep in ``tests/test_gradcheck.py`` and the
``tensorkit gradcheck`` CLI both read this table, so there is exactly one definition of what
"every primitive passes gradcheck" means.

It lives in ``rubric/`` rather than beside the tests because a student may not edit it. Loosening
a tolerance or shrinking a shape here would make the sweep pass without the gradients being right.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tensorkit.tensor import Tensor

__all__ = ["PrimitiveCase", "PRIMITIVE_CASES", "build_inputs", "check_primitive"]


@dataclass(frozen=True)
class PrimitiveCase:
    """How to gradcheck one primitive.

    Attributes:
        name: Must match the key in ``tensorkit.ops.PRIMITIVES``.
        fn: Called with the built input tensors; returns a Tensor.
        shapes: One shape per Tensor argument. Ragged on purpose where broadcasting applies --
            equal shapes make the unbroadcast an identity and hide the bug it exists to catch.
        positive: Keep inputs in [0.5, 2.5]. For log, sqrt, and division denominators.
        kink_free: False for relu/abs/max. Gradcheck then warns when an input lands within eps
            of the non-differentiable point, where the numerical gradient is meaningless rather
            than merely imprecise.
        offset: Added to every input, to push relu-family inputs off the kink.
        milestone: Which milestone introduces it.
        note: Why this case is shaped the way it is, for the failure message.
    """

    name: str
    fn: Callable[..., Tensor]
    shapes: tuple[tuple[int, ...], ...]
    positive: bool = False
    kink_free: bool = True
    offset: float = 0.0
    milestone: int = 2
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _concat(a: Tensor, b: Tensor) -> Tensor:
    """Concatenate two tensors. A named helper so the table stays readable."""
    from tensorkit.tensor import Tensor as TensorType

    return TensorType.concat([a, b], axis=0)


def _softmax(a: Tensor) -> Tensor:
    """Softmax over the last axis, via the ops registry."""
    from tensorkit.ops import softmax

    return softmax(a, axis=-1)


def _cases() -> tuple[PrimitiveCase, ...]:
    """Build the table lazily, so importing this module never touches an unfinished Tensor."""
    return (
        PrimitiveCase(
            "add",
            lambda a, b: a + b,
            ((4, 3), (3,)),
            note="ragged shapes so the unbroadcast is exercised, not skipped",
        ),
        PrimitiveCase(
            "sub",
            lambda a, b: a - b,
            ((4, 3), (1, 3)),
            note="the right operand's gradient must be negated AND unbroadcast",
        ),
        PrimitiveCase(
            "mul",
            lambda a, b: a * b,
            ((2, 3, 4), (4,)),
            note="multiply before unbroadcasting, not after",
        ),
        PrimitiveCase(
            "div",
            lambda a, b: a / b,
            ((4, 3), (3,)),
            positive=True,
            note="the quotient rule has two terms; the denominator's is the one people drop",
        ),
        PrimitiveCase(
            "pow",
            lambda a: a**3,
            ((3, 4),),
            positive=True,
            note="constant exponent only",
        ),
        PrimitiveCase(
            "neg",
            lambda a: -a,
            ((3, 4),),
        ),
        PrimitiveCase(
            "matmul",
            lambda a, b: a @ b,
            ((2, 3, 4), (4, 5)),
            note="batched left operand; the right one must unbroadcast back over the batch",
        ),
        PrimitiveCase(
            "sum",
            lambda a: a.sum(axis=1),
            ((3, 4, 2),),
            note="keepdims=False, so the backward pass must reinsert the collapsed axis",
        ),
        PrimitiveCase(
            "mean",
            lambda a: a.mean(axis=(0, 2)),
            ((3, 4, 2),),
        ),
        PrimitiveCase(
            "max",
            lambda a: a.max(axis=1),
            ((3, 5),),
            kink_free=False,
            note="ties are broken to the first occurrence; inputs are drawn continuously so "
            "exact ties do not arise",
        ),
        PrimitiveCase(
            "exp",
            lambda a: a.exp(),
            ((3, 4),),
            note="inputs stay small: exp(30) in float64 is fine, but its gradient dominates "
            "the relative-error denominator and makes the check meaningless",
        ),
        PrimitiveCase(
            "log",
            lambda a: a.log(),
            ((3, 4),),
            positive=True,
        ),
        PrimitiveCase(
            "sqrt",
            lambda a: a.sqrt(),
            ((3, 4),),
            positive=True,
            note="singular at 0, so inputs are bounded away from it",
        ),
        PrimitiveCase(
            "tanh",
            lambda a: a.tanh(),
            ((3, 4),),
        ),
        PrimitiveCase(
            "relu",
            lambda a: a.relu(),
            ((3, 4),),
            kink_free=False,
            offset=1.5,
            note="offset pushes inputs off the kink at 0, where no gradient exists to check",
        ),
        PrimitiveCase(
            "abs",
            lambda a: a.abs(),
            ((3, 4),),
            kink_free=False,
            offset=2.0,
        ),
        PrimitiveCase(
            "reshape",
            lambda a: a.reshape(6, 4),
            ((3, 4, 2),),
        ),
        PrimitiveCase(
            "transpose",
            lambda a: a.transpose(2, 0, 1),
            ((3, 4, 2),),
            note="a non-symmetric permutation, so applying it twice instead of inverting it fails",
        ),
        PrimitiveCase(
            "getitem",
            lambda a: a[[0, 0, 2]],
            ((4, 3),),
            note="a repeated index, so scatter-assign fails where scatter-add passes",
        ),
        PrimitiveCase(
            "concat",
            _concat,
            ((2, 3), (4, 3)),
        ),
        PrimitiveCase(
            "softmax",
            _softmax,
            ((3, 5),),
            milestone=4,
            note="the closed-form VJP s * (g - sum(g*s)); the full Jacobian is never built",
        ),
    )


#: The table. Built on first access so an unfinished ``tensorkit.tensor`` does not break import.
PRIMITIVE_CASES: dict[str, PrimitiveCase] = {}


def _ensure_loaded() -> dict[str, PrimitiveCase]:
    """Populate :data:`PRIMITIVE_CASES` on first use."""
    if not PRIMITIVE_CASES:
        for case in _cases():
            PRIMITIVE_CASES[case.name] = case
    return PRIMITIVE_CASES


def build_inputs(case: PrimitiveCase, seed: int = 20250820) -> list[Tensor]:
    """Construct the float64 inputs for one case.

    float64 always: central differences in float32 have a relative error floor around 1e-3, so
    a float32 sweep can only assert a tolerance loose enough to pass a broken implementation.
    """
    from tensorkit.tensor import Tensor

    rng = np.random.default_rng(seed)
    inputs: list[Tensor] = []
    for shape in case.shapes:
        if case.positive:
            data = rng.uniform(0.5, 2.5, size=shape)
        else:
            data = rng.standard_normal(shape) + case.offset
        inputs.append(Tensor(data.astype(np.float64), requires_grad=True, dtype=np.float64))
    return inputs


def check_primitive(name: str, *, tol: float = 1e-6, seed: int = 20250820) -> float:
    """Gradcheck one primitive. Returns its maximum relative error.

    Raises:
        KeyError: if the name is not in the table.
        AssertionError: if any argument's gradient is outside ``tol``.
    """
    from tensorkit.gradcheck import gradcheck_or_raise

    cases = _ensure_loaded()
    if name not in cases:
        raise KeyError(f"{name!r} has no gradcheck case; add one to rubric/tensorkit_primitives.py")

    case = cases[name]
    inputs = build_inputs(case, seed=seed)
    return gradcheck_or_raise(case.fn, inputs, tol=tol, kink_free=case.kink_free)


def cases_for_milestone(milestone: int | None = None) -> dict[str, PrimitiveCase]:
    """Return the table, optionally filtered to one milestone."""
    cases = _ensure_loaded()
    if milestone is None:
        return dict(cases)
    return {k: v for k, v in cases.items() if v.milestone == milestone}

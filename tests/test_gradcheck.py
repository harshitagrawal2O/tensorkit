"""Milestone 3 — numerical gradient checking as the correctness spine.

Milestones 1 and 2 assert gradients against derivatives worked out by hand. That only scales as
far as your patience. This milestone replaces hand-checking with an oracle: central differences,
in float64, applied to every registered primitive.

The oracle itself is given (``tensorkit/gradcheck.py``) and its fixtures are in the rubric. What
is being graded is whether your gradient rules survive it — and, in one test, whether the oracle
has teeth at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from rubric import assert_raises_clearly, invariant
from rubric.tensorkit_primitives import build_inputs, cases_for_milestone
from tensorkit.gradcheck import DEFAULT_TOL, gradcheck, gradcheck_or_raise
from tensorkit.tensor import Tensor

pytestmark = pytest.mark.m3

SPEC = "01-tensorkit/SPEC.md section 4, M3"

I_GRADCHECK = invariant(
    "I-GRADCHECK",
    "every registered primitive's analytical VJP matches central differences to 1e-6 in float64",
    "This is the only mechanised check that a gradient rule is right. A rule that is wrong but "
    "plausible -- a missing quotient-rule term, a mean where a sum belongs -- produces a model "
    "that trains, just worse, and nothing else in the suite will tell you.",
    SPEC,
)
I_ORACLE = invariant(
    "I-ORACLE",
    "gradcheck fails on a deliberately broken gradient",
    "A test oracle that has never been shown to fail is not evidence. If gradcheck cannot catch "
    "a planted bug, every green result it has ever produced means nothing.",
    SPEC,
)
I_FLOAT64 = invariant(
    "I-FLOAT64",
    "gradcheck refuses float32 input rather than checking it loosely",
    "Central differences in float32 have a relative error floor near 1e-3. Running the sweep in "
    "float32 sends you hunting for a gradient bug that does not exist -- or forces a tolerance "
    "so loose that a real one walks through it.",
    SPEC,
)
I_COVERAGE = invariant(
    "I-COVERAGE",
    "every primitive in tensorkit.ops.PRIMITIVES has a gradcheck case",
    "The sweep is only as good as its coverage. Without this check the registry grows and the "
    "sweep quietly does not, so new ops arrive unverified.",
    SPEC,
)


def _milestone_cases():
    """Every case introduced at or before M3 — the ones in scope for this milestone."""
    cases = {}
    for m in (2, 3):
        cases.update(cases_for_milestone(m))
    return cases


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_milestone_cases()))
def test_primitive_passes_gradcheck(name, record_property):
    """Each primitive, against the oracle, in float64."""
    case = _milestone_cases()[name]
    inputs = build_inputs(case)

    try:
        max_error = gradcheck_or_raise(case.fn, inputs, tol=DEFAULT_TOL, kink_free=case.kink_free)
    except AssertionError as exc:
        hint = f"\n  case note: {case.note}" if case.note else ""
        I_GRADCHECK.check(False, f"{name}: {exc}{hint}")
        raise

    # Recorded so the grading harness can pull the worst error per primitive into BENCHMARKS.md
    # without re-running the sweep.
    record_property("max_rel_error", max_error)
    I_GRADCHECK.check(
        max_error <= DEFAULT_TOL,
        f"{name} reported max relative error {max_error:.3e} > {DEFAULT_TOL:.1e}",
    )


def test_sweep_reports_the_worst_error_across_all_primitives(record_property):
    """The headline number for BENCHMARKS.md: max relative error over the whole library."""
    worst_name, worst_error = "", 0.0
    for name, case in sorted(_milestone_cases().items()):
        error = gradcheck_or_raise(case.fn, build_inputs(case), kink_free=case.kink_free)
        if error > worst_error:
            worst_name, worst_error = name, error

    record_property("worst_primitive", worst_name)
    record_property("worst_max_rel_error", worst_error)
    I_GRADCHECK.check(
        worst_error <= DEFAULT_TOL,
        f"worst primitive was {worst_name!r} at {worst_error:.3e}",
    )


def test_every_primitive_is_checked():
    """The registry and the fixture table must not drift apart."""
    from tensorkit.ops import PRIMITIVES

    registered = {name for name, p in PRIMITIVES.items() if p.milestone <= 3}
    covered = set(_milestone_cases())

    missing = sorted(registered - covered)
    I_COVERAGE.check(
        not missing,
        f"registered in tensorkit.ops but absent from the gradcheck table: {missing}. "
        f"Add a PrimitiveCase in rubric/tensorkit_primitives.py, or the op ships unverified.",
    )


# ---------------------------------------------------------------------------
# Does the oracle have teeth?
# ---------------------------------------------------------------------------


def _half_gradient_div(a: Tensor, b: Tensor) -> Tensor:
    """``a / b`` with the denominator's gradient deliberately halved.

    The bug is planted by hand, on the tape, rather than by severing the path with ``detach()``.
    That distinction matters and an earlier draft of this file got it wrong: a severed path
    leaves ``b.grad is None``, which gradcheck reports as "no gradient at all" -- a *different*
    diagnosis from "this number is wrong", and the one
    ``test_gradcheck_reports_a_missing_gradient_distinctly`` exists to pin. Planting a wrong
    *value* keeps the two failure modes distinguishable, which is the whole point of having
    both tests.

    Halved rather than dropped, because a factor of 2 is what the ratio hints in
    ``GradCheckResult.report`` are for, so this exercises the diagnosis as well as the detection.
    """
    out = Tensor(
        a.numpy() / b.numpy(),
        requires_grad=True,
        dtype=np.float64,
        _children=(a, b),
        _op="half_gradient_div",
    )

    def _backward() -> None:
        grad = out.grad
        a.accumulate_grad(grad / b.numpy())
        b.accumulate_grad(-0.5 * grad * a.numpy() / (b.numpy() ** 2))

    out._backward = _backward
    return out


def test_gradcheck_catches_a_planted_bug():
    """Halve the denominator's gradient and gradcheck must notice.

    ``d(a/b)/db = -a/b^2``. A ``div`` backward that gets the denominator's term wrong is a real,
    common bug: the forward pass is exact, the loss still decreases, and one operand simply
    learns at the wrong rate. If the oracle lets this through, it lets everything through, and
    every green result it has ever produced means nothing.
    """
    rng = np.random.default_rng(7)
    a = Tensor(rng.uniform(0.5, 2.5, (3, 4)), requires_grad=True, dtype=np.float64)
    b = Tensor(rng.uniform(0.5, 2.5, (3, 4)), requires_grad=True, dtype=np.float64)

    with pytest.raises(AssertionError) as caught:
        gradcheck_or_raise(_half_gradient_div, [a, b], argnums=[1])

    message = str(caught.value)
    I_ORACLE.check(
        "gradcheck" in message.lower() or "relative error" in message.lower(),
        f"gradcheck failed, but its message does not explain what went wrong: {message[:200]!r}",
    )
    I_ORACLE.check(
        "no gradient" not in message.lower(),
        "gradcheck reported 'no gradient at all' for an input that received a wrong one. Those "
        "are different diagnoses and conflating them sends the reader to the wrong place.",
    )


def test_gradcheck_rejects_float32():
    rng = np.random.default_rng(1)
    x = Tensor(rng.standard_normal((3, 3)).astype(np.float32), requires_grad=True)

    with assert_raises_clearly(TypeError, "float64"):
        gradcheck(lambda t: t * 2.0, [x])


def test_gradcheck_reports_a_missing_gradient_distinctly():
    """A rule that never fires gives ``grad is None``, not a wrong number.

    Worth its own message: "no gradient at all" points at requires_grad or no_grad, while
    "wrong by 2x" points at the rule itself. Conflating them wastes an afternoon.
    """
    x = Tensor(np.ones((2, 2)), requires_grad=True, dtype=np.float64)

    with pytest.raises(AssertionError) as caught:
        gradcheck_or_raise(lambda t: Tensor(t.numpy() * 2.0, dtype=np.float64), [x])

    assert "no gradient" in str(caught.value).lower(), (
        "when an input receives no gradient at all, say so explicitly rather than reporting a "
        f"numerical mismatch. Got: {str(caught.value)[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Properties of the oracle itself, so its numbers can be trusted
# ---------------------------------------------------------------------------


def test_numerical_gradient_of_a_known_function():
    """Sanity-check the oracle against a derivative anyone can do in their head.

    ``f(x) = sum(x^3)`` has ``df/dx = 3x^2``. If this drifts, every other number in the
    milestone is suspect, so it is checked directly rather than assumed.
    """
    from tensorkit.gradcheck import numerical_gradient

    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True, dtype=np.float64)
    numeric = numerical_gradient(lambda t: t**3, [x])

    I_GRADCHECK.all_close(
        numeric,
        3 * np.array([1.0, 2.0, 3.0]) ** 2,
        rtol=1e-6,
        detail="central differences on sum(x^3)",
    )


def test_gradcheck_result_names_the_worst_element():
    """The failure message has to be navigable, not just correct."""
    rng = np.random.default_rng(3)
    a = Tensor(rng.uniform(0.5, 2.5, (3, 4)), requires_grad=True, dtype=np.float64)
    b = Tensor(rng.uniform(0.5, 2.5, (3, 4)), requires_grad=True, dtype=np.float64)

    results = gradcheck(lambda x, y: x / y.detach(), [a, b], argnums=[1])
    report = results[0].report()

    for token in ("worst element", "relative error", "analytical", "numerical"):
        assert token in report.lower(), (
            f"the gradcheck report omits {token!r}. Without it the reader knows a gradient is "
            f"wrong but not which one or by how much.\n{report}"
        )

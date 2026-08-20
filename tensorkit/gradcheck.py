"""Numerical gradient checking -- GIVEN, fully implemented.

This is the oracle the whole library is graded against, so it is written for you: an oracle you
wrote yourself and got subtly wrong is worse than no oracle at all.

**Why central differences and not forward differences.** The forward difference
``(f(x+h) - f(x)) / h`` has truncation error O(h). The central difference
``(f(x+h) - f(x-h)) / 2h`` cancels the second-order term of the Taylor expansion and has
truncation error O(h**2), for one extra function evaluation. That is a straight trade of 2x
cost for roughly six orders of magnitude of accuracy at typical step sizes.

**Why the step size is what it is.** Two errors pull in opposite directions:

* truncation error grows as O(h**2) -- larger steps approximate the limit worse;
* round-off error grows as O(eps / h) -- smaller steps subtract two nearly-equal floats and
  cancellation eats the significant digits.

The total is minimised near ``h = eps**(1/3)``: about 6e-6 in float64 (eps ~ 2.2e-16) and about
5e-3 in float32 (eps ~ 1.2e-7). float32's optimum error floor is ~1e-3 relative, which is above
any tolerance worth asserting. **Gradient checking therefore runs in float64. Always.**

**Why relative error and not absolute.** A gradient of 1e-8 and one of 1e8 are both perfectly
ordinary. The denominator ``max(|a|, |b|, eps)`` makes the comparison scale-free while avoiding
a division by zero when both gradients are legitimately zero.

Concepts: ``docs/concepts/gradcheck.md``.
Tests: ``tests/test_gradcheck.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from tensorkit.tensor import Tensor

__all__ = ["GradCheckResult", "numerical_gradient", "gradcheck", "gradcheck_or_raise"]

#: Near-optimal central-difference step for float64: cube root of machine epsilon.
DEFAULT_EPS: float = float(np.cbrt(np.finfo(np.float64).eps))

#: Default relative tolerance. Achievable in float64; not achievable in float32.
DEFAULT_TOL: float = 1e-6


@dataclass
class GradCheckResult:
    """Outcome of one gradient check.

    Attributes:
        passed: Whether every element was within tolerance.
        max_rel_error: Largest relative error observed. This number goes into BENCHMARKS.md.
        max_abs_error: Largest absolute error observed.
        worst_index: Flat index of the worst element, as a tuple of per-axis indices.
        analytical: The gradient the library produced.
        numerical: The gradient central differences produced.
        argument: Which input argument this result is for.
    """

    passed: bool
    max_rel_error: float
    max_abs_error: float
    worst_index: tuple[int, ...]
    analytical: np.ndarray
    numerical: np.ndarray
    argument: int = 0
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Return a failure message that names the invariant and points at the worst element."""
        status = "PASSED" if self.passed else "FAILED"
        i = self.worst_index
        return (
            f"gradcheck {status} for argument {self.argument}\n"
            f"  max relative error : {self.max_rel_error:.3e} (tolerance {DEFAULT_TOL:.1e})\n"
            f"  max absolute error : {self.max_abs_error:.3e}\n"
            f"  worst element      : index {i}\n"
            f"    analytical = {self.analytical[i]!r}\n"
            f"    numerical  = {self.numerical[i]!r}\n"
            + ("".join(f"  note: {n}\n" for n in self.notes))
            + (
                ""
                if self.passed
                else "\n  Invariant: the analytical VJP must equal the central-difference\n"
                "  gradient. A mismatch means a local gradient rule in tensorkit/ops.py is\n"
                "  wrong, or an unbroadcast is missing. Ratios worth recognising:\n"
                "    ~2        -> a term is counted twice\n"
                "    <1        -> += was written as =, so a shared subgraph kept one\n"
                "                 path instead of the sum of them (0.5 when symmetric)\n"
                "    ~1/N      -> a mean where a sum belongs (unbroadcast averaging)\n"
                "    ~N        -> a sum where a mean belongs\n"
                "    ~0        -> the rule never fires; check the op is on the tape\n"
                "    sign flip -> an operand order swap in a non-commutative rule\n"
            )
        )


def numerical_gradient(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    argnum: int = 0,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Estimate d(fn(inputs).sum()) / d(inputs[argnum]) by central differences.

    ``fn`` must return a Tensor; its ``.sum()`` is the scalar differentiated, which makes the
    numerical gradient directly comparable to a backward pass seeded with ones.

    Args:
        fn: Function under test.
        inputs: Its Tensor arguments.
        argnum: Which argument to differentiate with respect to.
        eps: Perturbation size. Leave at the default unless you know why you are changing it.

    Returns:
        An array shaped like ``inputs[argnum].data``.

    Complexity: two forward passes per element. Only ever run this on small tensors -- that is
    the whole reason it is a test oracle and not the engine.
    """
    target = inputs[argnum]
    base = np.array(target.data, dtype=np.float64, copy=True)
    grad = np.zeros_like(base)

    # np.ndindex rather than np.nditer: same traversal, no mutable iterator state, and it
    # handles the 0-d case (one empty index) without a special branch.
    for idx in np.ndindex(*base.shape):
        original = float(base[idx])

        base[idx] = original + eps
        plus = _evaluate(fn, inputs, argnum, base)

        base[idx] = original - eps
        minus = _evaluate(fn, inputs, argnum, base)

        base[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)

    return grad


def _evaluate(
    fn: Callable[..., Tensor], inputs: Sequence[Tensor], argnum: int, data: np.ndarray
) -> float:
    """Run ``fn`` with ``inputs[argnum].data`` replaced by ``data``; return the scalar sum."""
    perturbed = list(inputs)
    perturbed[argnum] = Tensor(np.array(data, copy=True), dtype=np.float64)
    out = fn(*perturbed)
    return float(np.sum(out.data, dtype=np.float64))


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    eps: float = DEFAULT_EPS,
    tol: float = DEFAULT_TOL,
    argnums: Sequence[int] | None = None,
    kink_free: bool = True,
) -> list[GradCheckResult]:
    """Compare analytical gradients against central differences for each argument.

    Args:
        fn: Function under test. Must be differentiable at every input point.
        inputs: Tensor arguments, all ``float64``, all ``requires_grad=True``.
        eps: Central-difference step.
        tol: Relative-error tolerance.
        argnums: Which arguments to check. Defaults to all of them.
        kink_free: Set False for ops with a non-differentiable point (relu, abs, max). The
            check then warns when an input sits within ``eps`` of a kink, where the numerical
            gradient straddles the discontinuity and is *meaningless* rather than merely
            inaccurate -- a distinction worth having in the failure message.

    Returns:
        One :class:`GradCheckResult` per checked argument.

    Raises:
        TypeError: if any input is not float64. This is not pedantry: float32 cannot reach the
            default tolerance no matter how correct your gradients are, and the resulting
            failures send you hunting for a bug that does not exist.
    """
    for i, t in enumerate(inputs):
        if t.data.dtype != np.float64:
            raise TypeError(
                f"gradcheck input {i} has dtype {t.data.dtype}, expected float64. "
                "Central differences in float32 have a relative error floor around 1e-3, "
                "which is above the tolerance being asserted -- see docs/concepts/gradcheck.md."
            )

    indices = list(range(len(inputs))) if argnums is None else list(argnums)
    results: list[GradCheckResult] = []

    for argnum in indices:
        for t in inputs:
            t.zero_grad()

        out = fn(*inputs)
        out.sum().backward()

        analytical = inputs[argnum].grad
        if analytical is None:
            raise AssertionError(
                f"argument {argnum} received no gradient at all. Either the op never recorded "
                f"onto the tape (check requires_grad and no_grad), or its _backward closure "
                f"does not touch this input. Invariant I-ACCUM."
            )

        numerical = numerical_gradient(fn, inputs, argnum, eps)

        notes: list[str] = []
        if not kink_free:
            near = np.abs(inputs[argnum].data) < (10.0 * eps)
            if bool(np.any(near)):
                notes.append(
                    f"{int(np.sum(near))} input element(s) lie within 10*eps of 0, where this "
                    "op is not differentiable. The numerical gradient there is meaningless, "
                    "not merely imprecise -- perturb the test inputs away from the kink."
                )

        abs_err = np.abs(analytical - numerical)
        denom = np.maximum(np.maximum(np.abs(analytical), np.abs(numerical)), 1e-12)
        rel_err = abs_err / denom
        worst = np.unravel_index(int(np.argmax(rel_err)), rel_err.shape)

        results.append(
            GradCheckResult(
                passed=bool(np.max(rel_err) <= tol),
                max_rel_error=float(np.max(rel_err)),
                max_abs_error=float(np.max(abs_err)),
                worst_index=tuple(int(v) for v in np.atleast_1d(worst)),
                analytical=analytical,
                numerical=numerical,
                argument=argnum,
                notes=notes,
            )
        )

    return results


def gradcheck_or_raise(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    eps: float = DEFAULT_EPS,
    tol: float = DEFAULT_TOL,
    argnums: Sequence[int] | None = None,
    kink_free: bool = True,
) -> float:
    """Run :func:`gradcheck` and raise ``AssertionError`` on the first failure.

    Returns:
        The maximum relative error across every checked argument -- the number the milestone-3
        test suite aggregates into BENCHMARKS.md.
    """
    results = gradcheck(fn, inputs, eps=eps, tol=tol, argnums=argnums, kink_free=kink_free)
    for r in results:
        if not r.passed:
            raise AssertionError(r.report())
    return max(r.max_rel_error for r in results)

"""The grading library: named invariants with failure messages that teach.

Every test in this repository asserts a **named** invariant from a ``SPEC.md``. A bare
``assert x == y`` tells you a number was wrong. An invariant tells you *which contract* broke
and *why anyone cared* — which is the difference between a test that reports a symptom and a
test that reports a cause.

Usage::

    from rubric import invariant

    I_ACCUM = invariant(
        "I-ACCUM",
        "_backward accumulates into .grad with +=; it never assigns",
        "A tensor consumed k times downstream receives k contributions and their sum is the "
        "total derivative. Assignment keeps only the last one -- which does not raise, does "
        "not NaN, and simply trains worse.",
        spec="01-tensorkit/SPEC.md section 3.1",
    )

    def test_diamond_accumulates():
        ...
        I_ACCUM.check(a.grad == 4.0, f"a.grad is {a.grad}, expected 4.0 (2 paths x 2.0)")

This module is owned by the instructor. Students implement against these tests and may not
edit them; the grading harness restores ``tests/`` and ``rubric/`` from ``main`` before
scoring, so weakening a test scores zero rather than passing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Invariant", "invariant", "InvariantViolation", "assert_raises_clearly"]


class InvariantViolation(AssertionError):
    """Raised when a named invariant fails. An AssertionError so pytest reports it normally."""


@dataclass(frozen=True)
class Invariant:
    """A named contract from a SPEC.

    Attributes:
        ident: The invariant id, e.g. ``"I-ACCUM"``.
        statement: What must hold, in one line.
        why: Why it matters — specifically, what goes wrong when it does not. The most useful
            field: nearly every invariant here guards a failure that is silent rather than loud.
        spec: Where it is defined, so the reader can go and look.
    """

    ident: str
    statement: str
    why: str
    spec: str = ""

    def _fail(self, detail: str) -> None:
        """Raise with the full explanation."""
        parts = [
            f"INVARIANT {self.ident} VIOLATED",
            f"  must hold : {self.statement}",
        ]
        if detail:
            parts.append(f"  observed  : {detail}")
        parts.append(f"  why it matters: {self.why}")
        if self.spec:
            parts.append(f"  defined in: {self.spec}")
        raise InvariantViolation("\n".join(parts))

    def check(self, condition: object, detail: str = "") -> None:
        """Assert ``condition`` is truthy."""
        if not condition:
            self._fail(detail)

    def equal(self, actual: Any, expected: Any, detail: str = "") -> None:
        """Assert exact equality."""
        if actual != expected:
            suffix = f" ({detail})" if detail else ""
            self._fail(f"got {actual!r}, expected {expected!r}{suffix}")

    def close(
        self,
        actual: float,
        expected: float,
        *,
        rtol: float = 1e-7,
        atol: float = 0.0,
        detail: str = "",
    ) -> None:
        """Assert two floats agree to a tolerance, reporting the relative error."""
        if math.isnan(actual) or math.isnan(expected):
            self._fail(f"got {actual!r}, expected {expected!r} — NaN is never within tolerance")
        if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
            denom = max(abs(actual), abs(expected), 1e-300)
            rel = abs(actual - expected) / denom
            suffix = f" ({detail})" if detail else ""
            self._fail(
                f"got {actual!r}, expected {expected!r}; "
                f"relative error {rel:.3e} exceeds rtol {rtol:.1e}{suffix}"
            )

    def all_close(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 1e-12,
        detail: str = "",
    ) -> None:
        """Assert two arrays agree elementwise, naming the worst element.

        Reports the *worst* element rather than the first mismatch: the worst one carries the
        diagnostic ratio (2x, 1/N, sign flip) that identifies which class of bug this is.
        """
        import numpy as np

        a = np.asarray(actual, dtype=np.float64)
        e = np.asarray(expected, dtype=np.float64)

        if a.shape != e.shape:
            self._fail(
                f"shape {a.shape} != expected shape {e.shape}" + (f" ({detail})" if detail else "")
            )

        if np.isnan(a).any() and not np.isnan(e).any():
            n = int(np.isnan(a).sum())
            self._fail(f"{n} NaN element(s) in the result" + (f" ({detail})" if detail else ""))

        abs_err = np.abs(a - e)
        tol = atol + rtol * np.abs(e)
        if bool(np.all(abs_err <= tol)):
            return

        worst = np.unravel_index(int(np.argmax(abs_err - tol)), a.shape)
        got, want = float(a[worst]), float(e[worst])
        ratio = got / want if want != 0 else float("inf")
        suffix = f" ({detail})" if detail else ""
        self._fail(
            f"worst element at index {tuple(int(i) for i in worst)}: "
            f"got {got!r}, expected {want!r}, ratio {ratio:.6g}{suffix}\n"
            f"  {int(np.sum(abs_err > tol))} of {a.size} elements outside tolerance\n"
            "  ratio hints (analytical / numerical):\n"
            "    ~2       -> a term is counted twice\n"
            "    <1       -> += was written as =, so a shared subgraph kept one path\n"
            "                instead of the sum of them (exactly 0.5 in the symmetric case)\n"
            "    ~1/N     -> a mean where a sum belongs, e.g. averaging in unbroadcast\n"
            "    ~N       -> a sum where a mean belongs\n"
            "    ~0       -> the rule never fired; check the op is on the tape\n"
            "    negative -> operand order swapped in a non-commutative rule"
        )

    def sorted_ascending(self, values: Sequence[Any], detail: str = "") -> None:
        """Assert a sequence is non-decreasing, naming the first inversion."""
        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                self._fail(
                    f"inversion at index {i}: {values[i - 1]!r} then {values[i]!r}"
                    + (f" ({detail})" if detail else "")
                )


def invariant(ident: str, statement: str, why: str, spec: str = "") -> Invariant:
    """Define a named invariant. See the module docstring."""
    return Invariant(ident=ident, statement=statement, why=why, spec=spec)


def assert_raises_clearly(
    exc_type: type[BaseException],
    *must_mention: str,
) -> Any:
    """Context manager asserting an exception is raised *and* that its message is useful.

    A test that only checks the exception type passes for ``raise ValueError("")``. In a
    library whose docstrings promise "raise with a readable message", the message is part of
    the contract, so it gets asserted::

        with assert_raises_clearly(ValueError, "shape", "(4, 3)"):
            unbroadcast(g, (5,))
    """
    import contextlib

    @contextlib.contextmanager
    def _cm() -> Any:
        try:
            yield
        except exc_type as exc:
            message = str(exc)
            missing = [token for token in must_mention if token.lower() not in message.lower()]
            if missing:
                raise AssertionError(
                    f"{exc_type.__name__} was raised, but its message does not mention "
                    f"{missing!r}, so it will not help whoever hits it.\n"
                    f"  message: {message!r}\n"
                    f"  An error a user cannot act on is barely better than no error."
                ) from exc
        else:
            raise AssertionError(
                f"expected {exc_type.__name__} to be raised, and nothing was raised at all"
            )

    return _cm()

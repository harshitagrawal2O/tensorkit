#!/usr/bin/env python3
"""Is the test suite healthy, and how much of it passes yet?

On a spec-first repository the suite is *supposed* to fail: `main` ships the specification and
the tests, not the implementation. So "did pytest exit non-zero" is a useless signal here. Two
questions are worth asking instead, and this script answers both.

**Is the suite well-formed?** A test that *fails* is an unimplemented milestone -- expected. A
test that *errors* is usually a fixture blowing up, which is also expected when the fixture
constructs something unimplemented and gets a ``NotImplementedError``. Any *other* error is a
defect in the exam itself: a broken fixture, a missing symbol, a typo in a test. That is worth
failing a build over, because it looks to a contributor like their implementation is wrong when
the test was never runnable.

**How much passes?** Per milestone, so progress is visible without pretending to be pass/fail.

    python scripts/suite_health.py              # report; exit 1 only on an unexpected error
    python scripts/suite_health.py --quiet      # just the verdict
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MILESTONES = (1, 2, 3, 4, 5, 6, 7, 8)

#: An error carrying one of these is the expected consequence of an unimplemented stub, not a
#: defect in the test suite. ``NotImplementedError`` is what every stub raises; the pytest
#: wrapper text appears when a fixture is the thing that raised.
EXPECTED_IN_ERRORS = ("NotImplementedError",)

#: A *failure* carrying one of these is not an unimplemented milestone -- it is a test that
#: cannot run as written, which is a defect in the exam. These arrive as failures rather than
#: errors, so the error-side check above never sees them.
UNRUNNABLE_IN_FAILURES = (
    "FailedHealthCheck",  # e.g. @given over a function-scoped fixture
    "Failed: Timeout",  # pytest-timeout killed it
    "InvalidArgument",  # a malformed hypothesis strategy
    "fixture '",  # "fixture 'x' not found"
)


@dataclass
class MilestoneResult:
    """What one milestone's suite did."""

    milestone: int
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    #: Of ``failed``, how many failed because a stub raised rather than because an assertion
    #: genuinely did not hold. The difference is the whole story on a spec-first branch.
    stub_failed: int = 0
    unexpected: list[tuple[str, str]] = field(default_factory=list)
    #: Set when pytest could not be run to a conclusion at all -- a missing interpreter, an
    #: unparseable report, a collection error. Distinct from "ran and matched no tests", which
    #: is the ordinary state of a milestone whose tests are not written yet.
    broken: str | None = None

    @property
    def total(self) -> int:
        """Tests that ran or tried to."""
        return self.passed + self.failed + self.errored


def _why(proc: subprocess.CompletedProcess[str], prefix: str) -> str:
    """``prefix``, plus the most informative line pytest printed before dying."""
    for stream in (proc.stderr, proc.stdout):
        for line in reversed((stream or "").splitlines()):
            if line.strip():
                return f"{prefix}: {line.strip()[:200]}"
    return prefix


def run_milestone(milestone: int, python: str) -> MilestoneResult:
    """Run one milestone's tests and parse the JUnit report."""
    result = MilestoneResult(milestone=milestone)

    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.xml"
        proc = subprocess.run(
            [
                python,
                "-m",
                "pytest",
                "-m",
                f"m{milestone} and not slow",
                f"--junitxml={report}",
                "-q",
                "--tb=no",
                "-p",
                "no:cacheprovider",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        # pytest's exit codes: 0 all passed, 1 tests failed, 2 interrupted, 3 internal error,
        # 4 usage error, 5 nothing collected. Only 0, 1 and 5 mean the run reached a verdict --
        # 5 being the ordinary "this milestone has no tests yet". Anything else means pytest
        # never got as far as running the suite, and a report of zero tests would be a lie.
        if proc.returncode not in (0, 1, 5):
            result.broken = _why(proc, f"pytest exited {proc.returncode}")
            return result

        if not report.exists():
            result.broken = _why(proc, "pytest wrote no JUnit report")
            return result

        try:
            root = ET.parse(report).getroot()
        except ET.ParseError as exc:
            result.broken = f"JUnit report is not parseable: {exc}"
            return result

    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        result.broken = "JUnit report contains no <testsuite> element"
        return result

    total = int(suite.get("tests", 0))
    result.failed = int(suite.get("failures", 0))
    result.errored = int(suite.get("errors", 0))
    result.skipped = int(suite.get("skipped", 0))
    result.passed = total - result.failed - result.errored - result.skipped

    for case in suite.iter("testcase"):
        node = case.find("error")
        if node is None:
            # A *failure* is an unimplemented milestone and is expected -- unless it is one of
            # these, which mean the test could not run as written. Hypothesis rejecting a
            # fixture scope is the motivating case: it looks identical to a failing assertion in
            # the summary line, so it hides among the expected failures indefinitely, and a
            # contributor who implements the milestone correctly still sees it red.
            fail = case.find("failure")
            if fail is None:
                continue
            text = f"{fail.get('message', '')}\n{fail.text or ''}"
            if not any(marker in text for marker in UNRUNNABLE_IN_FAILURES):
                # An ordinary failure. Record whether it is a stub raising or an assertion that
                # genuinely did not hold, so the summary can say which without guessing.
                if any(marker in text for marker in EXPECTED_IN_ERRORS):
                    result.stub_failed += 1
                continue
        else:
            text = f"{node.get('message', '')}\n{node.text or ''}"
            if any(marker in text for marker in EXPECTED_IN_ERRORS):
                continue

        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        first = next(
            (ln.strip() for ln in reversed(text.splitlines()) if ln.strip()),
            "no detail",
        )
        result.unexpected.append((name, first[:160]))

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results = [run_milestone(m, args.python) for m in MILESTONES]

    total_pass = sum(r.passed for r in results)
    total_all = sum(r.total for r in results)
    unexpected = [(r.milestone, name, why) for r in results for name, why in r.unexpected]
    total_err = sum(r.errored for r in results)
    total_fail = sum(r.failed for r in results)
    total_stub_fail = sum(r.stub_failed for r in results)
    total_real_fail = total_fail - total_stub_fail
    broken = [(r.milestone, r.broken) for r in results if r.broken]

    # A run that collected nothing at all is never good news on a repository that ships tests.
    # Without this the script reports "healthy, 0/0" when pytest is missing or a top-level
    # conftest fails to import -- a green gate over a suite that never executed.
    if not broken and total_all == 0:
        broken = [(0, "no tests ran in any milestone -- the suite did not execute")]

    lines = ["## Milestone progress", ""]
    lines.append(
        "`main` ships the specification and the tests, not the implementation, so these start "
        "near zero and fill in as milestones land. See SPEC.md section 4."
    )
    lines.append("")
    lines.append("| Milestone | Passing | State |")
    lines.append("|---|---|---|")
    for r in results:
        if r.broken:
            lines.append(f"| M{r.milestone} | **did not run** | broken |")
        elif r.total == 0:
            lines.append(f"| M{r.milestone} | no tests | - |")
        elif r.passed == r.total:
            lines.append(f"| M{r.milestone} | {r.passed}/{r.total} | done |")
        else:
            lines.append(f"| M{r.milestone} | {r.passed}/{r.total} | open |")
    lines.append(f"| **Total** | **{total_pass}/{total_all}** | |")
    lines.append("")
    lines.append("Milestones marked `open` are available to claim -- see CONTRIBUTING.md.")
    lines.append("")

    lines.append("### Suite health")
    lines.append("")
    if broken:
        lines.append(
            f"**The suite did not run to a verdict in {len(broken)} milestone(s).** This is not "
            f"a failing test -- it means pytest never got far enough to have an opinion, so the "
            f"progress numbers above are not trustworthy. Usually a missing dependency or a "
            f"module that fails to import at collection time."
        )
        lines.append("")
        for milestone, why in broken:
            where = f"M{milestone}" if milestone else "suite"
            lines.append(f"- `{where}` -- {why}")
    elif unexpected:
        lines.append(
            f"**{len(unexpected)} test(s) errored for a reason other than an unimplemented "
            f"stub.** That is a defect in the test suite itself, not in anybody's "
            f"implementation, and it makes the affected milestone unstartable."
        )
        lines.append("")
        for milestone, name, why in unexpected[:25]:
            lines.append(f"- `M{milestone}` `{name}` -- {why}")
    else:
        parts = []
        if total_stub_fail:
            parts.append(f"{total_stub_fail} fail because a stub raised `NotImplementedError`")
        if total_real_fail:
            parts.append(f"{total_real_fail} fail on an assertion")
        if total_err:
            parts.append(f"{total_err} error inside a fixture building something unimplemented")
        detail = ", ".join(parts) if parts else "nothing is failing"
        lines.append(
            f"Healthy -- meaning every test that is not passing is not passing for a reason this "
            f"branch expects. Of {total_all - total_pass} not passing: {detail}. No test is "
            f"unrunnable."
        )
        if total_real_fail:
            lines.append("")
            lines.append(
                f"The {total_real_fail} assertion failure(s) are unimplemented or incorrect "
                f"behaviour, not a broken suite -- that is work remaining, and it is the "
                f"number to watch as milestones land."
            )
    lines.append("")

    report = "\n".join(lines)
    if not args.quiet:
        print(report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    print(
        f"progress: {total_pass}/{total_all} passing; {len(unexpected)} unexpected error(s); "
        f"{len(broken)} milestone(s) that did not run"
    )

    if broken:
        for milestone, why in broken:
            where = f"M{milestone}" if milestone else "suite"
            print(f"::error::{where} did not run: {why}")
        return 1
    if unexpected:
        for milestone, name, why in unexpected[:25]:
            print(f"::error::M{milestone} {name} errored unexpectedly: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

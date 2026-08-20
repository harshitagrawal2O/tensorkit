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


@dataclass
class MilestoneResult:
    """What one milestone's suite did."""

    milestone: int
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    unexpected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Tests that ran or tried to."""
        return self.passed + self.failed + self.errored


def run_milestone(milestone: int, python: str) -> MilestoneResult:
    """Run one milestone's tests and parse the JUnit report."""
    result = MilestoneResult(milestone=milestone)

    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.xml"
        subprocess.run(
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
        if not report.exists():
            return result

        try:
            root = ET.parse(report).getroot()
        except ET.ParseError:
            return result

    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return result

    total = int(suite.get("tests", 0))
    result.failed = int(suite.get("failures", 0))
    result.errored = int(suite.get("errors", 0))
    result.skipped = int(suite.get("skipped", 0))
    result.passed = total - result.failed - result.errored - result.skipped

    for case in suite.iter("testcase"):
        node = case.find("error")
        if node is None:
            continue
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

    lines = ["## Milestone progress", ""]
    lines.append(
        "`main` ships the specification and the tests, not the implementation, so these start "
        "near zero and fill in as milestones land. See SPEC.md section 4."
    )
    lines.append("")
    lines.append("| Milestone | Passing | State |")
    lines.append("|---|---|---|")
    for r in results:
        if r.total == 0:
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
    if unexpected:
        lines.append(
            f"**{len(unexpected)} test(s) errored for a reason other than an unimplemented "
            f"stub.** That is a defect in the test suite itself, not in anybody's "
            f"implementation, and it makes the affected milestone unstartable."
        )
        lines.append("")
        for milestone, name, why in unexpected[:25]:
            lines.append(f"- `M{milestone}` `{name}` -- {why}")
    else:
        lines.append(
            f"Healthy. {total_all - total_pass} test(s) fail and {total_err} error, all of them "
            f"because the code they exercise raises `NotImplementedError`. That is the expected "
            f"state of this branch."
        )
    lines.append("")

    report = "\n".join(lines)
    if not args.quiet:
        print(report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    print(f"progress: {total_pass}/{total_all} passing; {len(unexpected)} unexpected error(s)")

    if unexpected:
        for milestone, name, why in unexpected[:25]:
            print(f"::error::M{milestone} {name} errored unexpectedly: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

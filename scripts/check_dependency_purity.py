#!/usr/bin/env python3
"""Assert that 01-tensorkit depends on NumPy and nothing heavier.

The portfolio's first headline claim is "an autograd engine written against NumPy alone".
That claim is only credible if it is mechanically enforced, so this script is a CI gate
rather than a comment in a README.

Two independent checks run:

1. *Static* -- walk every ``.py`` file under ``01-tensorkit/tensorkit`` with :mod:`ast`
   and reject any ``import``/``from ... import`` naming a banned top-level module.
   Catches banned imports on code paths that are never executed.
2. *Dynamic* -- import ``tensorkit`` in a clean subprocess and diff ``sys.modules``
   before and after. Catches banned modules pulled in transitively by a permitted one.

Exit status is 0 when clean, 1 when a violation is found, so it drops straight into CI.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Top-level module names that must never appear in tensorkit's import graph.
BANNED: frozenset[str] = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "tf",
        "keras",
        "jax",
        "jaxlib",
        "flax",
        "mlx",
        "cupy",
        "tinygrad",
        "autograd",  # HIPS autograd -- would give away the whole exercise
        "micrograd",
    }
)

#: (package import name, directory holding it) pairs to audit.
TARGETS: tuple[tuple[str, str], ...] = (("tensorkit", "."),)


class Violation(Exception):
    """Raised when a banned module is reachable from an audited package."""


def _banned_root(dotted: str) -> str | None:
    """Return the banned top-level name in ``dotted``, or ``None`` if it is clean."""
    root = dotted.split(".", 1)[0]
    return root if root in BANNED else None


def scan_source(package_dir: Path) -> list[str]:
    """Statically scan every module under ``package_dir`` for banned imports.

    Returns:
        A list of human-readable violation strings; empty means clean.
    """
    problems: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - a syntax error is its own CI failure
            problems.append(f"{path}: could not parse ({exc})")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (bad := _banned_root(alias.name)) is not None:
                        problems.append(f"{path}:{node.lineno}: imports banned module {bad!r}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.level == 0
                and (bad := _banned_root(node.module)) is not None
            ):
                problems.append(f"{path}:{node.lineno}: imports banned module {bad!r}")
    return problems


_PROBE = r"""
import json, sys
before = set(sys.modules)
import {package}  # noqa: F401
after = set(sys.modules)
print(json.dumps(sorted({{m.split(".", 1)[0] for m in after - before}})))
"""


def scan_runtime(package: str, package_root: Path) -> list[str]:
    """Import ``package`` in a subprocess and report banned modules it dragged in."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(package=package)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(package_root), "PATH": ""},
        check=False,
    )
    if proc.returncode != 0:
        return [f"importing {package!r} failed:\n{proc.stderr.strip()}"]

    try:
        loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [f"could not parse probe output for {package!r}: {proc.stdout!r}"]

    return [
        f"importing {package!r} transitively loads banned module {name!r}"
        for name in loaded
        if name in BANNED
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="skip the subprocess import probe (useful before the package is installable)",
    )
    args = parser.parse_args(argv)

    all_problems: list[str] = []
    for package, directory in TARGETS:
        package_root = REPO_ROOT / directory
        package_dir = package_root / package
        if not package_dir.is_dir():
            all_problems.append(f"{package_dir} does not exist")
            continue

        all_problems.extend(scan_source(package_dir))
        if not args.static_only:
            all_problems.extend(scan_runtime(package, package_root))

    if all_problems:
        print("dependency purity check FAILED:", file=sys.stderr)
        for problem in all_problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\ntensorkit's value as a portfolio piece is that it is NumPy-only.\n"
            "Move whatever needs a deep-learning framework into 02-nanolm's torch backend.",
            file=sys.stderr,
        )
        return 1

    audited = ", ".join(p for p, _ in TARGETS)
    print(f"dependency purity check passed ({audited}: NumPy only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""TensorKit command line -- GIVEN, fully implemented.

Argument parsing, logging, and dispatch are plumbing. They are written so that the moment a
milestone lands there is already a way to run it from a shell.

    tensorkit gradcheck --milestone 3
    tensorkit train-mnist --epochs 10 --batch-size 64 --seed 0
    tensorkit bench --suite conv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main", "build_parser"]

log = logging.getLogger("tensorkit")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="tensorkit",
        description="Autograd engine and neural network library. NumPy only.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="repeat for DEBUG")
    parser.add_argument("--seed", type=int, default=None, help="global RNG seed")

    sub = parser.add_subparsers(dest="command", required=True)

    gc = sub.add_parser(
        "gradcheck",
        help="sweep registered primitives against numerical differentiation",
        description=(
            "Runs every primitive in tensorkit.ops.PRIMITIVES through gradcheck in float64 and "
            "reports the maximum relative error -- the number that goes into BENCHMARKS.md."
        ),
    )
    gc.add_argument(
        "--milestone", type=int, default=None, help="only primitives from this milestone"
    )
    gc.add_argument("--tol", type=float, default=1e-6, help="relative error tolerance")
    gc.add_argument("--op", type=str, default=None, help="check a single named primitive")

    tr = sub.add_parser("train-mnist", help="train the MNIST CNN")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--data-root", type=Path, default=None)
    tr.add_argument("--checkpoint", type=Path, default=None, help="where to write the best model")
    tr.add_argument("--limit-train", type=int, default=None, help="subsample for a smoke run")

    bn = sub.add_parser("bench", help="run a benchmark suite")
    bn.add_argument(
        "--suite",
        choices=["conv", "throughput", "memory", "all"],
        default="all",
    )
    bn.add_argument("--output", type=Path, default=None, help="write raw JSON results here")

    return parser


def _configure_logging(verbosity: int) -> None:
    """Set the root log level from a repeated -v count."""
    level = logging.WARNING - min(verbosity, 2) * 10
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _cmd_gradcheck(args: argparse.Namespace) -> int:
    """Sweep the primitive registry against numerical differentiation."""
    from tensorkit.ops import PRIMITIVES

    selected = {
        name: p
        for name, p in PRIMITIVES.items()
        if (args.milestone is None or p.milestone == args.milestone)
        and (args.op is None or name == args.op)
    }
    if not selected:
        log.error("no primitives matched the filter")
        return 2

    print(f"{'primitive':<16} {'max rel error':>14}  status")
    print("-" * 44)

    # The per-primitive fixtures -- shapes, positivity, kink offsets -- live in the rubric,
    # which owns what "passes gradcheck" means. Duplicating them here would let the two
    # definitions drift.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from rubric.tensorkit_primitives import check_primitive
    except ImportError:
        log.error("rubric/tensorkit_primitives.py not found -- run from the repository root")
        return 2

    worst = 0.0
    failures = 0
    for name in sorted(selected):
        try:
            err = check_primitive(name, tol=args.tol)
            worst = max(worst, err)
            print(f"{name:<16} {err:>14.3e}  ok")
        except NotImplementedError:
            print(f"{name:<16} {'--':>14}  not implemented")
        except AssertionError as exc:
            failures += 1
            print(f"{name:<16} {'--':>14}  FAILED")
            print(f"  {exc}")

    print("-" * 44)
    print(f"max relative error across all checked primitives: {worst:.3e}")
    return 1 if failures else 0


def _cmd_train_mnist(args: argparse.Namespace) -> int:
    """Train the MNIST CNN via the example script's entry point."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    try:
        from train_mnist import run_training  # type: ignore[import-not-found]
    except ImportError:
        log.error("examples/train_mnist.py not importable -- run from the repository root")
        return 2

    started = time.perf_counter()
    accuracy = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        limit_train=args.limit_train,
    )
    elapsed = time.perf_counter() - started

    print(f"\nfinal test accuracy: {accuracy:.4f}   ({elapsed:.1f}s)")
    if accuracy < 0.98:
        print("Milestone 8 requires >98%. Not there yet.")
        return 1
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    """Run a benchmark suite from the benchmarks package."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from benchmarks.run import run_suite  # type: ignore[import-not-found]
    except ImportError:
        log.error("benchmarks/run.py not found -- run from the repository root")
        return 2
    return int(run_suite(args.suite, output=args.output))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    handlers = {
        "gradcheck": _cmd_gradcheck,
        "train-mnist": _cmd_train_mnist,
        "bench": _cmd_bench,
    }
    try:
        return handlers[args.command](args)
    except NotImplementedError as exc:
        log.error("not implemented yet: %s", exc)
        return 3
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

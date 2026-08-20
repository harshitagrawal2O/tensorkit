"""Train a CNN on MNIST to >98% test accuracy using TensorKit alone -- Milestone 8.

The plumbing here is written for you: data loading, the epoch loop, metric tracking, timing,
checkpointing, and the curve dump that BENCHMARKS.md reads. Two functions are yours, and they
are the two that constitute the milestone:

    build_model()   -- the architecture
    train_step()    -- forward, backward, update

Run it:

    python 01-tensorkit/examples/train_mnist.py --epochs 10 --batch-size 64 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tensorkit.data import ArrayDataset, DataLoader, load_mnist

from tensorkit.losses import CrossEntropyLoss
from tensorkit.nn import Module
from tensorkit.optim import Adam
from tensorkit.tensor import Tensor

CURVE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "mnist_curve.json"


# ---------------------------------------------------------------------------
# YOURS -- Milestone 8
# ---------------------------------------------------------------------------


def build_model() -> Module:
    """Return the CNN architecture to train.

    The target is >98% test accuracy. Getting there on MNIST is not hard; getting there with a
    model that trains in a sane wall-clock time on a NumPy-only engine takes some thought about
    how many convolutions you can afford.

    A configuration in this neighbourhood is known to work:

        Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)     ->  (16, 14, 14)
        Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)    ->  (32, 7, 7)
        Flatten -> Linear(32*7*7, 128) -> ReLU -> Dropout(0.25)
        Linear(128, 10)

    Return **logits**, not probabilities -- CrossEntropyLoss fuses the log-softmax and expects
    raw scores. Feeding it a softmax output trains anyway, just worse, which is the annoying
    kind of bug.

    Tests: tests/test_mnist_integration.py::test_model_output_shape
    """
    raise NotImplementedError("Milestone 8")


def train_step(
    model: Module,
    optimizer: Adam,
    loss_fn: CrossEntropyLoss,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, int]:
    """Run one optimisation step. Return ``(loss, n_correct)``.

    The five lines, in the order that matters:
      1. forward
      2. loss
      3. ``optimizer.zero_grad()``  -- *before* backward, not after
      4. ``loss.backward()``
      5. ``optimizer.step()``

    Zeroing after backward wipes the gradients you just computed and the model never learns;
    zeroing before *the next* forward instead of before backward accumulates across steps and
    the effective learning rate climbs every step. Both produce a plausible-looking training
    loop and neither raises.

    Count correct predictions from the argmax of the logits. Do it under ``no_grad`` or on
    detached data -- building tape nodes for a metric is pure waste.

    Tests: tests/test_mnist_integration.py::test_train_step_decreases_loss
    """
    raise NotImplementedError("Milestone 8")


# ---------------------------------------------------------------------------
# GIVEN -- everything below here
# ---------------------------------------------------------------------------


def evaluate(model: Module, loader: DataLoader, loss_fn: CrossEntropyLoss) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` over the loader, in eval mode."""
    from tensorkit import no_grad

    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    with no_grad():
        for x, y in loader:
            logits = model(Tensor(x))
            loss = loss_fn(logits, Tensor(y))
            total_loss += float(loss.item()) * len(x)
            correct += int(np.sum(np.argmax(logits.numpy(), axis=-1) == y))
            seen += len(x)

    model.train()
    return total_loss / max(seen, 1), correct / max(seen, 1)


def run_training(
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int | None = 0,
    data_root: Path | None = None,
    checkpoint: Path | None = None,
    limit_train: int | None = None,
) -> float:
    """Train and evaluate. Returns final test accuracy.

    Writes the per-epoch curve to ``benchmarks/results/mnist_curve.json`` so BENCHMARKS.md
    plots a measured curve rather than a remembered one.
    """
    if seed is not None:
        np.random.seed(seed)

    print("loading MNIST...")
    kwargs: dict[str, Any] = {} if data_root is None else {"root": data_root}
    x_train, y_train, x_test, y_test = load_mnist(**kwargs)

    if limit_train is not None:
        x_train, y_train = x_train[:limit_train], y_train[:limit_train]

    print(f"  train {x_train.shape}  test {x_test.shape}")

    train_loader = DataLoader(
        ArrayDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    test_loader = DataLoader(ArrayDataset(x_test, y_test), batch_size=256)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    print(f"  {n_params:,} parameters\n")

    curve: list[dict[str, float]] = []
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            loss, correct = train_step(model, optimizer, loss_fn, x, y)
            running_loss += loss * len(x)
            running_correct += correct
            seen += len(x)

            if step % 100 == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)}  "
                    f"loss {running_loss / seen:.4f}  acc {running_correct / seen:.4f}",
                    end="\r",
                )

        train_time = time.perf_counter() - started
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)
        best_accuracy = max(best_accuracy, test_accuracy)

        print(
            f"  epoch {epoch:>2}  "
            f"train_loss {running_loss / seen:.4f}  train_acc {running_correct / seen:.4f}  "
            f"test_loss {test_loss:.4f}  test_acc {test_accuracy:.4f}  "
            f"{train_time:.1f}s  ({seen / train_time:.0f} img/s)"
        )

        curve.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "train_accuracy": running_correct / seen,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "seconds": train_time,
                "images_per_second": seen / train_time,
            }
        )

        if checkpoint is not None and test_accuracy >= best_accuracy:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            np.savez(checkpoint, **model.state_dict())

    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURVE_PATH.write_text(
        json.dumps(
            {
                "config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                    "parameters": n_params,
                },
                "curve": curve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ncurve written to {CURVE_PATH}")

    return curve[-1]["test_accuracy"]


def main() -> int:
    """Parse arguments and train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    args = parser.parse_args()

    accuracy = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        limit_train=args.limit_train,
    )
    print(f"final test accuracy: {accuracy:.4f}")
    return 0 if accuracy > 0.98 else 1


if __name__ == "__main__":
    raise SystemExit(main())

# TensorKit

[![CI](https://github.com/harshitagrawal2O/tensorkit/actions/workflows/ci.yml/badge.svg)](https://github.com/harshitagrawal2O/tensorkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![NumPy only](https://img.shields.io/badge/dependencies-NumPy%20only-orange)](scripts/check_dependency_purity.py)

**A reverse-mode automatic differentiation engine and neural network library, in NumPy alone.**

No PyTorch, TensorFlow, or JAX anywhere in the dependency tree — and that claim is enforced by a
CI job, not by good intentions. `scripts/check_dependency_purity.py` walks every import with
`ast`, then imports the package in a clean subprocess and diffs `sys.modules`. If a deep-learning
framework ever becomes reachable, the build goes red.

---

## This repository is built backwards, on purpose

**The specification and the tests came first. The implementation is deliberately missing.**

`main` contains a complete design document, 94 tests that fail today, and a package of stubs
whose docstrings carry the exact contract each function must satisfy. Filling those stubs in is
the exercise.

```python
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape``, undoing whatever broadcasting the forward pass did.

    Invariants:
        I-UB-SHAPE: the result's shape equals ``shape``, exactly, always.
        I-UB-SUMPRESERVE: ``result.sum() == grad.sum()``. Broadcasting duplicates a value, so
            its gradient must *add up*, not average. Getting this wrong yields gradients off by
            exactly the broadcast factor -- which trains slowly rather than failing.
    """
    raise NotImplementedError("Milestone 2")
```

That is the whole style of the project. Every invariant has a name, a test, and a written
explanation of the failure it prevents — and nearly all of those failures are *silent*, which is
why they are worth naming. A gradient that is wrong by a factor of two does not raise. It just
trains worse.

## Why bother, when PyTorch exists

Because using PyTorch teaches you the layer library, and building this teaches you the tape
underneath it: a directed acyclic graph, a topological sort, gradient accumulation for shared
subgraphs, and the exact inverse of NumPy's broadcasting rules.

Two shortcuts look tempting and both are wrong, which is a large part of the lesson:

- **Finite differences as the engine.** Differentiating *n* parameters numerically costs *n*
  forward passes. Reverse mode gets all *n* partials in one backward pass at roughly 2× the
  forward cost, because it propagates a vector–Jacobian product and never materialises a
  Jacobian. Finite differences survive here only as the test oracle.
- **Recursive `backward()` at each node.** A node with two consumers is visited twice; a chain of
  diamonds, exponentially often. The graph has to be linearised once and walked in reverse.

## Status

Milestone 3 of 8. Honest about what exists:

| # | Milestone | CS subject | State |
|---|---|---|:---:|
| 1 | Scalar reverse-mode autodiff | DSA (graphs, topological sort), OOP | ☐ |
| 2 | Tensor autodiff with correct unbroadcasting | Deep Learning, broadcast semantics | ☐ |
| 3 | Numerical gradient checking as the correctness spine | Numerical methods | ☐ |
| 4 | Module system and the core layers | OOP | ☐ |
| 5 | Normalisation layers | Deep Learning | ☐ |
| 6 | Losses and optimisers | ML optimisation, numerical stability | ☐ |
| 7 | Conv2d, pooling, im2col | DSA, convolution arithmetic, performance | ☐ |
| 8 | MNIST CNN to >98%, end to end | Integration, experimental methodology | ☐ |

Boxes on `main` are unticked because `main` holds no implementation. Milestones 1–3 have been
demonstrated to be achievable against this exact test suite — see
[the reference branch](#the-reference-branch) — and 4–8 are open.

**Nothing here trains a network yet.** There are no layers, no optimisers, and no MNIST run. That
arrives at milestone 8.

## The book

[**How Automatic Differentiation Works**](docs/tensorkit-book.pdf) — a 40-page companion that teaches this subject from first principles, in plain language, assuming nothing beyond Python.

It explains the ideas, the failure modes, and the reasoning behind every design decision in `SPEC.md` — deliberately in prose, pseudocode and worked numeric examples only. It will not hand you an implementation, because that is the exercise.

## Can you defend it?

[`INTERVIEW.md`](INTERVIEW.md) has 28 questions ordered easy to hard, each with follow-up
chains, because the follow-up is where an interview actually goes. Hints, not answers.

If you implement this repository you should be able to answer everything down to about question
22 without notes. A sample of the last six:

> Derive the backward pass of LayerNorm. Explain why it has three terms and not one. Which term
> would you drop if you were being lazy, and how wrong would the result be on a batch of 256?

> Your `Conv2d` passes every shape test and every gradcheck at `stride == kernel_size`, and
> produces subtly wrong gradients at `stride=1`. What is the bug, and why did the stride-2 tests
> not catch it?

## Quickstart

```bash
git clone https://github.com/harshitagrawal2O/tensorkit.git
cd tensorkit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[all]"

python -m pytest -m m1 -q      # milestone 1: 22 failures, one per contract you have not met
python -m pytest -q            # everything
```

Then open [`SPEC.md`](SPEC.md), read section 4 milestone 1, and start on `tensorkit/scalar.py`.

## What the failures tell you

Tests here do not report that a number was wrong. They report which contract broke:

```
INVARIANT I-ACCUM VIOLATED
  must hold : _backward accumulates into .grad with +=; it never assigns
  observed  : a.grad is 8.0, expected 10.0 (two paths contribute (a-b) and (a+b); 2+8 = 10)
  why it matters: A value consumed k times downstream receives k contributions, and their sum
    is the total derivative. Assignment keeps whichever fired last. Nothing raises, nothing
    NaNs -- the network simply trains with the gradient of every shared subgraph truncated.
  defined in: SPEC.md section 3.1
```

Array comparisons also print the ratio of the worst element with a hint attached, because the
ratio usually identifies the bug outright: `~2` is double counting, `~1/N` is a mean where a sum
belongs, a sign flip is a swapped operand order.

## Layout

| Path | What |
|---|---|
| [`SPEC.md`](SPEC.md) | Scope boundary, module breakdown, every invariant, 8 milestones with a definition of done each |
| [`INTERVIEW.md`](INTERVIEW.md) | 28 questions with follow-up chains. Hints, not answers |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Where measured numbers go, and the methodology they must follow |
| [`docs/tensorkit-book.pdf`](docs/tensorkit-book.pdf) | The 40-page companion book |
| `tests/` | The executable specification, organised by milestone |
| `tensorkit/` | The implementation — stubs on `main` |
| `rubric/` | The grading library: named invariants and the gradcheck fixture table |
| `docs/concepts/` | The ideas in prose and pseudocode. Mostly unwritten — a good place to contribute |
| `benchmarks/` | Where every measured claim comes from. A skeleton |

## Contributing

Genuinely open to it — see [CONTRIBUTING.md](CONTRIBUTING.md). Short version:

- **Claim a milestone** with the issue template. One person per milestone at a time.
- **Do not weaken a test to make it pass.** If you think a test is wrong, open a spec-defect
  issue with your reasoning. Several tests have already been corrected that way, and finding a
  real contradiction is worth more than an implementation.
- **You do not have to implement anything to help.** `docs/concepts/` is largely empty and the
  benchmark harness is a skeleton.

## The reference branch

`reference/m1-3` holds an independent implementation of the first three milestones, **written by
an AI agent** working from `SPEC.md` and the tests under the same rules contributors follow. It
scores 92/92 with a maximum gradient error of `8.99e-10` against numerical differentiation.

It exists as a diff target — write yours first, then compare. It is never merged into `main` and
it is not authoritative. Two defects in this repository's own test suite were found by that agent
and are fixed in `main`.

## Sibling projects

TensorKit is the base of a four-project chain. Each stands alone; together they cover a full
stack built by hand.

| Project | What |
|---|---|
| **TensorKit** *(here)* | Autograd engine and NN library |
| [NanoLM](https://github.com/harshitagrawal2O/nanolm) | Transformer + BPE tokenizer — trains on this autograd |
| [InferServe](https://github.com/harshitagrawal2O/inferserve) | Inference server — serves NanoLM |
| [VectorForge](https://github.com/harshitagrawal2O/vectorforge) | Vector database — queries NanoLM through InferServe |

`SPEC.md` section 6 freezes the surface NanoLM depends on.

## Licence

[MIT](LICENSE).

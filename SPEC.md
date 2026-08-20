# TensorKit — Specification

> Reverse-mode automatic differentiation engine and neural network library.
> **NumPy only.** No PyTorch, TensorFlow, JAX, or any other autodiff package anywhere in the
> dependency tree — enforced by `scripts/check_dependency_purity.py` and a CI job.

---

## 1. Problem statement

Every deep learning framework is, at its core, three things stacked: a tensor type that records
the operations performed on it, a graph walk that turns those records into gradients, and a
library of layers and optimisers built on top. Using PyTorch teaches you the third. Building
TensorKit teaches you the first two, which is where the actual computer science is — a directed
acyclic graph, a topological sort, careful accumulation semantics for shared subgraphs, and the
exact inverse of NumPy's broadcasting rules.

The deliverable is a library that can train a convolutional network on MNIST to **>98% test
accuracy**, where every gradient it computes has been validated against numerical
differentiation to a stated tolerance.

### Why the obvious approach fails

Two shortcuts look tempting and both are wrong:

* **Finite differences as the engine.** Numerically differentiating a network with *n*
  parameters costs *n* forward passes. For even a small CNN that is ~10⁵ forward passes per
  step. Reverse mode gets all *n* partials in **one** backward pass at ~2× the cost of the
  forward pass, because it propagates a vector–Jacobian product rather than materialising any
  Jacobian. Finite differences survive here only as the *test oracle*, never as the engine.
* **Recursive `backward()` at each node.** A node with two consumers gets visited twice; a
  chain of diamonds gets visited exponentially often; and recursion depth blows the stack on a
  deep network. The graph must be linearised once by topological sort, then walked in reverse
  exactly once per node.

### Scope boundary — explicitly NOT building

| Out of scope | Why |
|---|---|
| GPU / CUDA kernels | The interesting part is the graph, not the SIMT model. NumPy on CPU is enough to hit 98% on MNIST. |
| Distributed / multi-device training | Orthogonal systems problem, already covered by 03-inferserve. |
| Forward-mode autodiff, higher-order gradients | Reverse mode over a first-order graph is the thing being learnt. `grad-of-grad` is a documented non-goal. |
| Graph optimisation (operator fusion, JIT, constant folding) | Compiler work, not autodiff work. Listed as an extension in `docs/concepts/autodiff.md`. |
| RNN / LSTM / attention layers | Attention is built in **02-nanolm**, on top of this engine. Building it twice adds nothing. |
| Sparse tensors, complex dtypes, mixed precision | `float32`/`float64` dense only. Mixed precision is an `INTERVIEW.md` extension question. |
| Shapes changing between forward and backward | Define-by-run: the graph is rebuilt each forward pass; shapes are fixed within one pass. |
| Serialising the graph | Parameters serialise (`np.savez`); the tape does not. |

---

## 2. Module breakdown

```
tensorkit/
├── __init__.py          Public surface: Tensor, no_grad, __version__
├── scalar.py            Value  — scalar reverse-mode autodiff (M1, pedagogical)
├── tensor.py            Tensor — the tape node: data, grad, _backward, _prev, _op
├── autograd.py          Topological sort, the backward driver, the no_grad context
├── ops.py               Differentiable primitives and their local gradient rules
├── broadcasting.py      unbroadcast(): the inverse of NumPy broadcasting
├── gradcheck.py         Central-difference numerical oracle          [GIVEN]
├── losses.py            MSELoss, CrossEntropyLoss (log-sum-exp stable)
├── init.py              Xavier/Glorot, Kaiming/He, zeros, ones        [GIVEN]
├── nn/
│   ├── module.py        Module base: parameters(), zero_grad(), train()/eval(), __call__
│   ├── container.py     Sequential, ModuleList
│   ├── linear.py        Linear
│   ├── activations.py   ReLU, GELU, Softmax, Tanh, Sigmoid
│   ├── norm.py          LayerNorm, BatchNorm1d, BatchNorm2d
│   ├── conv.py          Conv2d (im2col), MaxPool2d, Flatten
│   ├── dropout.py       Dropout (inverted, train/eval aware)
│   └── embedding.py     Embedding (scatter-add gradient)
├── optim/
│   ├── optimizer.py     Optimizer base: param groups, step(), zero_grad()
│   ├── sgd.py           SGD with momentum, Nesterov, weight decay
│   └── adam.py          Adam with bias correction, AdamW decoupled decay
├── data/
│   ├── mnist.py         Download + parse IDX, checksum verified       [GIVEN]
│   └── loader.py        Dataset / DataLoader: shuffle, batch, drop_last [GIVEN]
└── cli.py               `tensorkit gradcheck | train-mnist | bench`   [GIVEN]
```

**[GIVEN]** = fully written for you: plumbing, I/O, and test tooling. Everything else is a stub
carrying a docstring contract, and the algorithm inside it is yours.

---

## 3. Data structures and their exact invariants

Every invariant below has an ID and a test that asserts it by that ID.

### 3.1 `Tensor` — the tape node

```python
class Tensor:
    data: np.ndarray  # dense, float32 or float64, C-contiguous
    grad: np.ndarray | None  # same shape and dtype as data, or None
    requires_grad: bool
    _prev: tuple[Tensor, ...]  # inputs this node was produced from
    _backward: Callable[[], None]  # closure: reads self.grad, accumulates into _prev grads
    _op: str  # operator name, for graph rendering and error messages
```

* **I-SHAPE** — if `grad is not None` then `grad.shape == data.shape` and
  `grad.dtype == data.dtype`. No exceptions, including scalars: a 0-d tensor has a 0-d gradient.
* **I-LEAF** — a tensor with `_prev == ()` is a *leaf*. Only leaves with `requires_grad=True`
  retain `grad` after `backward()`. Non-leaf gradients are freed unless `retain_grad()` was
  called, so memory after the backward pass is O(parameters), not O(activations).
* **I-ACCUM** — `_backward` **accumulates** (`+=`) into each input's `.grad`; it never assigns.
  A tensor consumed by *k* downstream operations receives *k* contributions and their sum is the
  correct total derivative. Overwriting here is the single most common autodiff bug, and it is
  *silently* wrong rather than loudly wrong.
* **I-ACYCLIC** — the graph reachable through `_prev` is a DAG. An operation may never take its
  own output as an input. In-place mutation of `.data` on a tensor already recorded as an input
  is forbidden and raises.
* **I-ONCE** — one `backward()` call invokes each reachable node's `_backward` **exactly once**,
  in reverse topological order. Not zero times, not twice.
* **I-READY** — when node *n*'s `_backward` runs, every consumer of *n* has already run, so
  `n.grad` is complete. This is exactly what reverse topological order buys, and it is why a
  DFS that recurses into children immediately is wrong.
* **I-SEED** — `backward()` on a non-scalar tensor requires an explicit `grad` argument. On a
  scalar it seeds `1.0`. Implicitly summing a non-scalar output is a silent semantic choice and
  is rejected.
* **I-NOGRAD** — inside `with no_grad():` no node records `_prev` or `_backward`. Forward
  arithmetic still works; the tape stays empty. Used by evaluation and by the optimiser step.

### 3.2 `unbroadcast` — the inverse of NumPy broadcasting

Forward, NumPy stretches a `(3,)` operand against a `(4, 3)` one. Backward, the gradient
arriving as `(4, 3)` must fold back to `(3,)`. The rule, exactly:

* **I-UB-SUM** — sum over every axis *inserted* on the left (rank promotion), then sum with
  `keepdims=True` over every axis where the original extent was 1 and the broadcast extent was
  greater than 1.
* **I-UB-SHAPE** — `unbroadcast(g, shape).shape == shape` for every `(g, shape)` pair NumPy
  would have broadcast. Property-tested over randomly generated compatible shape pairs.
* **I-UB-SUMPRESERVE** — `unbroadcast(g, shape).sum() ≈ g.sum()`. Broadcasting duplicates a
  value, so its gradient must **add up**, not average. Getting this wrong yields gradients off
  by exactly the broadcast factor, which trains slowly rather than failing — hence a test.

### 3.3 `Module` — parameter ownership

* **I-MOD-UNIQUE** — `parameters()` yields each parameter object **once**, even when the same
  `Linear` is referenced from two places (weight tying). Deduplicate by `id()`, not by name; a
  duplicated parameter gets its update applied twice per step.
* **I-MOD-MODE** — `train()`/`eval()` propagate recursively to every submodule. `Dropout` and
  `BatchNorm` read the flag; every other layer ignores it.
* **I-MOD-ZERO** — `zero_grad()` sets `.grad = None`, not `zeros_like`. Distinguishing "no
  gradient yet" from "gradient is exactly zero" is what makes I-ACCUM checkable.
* **I-MOD-BUFFER** — buffers (BatchNorm running statistics) appear in `state_dict()` but never
  in `parameters()`. The optimiser must not be able to reach them.

### 3.4 `Optimizer`

* **I-OPT-STATE** — per-parameter state is keyed by `id(param)`, created lazily on first
  `step()`, and always matches the parameter's shape.
* **I-OPT-BIAS** — Adam's `step_count` is **1** on the first update, not 0. The corrections are
  `1 - β₁ᵗ` and `1 - β₂ᵗ`; at `t = 0` both are zero and the update divides by zero. An
  off-by-one here produces a first step that is wildly too large and a loss curve that looks
  "just a bit unstable".
* **I-OPT-NOGRAD** — the update itself runs under `no_grad()`. Otherwise the optimiser extends
  the tape and the graph grows without bound across steps.

### 3.5 `Conv2d` via im2col

* **I-CONV-SHAPE** — `H_out = (H + 2p − dilation·(k−1) − 1) // s + 1`, and likewise for width.
  Asserted against a table of known-good configurations rather than recomputed by the test.
* **I-CONV-COL** — the im2col matrix has shape `(N·H_out·W_out, C_in·k_h·k_w)` and the backward
  pass is a col2im **scatter-add**: with `stride < kernel_size`, receptive fields overlap and one
  input pixel contributes to several outputs. Scatter-*assign* is the classic bug and it only
  shows up when strides overlap.

---

## 4. Milestones

Eight milestones, each independently testable, each roughly 3–5 hours.
Run one with `make test-tensorkit-m<N>`.

---

### M1 — Scalar reverse-mode autodiff
**CS subject:** DSA (directed graphs, topological sort) · OOP (operator overloading, closures)

Build `Value`: a scalar carrying `data`, `grad`, and a `_backward` closure. Support
`+ - * / **`, `exp`, `log`, `tanh`, `relu`, and the reflected dunders. Implement `backward()`
as an iterative topological sort followed by a reverse walk.

**Definition of done**
- [ ] `tests/test_scalar_autograd.py` passes in full.
- [ ] A diamond (`d = (a+b) * (a−b)`) gives `a.grad == 2a` and `b.grad == −2b` — proves I-ACCUM.
- [ ] The topological sort is **iterative**: a 10,000-node chain does not raise `RecursionError`.
- [ ] Hypothesis property: over randomly generated expression DAGs, every node's `_backward`
      fires exactly once (instrumented with a counter) — I-ONCE.
- [ ] Calling `backward()` twice without zeroing doubles the gradients, and that is *documented*
      rather than silently prevented.

---

### M2 — Tensor autodiff with correct unbroadcasting
**CS subject:** Deep Learning (the VJP formulation) · Numerical computing (broadcast semantics)

Promote `Value` to `Tensor` over `np.ndarray`. Implement `unbroadcast`, then the primitives:
`add`, `sub`, `mul`, `div`, `neg`, `pow`, `matmul`, `sum`, `mean`, `max`, `exp`, `log`, `sqrt`,
`reshape`, `transpose`, `getitem`, `concat`. Implement `no_grad`.

**Definition of done**
- [ ] `make test-tensorkit-m2` green.
- [ ] Hypothesis property over random compatible shape pairs: I-UB-SHAPE and I-UB-SUMPRESERVE.
- [ ] `matmul` handles 2-D×2-D, batched 3-D, and vector promotion; backward is `dA = dC @ Bᵀ`,
      `dB = Aᵀ @ dC` with batch dimensions unbroadcast.
- [ ] `sum(axis=..., keepdims=False)` backward re-expands the collapsed axis correctly — the
      single most common shape bug in a hand-rolled engine.
- [ ] Inside `no_grad()`, `_prev` is empty on every produced tensor — I-NOGRAD.

---

### M3 — Numerical gradient checking as the correctness spine
**CS subject:** Numerical methods (truncation vs round-off error, step-size selection)

`gradcheck.py` is given. Your job is to make every M2 primitive pass it, and to understand why
the tolerance is what it is: central differences carry truncation error O(h²) and round-off
error O(ε/h), so total error is minimised near `h ≈ ε^(1/3)` — about `6e-6` in float64, and
hopeless in float32. Every gradcheck test therefore runs in **float64**.

**Definition of done**
- [ ] Every primitive passes `gradcheck` with relative error < 1e-6 in float64.
- [ ] `docs/concepts/gradcheck.md` explains why ReLU is probed away from 0 and why `max`/`abs`
      need kink-avoiding perturbations.
- [ ] `make test-tensorkit-m3` reports the **maximum** relative error across all primitives —
      that number goes straight into `BENCHMARKS.md`.
- [ ] `test_gradcheck_catches_a_planted_bug` proves the oracle has teeth: a deliberately broken
      `div` backward (second term dropped) is caught.

---

### M4 — Module system and the core layers
**CS subject:** OOP (composition, template-method, parameter registration)

`Module` base with recursive `parameters()`, `buffers()`, `zero_grad()`, `train()`/`eval()`,
`state_dict()`/`load_state_dict()`. Then `Linear`, `ReLU`, `GELU` (tanh approximation *and*
exact erf form), `Softmax`, `Tanh`, `Sigmoid`, `Sequential`, `ModuleList`, `Flatten`, `Dropout`,
`Embedding`.

**Definition of done**
- [ ] `make test-tensorkit-m4` green.
- [ ] Weight tying: the same `Linear` referenced twice leaves `len(list(m.parameters()))`
      unchanged — I-MOD-UNIQUE.
- [ ] `Dropout` in `eval()` is exactly the identity; in `train()` the output expectation equals
      the input over 10,000 samples (inverted dropout — scale at train time, not test time).
- [ ] `Softmax` is numerically stable: `softmax([1000, 1001])` is finite.
- [ ] `Embedding` backward scatter-**adds**: an index repeated within a batch accumulates.
- [ ] `state_dict()` round-trips through `np.savez` and back with bitwise equality.

---

### M5 — Normalisation layers
**CS subject:** Deep Learning (normalisation statistics, train/eval divergence, running buffers)

`LayerNorm`, `BatchNorm1d`, `BatchNorm2d`. The backward pass through the statistics is the hard
part: μ and σ² each depend on every element of the reduction group, so the gradient has three
terms, not one.

**Definition of done**
- [ ] `make test-tensorkit-m5` green, every layer passing float64 gradcheck.
- [ ] `BatchNorm` in `train()` uses batch statistics and updates the running buffers; in `eval()`
      it uses the buffers and updates nothing. A test asserts that in `eval()` an item's output
      is **independent of the other items in the batch** — the user-visible property that matters.
- [ ] Running mean/var are buffers, not parameters — I-MOD-BUFFER.
- [ ] `LayerNorm` normalises over a configurable trailing axis set; `BatchNorm2d` normalises
      per-channel over `(N, H, W)`.
- [ ] Batch size 1 in `BatchNorm.train()` raises a clear error instead of dividing by zero.

---

### M6 — Losses and optimisers
**CS subject:** ML optimisation (momentum, adaptive moments, bias correction) · Numerical stability

`MSELoss`, `CrossEntropyLoss` fusing log-softmax with NLL, `SGD` (momentum, Nesterov, weight
decay), `Adam`, `AdamW`.

**Definition of done**
- [ ] `make test-tensorkit-m6` green.
- [ ] `CrossEntropyLoss` on logits of magnitude 1e4 gives a finite loss and finite gradients —
      the log-sum-exp trick, tested rather than assumed.
- [ ] The fused cross-entropy gradient equals `(softmax(logits) − onehot) / N` to 1e-9, checked
      against the unfused composition.
- [ ] Adam's first step matches a hand-computed reference to 1e-12 **with** bias correction, and
      a companion test asserts the uncorrected version *fails* — so the correction is
      demonstrably load-bearing (I-OPT-BIAS).
- [ ] `SGD(momentum=0)` is bitwise plain gradient descent (regression guard).
- [ ] Every optimiser drives `f(x) = (x−3)²` from `x=0` to 3 ± 1e-4.

---

### M7 — Conv2d, pooling, and im2col
**CS subject:** DSA (strided views, scatter-add) · Deep Learning (convolution arithmetic) · Performance

`Conv2d` via im2col (`np.lib.stride_tricks.as_strided` or explicit fancy indexing), `MaxPool2d`,
and the col2im scatter-add backward.

**Definition of done**
- [ ] `make test-tensorkit-m7` green.
- [ ] Output shape matches the I-CONV-SHAPE table across the full cross-product
      `kernel ∈ {1,3,5} × stride ∈ {1,2} × padding ∈ {0,1,2}`.
- [ ] Conv2d passes float64 gradcheck on a `(2, 3, 8, 8)` input.
- [ ] Overlapping receptive fields (`stride=1, kernel=3`) accumulate correctly in col2im — a
      test that fails specifically if scatter-assign is used instead of scatter-add.
- [ ] `MaxPool2d` backward routes gradient to the **argmax position only**, ties broken
      deterministically (first occurrence).
- [ ] im2col is ≥20× faster than the naive six-nested-loop reference on `(32, 3, 32, 32)` with a
      3×3 kernel. The naive reference lives in the test file and doubles as the correctness oracle.

---

### M8 — MNIST CNN to >98%, end to end
**CS subject:** Everything above, integrated · Experimental methodology

Train a CNN on MNIST using only TensorKit. Produce accuracy and loss curves and a full gradcheck
sweep over the trained architecture.

**Definition of done**
- [ ] `python 01-tensorkit/examples/train_mnist.py` reaches **>98% test accuracy**, reproducible
      from a fixed seed to ±0.2%.
- [ ] Every layer in the final architecture passes float64 gradcheck; the maximum relative error
      is recorded in `BENCHMARKS.md`.
- [ ] `BENCHMARKS.md` carries training throughput vs a PyTorch model of identical architecture,
      peak memory per batch size, the accuracy curve, and a hardware header.
- [ ] The PyTorch comparison script lives in `benchmarks/` and imports torch **there only** —
      `make purity` still passes.
- [ ] `README.md` states the wall-clock gap to PyTorch honestly and explains it: no fused
      kernels, no BLAS-level batching of small ops, Python-level graph construction overhead.

---

## 5. Definition of done for the project

TensorKit is finished when all eight milestones are checked, `make check` is green, `make purity`
is green, `BENCHMARKS.md` has real numbers under a hardware header, and `INTERVIEW.md`'s hardest
question — *"derive the gradient of LayerNorm and explain why it has three terms"* — can be
answered at a whiteboard without notes.

## 6. Downstream contract

**02-nanolm imports this.** The tiny NanoLM configuration trains on TensorKit's autograd, so the
following surface is frozen once M6 lands and may not change shape afterwards:

```python
from tensorkit import Tensor, no_grad
from tensorkit.nn import Module, Linear, LayerNorm, Softmax, Dropout, Embedding, Sequential
from tensorkit.optim import Adam
from tensorkit.losses import CrossEntropyLoss
```

Anything NanoLM needs beyond that list — notably causal masking and RoPE's rotation — is built
in NanoLM *against* these primitives, not added here. If NanoLM needs a new primitive, it is a
TensorKit change with its own gradcheck test, not a special case.

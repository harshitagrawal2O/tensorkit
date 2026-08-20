# TensorKit — benchmarks

No claim in this project appears without a measured number behind it. This file is where the
numbers live, and it is mostly empty on purpose: milestones 4-8 are unwritten, so most of these
rows have nothing to report yet.

**A number with no methodology is not a measurement.** Every row below names the script that
produced it, and every run records the hardware it ran on.

## Hardware

Fill this in when you record a result. Two numbers from different machines in the same table,
without this section, are not comparable.

| Field | Value |
|---|---|
| CPU | _unrecorded_ |
| Cores / threads | _unrecorded_ |
| RAM | _unrecorded_ |
| OS | _unrecorded_ |
| Python | _unrecorded_ |
| NumPy | _unrecorded_ |
| Date | _unrecorded_ |

## Methodology

- Every benchmark is run **three times**; the table reports the median, not the best.
- The machine is otherwise idle. A run competing with a test suite is not a measurement — this
  has already produced a spurious result once in this project's history.
- Timings use `time.perf_counter`, never `time.time`: the latter is wall-clock and can step
  backwards on an NTP adjustment.
- Latency is reported as percentiles, never as a mean. A 200 ms mean is consistent with
  everything taking 200 ms and with 95% taking 50 ms while 5% take 3 s, and those are different
  systems.
- Raw JSON output goes in `benchmarks/results/` and is gitignored. This table is committed.

## Results

| Claim | Number | Source |
|---|---|---|
| MNIST test accuracy | _not measured_ | `examples/train_mnist.py` |
| Max gradient error vs numerical differentiation (float64) | _not measured_ | `make test-m3` |
| Training throughput vs PyTorch, identical architecture | _not measured_ | `benchmarks/vs_pytorch.py` |
| Peak memory per batch size | _not measured_ | `benchmarks/memory.py` |
| im2col speedup over the naive six-loop reference | _not measured_ | `benchmarks/conv.py` |

## Reproducing

```bash
pip install -e ".[all]"
python -m benchmarks.run --all
```

The harness in `benchmarks/` is a skeleton. Building it out is a good contribution that needs
no implementation work — see [CONTRIBUTING.md](CONTRIBUTING.md).

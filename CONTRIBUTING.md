# Contributing to TensorKit

Thanks for looking. This project is unusual, so read this bit first — it will save you time.

## What this repository is

**The specification and the tests came first. The implementation is deliberately missing.**

`main` holds:

- [`SPEC.md`](SPEC.md) — the design, with every invariant named and explained
- `tests/` — a full suite written against that spec, which fails today
- `tensorkit/` — stubs whose docstrings carry the contract each function must satisfy
- CI, tooling, and a milestone roadmap

So a green test suite is not the starting state. It is the goal.

## How to contribute

### Claim a milestone

Open an issue using the **Milestone claim** template, or comment on an existing milestone issue.
One person per milestone at a time, so nobody duplicates a week of work. If a claim goes quiet
for two weeks it is fair game again — say so in the thread and take it.

Milestones are ordered and each builds on the last. Check [`SPEC.md`](SPEC.md) section 4 for the
definition of done on each one.

### The rules that actually matter

1. **Do not weaken a test to make it pass.** If you believe a test is wrong, say so in an issue
   with your reasoning — several tests have already been fixed that way, and finding a real
   defect in the spec is a more valuable contribution than an implementation. What is not
   acceptable is quietly loosening a tolerance or deleting an assertion.
2. **Keep the docstrings.** Every stub carries the invariants it must satisfy and an explanation
   of the failure each one prevents. Replace the `raise NotImplementedError(...)` line; leave the
   prose. Extend it if you learn something worth recording.
3. **The scope boundary is real.** [`SPEC.md`](SPEC.md) section 1 lists what this project
   deliberately does not build. A PR that adds something from that list will be declined, however
   good it is.

### Before you open a PR

```bash
python -m pytest -m "not slow" -q     # the suite
python -m ruff check .                # lint
python -m ruff format .               # formatting
python -m mypy                        # strict types
```

All four must be clean. CI runs the same commands, plus a milestone-sharded view so a red square
points at one milestone rather than the whole repo.

### Commit and PR style

- Present tense, imperative mood: `Add the col2im scatter-add backward`.
- Say **why** in the body, not just what. The diff already says what.
- One milestone per PR where possible. A PR that lands three milestones is very hard to review.
- Reference the issue you claimed.
- If you used an AI assistant, say so in the PR description. It is not disqualifying; it is
  context a reviewer is entitled to.

## Good first contributions

You do not have to implement anything to be useful here:

- **`docs/concepts/`** — one markdown file per idea, explaining it in prose and pseudocode with
  no runnable implementation. [`SPEC.md`](SPEC.md) references the files it expects; most do not
  exist yet.
- **Benchmarks** — `benchmarks/` is a skeleton. Every claim in this project is supposed to come
  from a measured number.
- **Spec defects** — the invariants are precise enough to be checkable. Several have already been
  found to be contradictory or unsatisfiable. Finding another one is a real contribution.
- **A test for an edge case nobody covered.** New tests are welcome; weakened ones are not.

## The `reference/m1-3` branch

There is a branch called `reference/m1-3` holding an independent implementation of the first
three milestones. **It was written by an AI agent**, under the same rules as everyone else, as a
diff target — so you can compare your approach against another one after you have written yours.

It is never merged into `main`, it is not authoritative, and it is not a solution key you should
read first. Milestones 1-3 complete: 92/92 tests green.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are accepted under the [MIT Licence](LICENSE).

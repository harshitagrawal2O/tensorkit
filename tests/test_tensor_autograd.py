"""Milestone 2 — tensor autodiff: shapes, broadcasting, memory, and no_grad."""

from __future__ import annotations

import numpy as np
import pytest

from rubric import assert_raises_clearly, invariant
from tensorkit import no_grad
from tensorkit.tensor import Tensor

pytestmark = pytest.mark.m2

SPEC = "01-tensorkit/SPEC.md section 3.1"

I_SHAPE = invariant(
    "I-SHAPE",
    "grad.shape == data.shape and grad.dtype == data.dtype, with no exceptions",
    "A mismatched gradient shape means an unbroadcast is missing somewhere upstream. Caught "
    "here it names the op; left uncaught, NumPy broadcasts it into a plausible wrong answer.",
    SPEC,
)
I_LEAF = invariant(
    "I-LEAF",
    "only leaves with requires_grad retain .grad after backward()",
    "Retaining every intermediate makes peak memory O(activations) rather than O(parameters). "
    "On a network deep enough to matter that is the difference between fitting in RAM and not.",
    SPEC,
)
I_SEED = invariant(
    "I-SEED",
    "backward() on a non-scalar requires an explicit grad argument",
    "Implicitly summing a non-scalar output is a silent semantic choice. A caller who wanted a "
    "mean gets gradients scaled by the element count and no warning whatsoever.",
    SPEC,
)
I_NOGRAD = invariant(
    "I-NOGRAD",
    "inside no_grad() no tensor records _prev or _backward",
    "Without it, evaluation retains the whole graph until GC, and the optimiser step extends "
    "the tape every iteration so it grows without bound across training.",
    SPEC,
)
I_ACCUM = invariant(
    "I-ACCUM",
    "gradients accumulate with +=, including through fancy indexing",
    "np.add.at accumulates over repeated indices; arr[idx] = v keeps only the last write. For "
    "an embedding table, that means every frequent token learns from one occurrence per batch.",
    SPEC,
)
I_ACYCLIC = invariant(
    "I-ACYCLIC",
    "a tensor already recorded as an input may not be mutated in place",
    "The backward closure captured the pre-mutation value. Mutating it makes the recorded "
    "gradient rule inconsistent with the data it will be evaluated against.",
    SPEC,
)


def _t(data, requires_grad=True):
    """Build a float64 tensor from a literal, since gradcheck-adjacent maths needs float64."""
    return Tensor(np.asarray(data, dtype=np.float64), requires_grad=requires_grad, dtype=np.float64)


# ---------------------------------------------------------------------------
# Shape and dtype contracts
# ---------------------------------------------------------------------------


def test_gradient_matches_value_shape_and_dtype():
    x = _t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    (x * 2.0).sum().backward()

    I_SHAPE.equal(x.grad.shape, x.data.shape, detail="2-D tensor")
    I_SHAPE.equal(x.grad.dtype, x.data.dtype, detail="dtype must match too")


def test_zero_dimensional_tensor_has_zero_dimensional_gradient():
    """The exception people make and should not: a 0-d tensor keeps a 0-d gradient."""
    x = _t(3.0)
    (x * x).backward()

    I_SHAPE.equal(x.grad.shape, (), detail="0-d input")
    assert float(x.grad) == pytest.approx(6.0)


def test_accumulate_grad_rejects_shape_mismatch():
    x = _t([1.0, 2.0, 3.0])
    with assert_raises_clearly(ValueError, "shape"):
        x.accumulate_grad(np.ones((2, 3)))


def test_zero_grad_sets_none_not_zeros():
    """None and zeros are different states, and the difference is what makes I-ACCUM testable."""
    x = _t([1.0, 2.0])
    x.sum().backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None, "zero_grad() must set .grad to None, not to an array of zeros"


# ---------------------------------------------------------------------------
# Seeding and memory
# ---------------------------------------------------------------------------


def test_backward_requires_seed_for_non_scalar():
    x = _t([1.0, 2.0, 3.0])
    y = x * 2.0
    with assert_raises_clearly(RuntimeError, "scalar"):
        y.backward()


def test_backward_accepts_explicit_seed_for_non_scalar():
    x = _t([1.0, 2.0, 3.0])
    y = x * 2.0
    y.backward(np.array([1.0, 0.0, 0.5]))

    I_SEED.all_close(x.grad, [2.0, 0.0, 1.0], detail="seed must flow through elementwise")


def test_non_leaf_gradients_are_released():
    x = _t([1.0, 2.0])
    mid = x * 3.0
    out = mid.sum()
    out.backward()

    I_LEAF.check(x.grad is not None, "the leaf must retain its gradient")
    I_LEAF.check(
        mid.grad is None,
        "an intermediate retained its gradient without retain_grad(); peak memory is now "
        "O(activations)",
    )


def test_retain_grad_keeps_an_intermediate():
    x = _t([1.0, 2.0])
    mid = x * 3.0
    mid.retain_grad()
    mid.sum().backward()

    I_LEAF.check(mid.grad is not None, "retain_grad() was called and the gradient was still freed")


# ---------------------------------------------------------------------------
# no_grad
# ---------------------------------------------------------------------------


def test_no_grad_leaves_the_tape_empty():
    x = _t([1.0, 2.0])
    with no_grad():
        y = x * 3.0 + 1.0

    I_NOGRAD.equal(y._prev, (), detail="an op inside no_grad() recorded its inputs")
    I_NOGRAD.all_close(y.numpy(), [4.0, 7.0], detail="the forward value must still be computed")


def test_no_grad_nesting_restores_the_previous_state():
    x = _t([1.0])
    with no_grad(), no_grad():
        pass
    y = x * 2.0
    I_NOGRAD.check(
        y._prev != (),
        "recording did not resume after leaving nested no_grad() blocks; the inner exit "
        "unconditionally re-enabled instead of restoring",
    )


# ---------------------------------------------------------------------------
# The ops where shape handling actually bites
# ---------------------------------------------------------------------------


def test_matmul_2d_backward():
    a = _t([[1.0, 2.0], [3.0, 4.0]])
    b = _t([[5.0, 6.0], [7.0, 8.0]])
    (a @ b).sum().backward()

    ones = np.ones((2, 2))
    I_SHAPE.all_close(a.grad, ones @ b.numpy().T, detail="dA = dC @ B.T")
    I_SHAPE.all_close(b.grad, a.numpy().T @ ones, detail="dB = A.T @ dC")


def test_matmul_batched_backward_unbroadcasts():
    """``(2, 3, 4) @ (4, 5)``: the right operand broadcasts over the batch axis.

    Its gradient must be summed back over that axis. A 2-D-only matmul backward returns a
    ``(2, 4, 5)`` gradient for a ``(4, 5)`` parameter — which either raises at the optimiser
    or, if the optimiser is lenient, updates a broadcast view.
    """
    rng = np.random.default_rng(0)
    a = Tensor(rng.standard_normal((2, 3, 4)), requires_grad=True, dtype=np.float64)
    b = Tensor(rng.standard_normal((4, 5)), requires_grad=True, dtype=np.float64)

    (a @ b).sum().backward()

    I_SHAPE.equal(a.grad.shape, (2, 3, 4), detail="batched left operand")
    I_SHAPE.equal(b.grad.shape, (4, 5), detail="right operand must unbroadcast back over the batch")
    expected_b = a.numpy().reshape(-1, 4).T @ np.ones((6, 5))
    I_SHAPE.all_close(b.grad, expected_b, detail="dB summed over the batch axis")


def test_sum_backward_reinserts_the_reduced_axis():
    """``keepdims=False`` removes the axis; the backward pass has to put it back."""
    x = _t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x.sum(axis=1).sum().backward()

    I_SHAPE.equal(x.grad.shape, (2, 3))
    I_SHAPE.all_close(x.grad, np.ones((2, 3)), detail="every element contributed once")


def test_mean_backward_divides_by_the_reduced_count():
    x = _t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x.mean().backward()

    I_SHAPE.all_close(x.grad, np.full((2, 3), 1 / 6), detail="mean over 6 elements")


def test_max_backward_routes_to_the_argmax_only():
    x = _t([[1.0, 7.0, 3.0]])
    x.max().backward()

    I_SHAPE.all_close(x.grad, [[0.0, 1.0, 0.0]], detail="only the argmax receives gradient")


def test_max_backward_breaks_ties_deterministically():
    """Routing the full gradient to every tied position multiplies it by the tie count."""
    x = _t([[5.0, 5.0, 1.0]])
    x.max().backward()

    total = float(np.sum(x.grad))
    I_SHAPE.close(total, 1.0, rtol=1e-12, detail=f"gradient sums to {total}, must sum to 1.0")


def test_getitem_backward_accumulates_repeated_indices():
    """The bug: ``grad[idx] = v`` keeps the last write; ``np.add.at`` accumulates."""
    x = _t([1.0, 2.0, 3.0])
    x[[0, 0, 2]].sum().backward()

    I_ACCUM.all_close(
        x.grad,
        [2.0, 0.0, 1.0],
        detail="index 0 appears twice, so it must receive 2.0",
    )


def test_reshape_and_transpose_backward_restore_the_input_shape():
    x = _t(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
    x.reshape(6, 4).sum().backward()
    I_SHAPE.equal(x.grad.shape, (2, 3, 4), detail="reshape backward")

    x.zero_grad()
    x.transpose(2, 0, 1).sum().backward()
    I_SHAPE.equal(
        x.grad.shape, (2, 3, 4), detail="transpose backward applies the INVERSE permutation"
    )


def test_masked_fill_blocks_gradient_at_masked_positions():
    """NanoLM's causal attention depends on this."""
    x = _t([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[False, True], [False, False]])
    x.masked_fill(mask, -1e9).sum().backward()

    I_SHAPE.all_close(
        x.grad,
        [[1.0, 0.0], [1.0, 1.0]],
        detail="a filled constant has no dependence on the input",
    )


def test_concat_backward_splits_at_the_same_offsets():
    a = _t([[1.0, 2.0]])
    b = _t([[3.0, 4.0], [5.0, 6.0]])
    Tensor.concat([a, b], axis=0).sum().backward()

    I_SHAPE.equal(a.grad.shape, (1, 2))
    I_SHAPE.equal(b.grad.shape, (2, 2))


def test_broadcast_add_unbroadcasts_the_bias():
    x = _t(np.ones((8, 3)))
    bias = _t([0.1, 0.2, 0.3])
    (x + bias).sum().backward()

    I_SHAPE.equal(bias.grad.shape, (3,))
    I_SHAPE.all_close(bias.grad, [8.0, 8.0, 8.0], detail="summed over the batch of 8, not averaged")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_in_place_mutation_of_a_recorded_tensor_raises():
    x = _t([1.0, 2.0])
    _ = x * 2.0
    with assert_raises_clearly(RuntimeError, "in-place"):
        x.data[0] = 99.0
        x.sum().backward()


def test_integer_input_is_rejected_at_construction():
    """An integer tensor gives an integer gradient, which truncates to zero.

    Indistinguishable from a vanishing gradient, and it fails at the wrong end of the network.
    """
    with assert_raises_clearly(TypeError, "float"):
        Tensor(np.array([1, 2, 3], dtype=np.int64), requires_grad=True)


def test_empty_tensor_backward_does_not_crash():
    x = Tensor(np.zeros((0, 3)), requires_grad=True, dtype=np.float64)
    x.sum().backward()
    I_SHAPE.equal(x.grad.shape, (0, 3), detail="empty input, empty gradient")

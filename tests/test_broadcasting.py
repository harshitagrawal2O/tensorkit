"""Milestone 2 — unbroadcasting, the inverse of NumPy's broadcast rules.

This is a 15-line function with two property tests, and it is worth that attention because
getting it wrong scales every affected gradient by a constant factor. Nothing raises. The model
trains, slowly, and you spend a week suspecting the learning rate.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rubric import assert_raises_clearly, invariant
from tensorkit.broadcasting import broadcast_shapes_or_raise, unbroadcast

pytestmark = pytest.mark.m2

SPEC = "01-tensorkit/SPEC.md section 3.2"

I_UB_SHAPE = invariant(
    "I-UB-SHAPE",
    "unbroadcast(g, shape).shape == shape for every pair NumPy would have broadcast",
    "The gradient must be the same shape as the operand it belongs to, or the optimiser "
    "update either raises or broadcasts into something meaningless.",
    SPEC,
)
I_UB_SUM = invariant(
    "I-UB-SUMPRESERVE",
    "unbroadcast(g, shape).sum() == g.sum()",
    "Broadcasting duplicates a value, so its gradient must ADD over the copies. Using mean "
    "divides every affected gradient by the broadcast factor -- a silent, uniform learning-rate "
    "cut on exactly the parameters that broadcast, which is usually every bias in the network.",
    SPEC,
)


# ---------------------------------------------------------------------------
# The cases you can reason about by hand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grad_shape", "target_shape"),
    [
        ((4, 3), (3,)),  # rank promotion: a bias against a batch
        ((4, 3), (4, 1)),  # size-1 stretching along the last axis
        ((4, 3), (1, 3)),  # size-1 stretching along the first
        ((2, 3, 4), (4,)),  # two axes of promotion at once
        ((2, 3, 4), (1, 3, 1)),  # promotion and stretching together
        ((5,), (5,)),  # no broadcasting happened; identity
        ((3, 1), (3, 1)),  # identity with a size-1 axis already present
        ((2, 3), ()),  # collapse all the way to a scalar
    ],
)
def test_unbroadcast_shapes(grad_shape, target_shape):
    g = np.ones(grad_shape, dtype=np.float64)
    out = unbroadcast(g, target_shape)

    I_UB_SHAPE.equal(out.shape, target_shape, detail=f"unbroadcasting {grad_shape}")
    I_UB_SUM.close(
        float(out.sum()), float(g.sum()), rtol=1e-12, detail=f"{grad_shape} -> {target_shape}"
    )


def test_bias_gradient_is_the_batch_sum():
    """The case this function exists for.

    ``y = x @ W + b`` with a batch of 8 broadcasts ``b`` eight times, so ``b.grad`` is the sum
    over the batch, not the mean. Using the mean makes the bias learn 8x slower than the weight
    beside it, which looks like nothing at all until you plot per-parameter update magnitudes.
    """
    per_sample = np.arange(1, 25, dtype=np.float64).reshape(8, 3)
    folded = unbroadcast(per_sample, (3,))

    I_UB_SHAPE.equal(folded.shape, (3,))
    I_UB_SUM.all_close(folded, per_sample.sum(axis=0), detail="bias gradient over a batch of 8")


def test_scalar_target_collapses_everything():
    g = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = unbroadcast(g, ())
    I_UB_SHAPE.equal(out.shape, ())
    I_UB_SUM.close(float(out), float(g.sum()), rtol=1e-12)


def test_incompatible_shapes_raise_readably():
    """A shape error deep in a network is only debuggable if the message names both shapes."""
    with assert_raises_clearly(ValueError, "(4, 3)", "(5,)"):
        broadcast_shapes_or_raise((4, 3), (5,))


# ---------------------------------------------------------------------------
# Properties: hold for every shape pair NumPy accepts, not just the eight above
# ---------------------------------------------------------------------------


@st.composite
def broadcastable_pair(draw):
    """Draw ``(result_shape, operand_shape)`` such that ``operand`` broadcasts to ``result``.

    Built by construction rather than by filtering: generate the result, then derive a valid
    operand by dropping leading axes and collapsing some remaining ones to 1. That is exactly
    the space of shapes NumPy accepts, so the property covers it without wasted draws.
    """
    ndim = draw(st.integers(min_value=0, max_value=4))
    result = tuple(draw(st.lists(st.integers(1, 5), min_size=ndim, max_size=ndim)))

    drop = draw(st.integers(min_value=0, max_value=ndim))
    tail = result[drop:]
    operand = tuple(1 if draw(st.booleans()) else extent for extent in tail)
    return result, operand


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(pair=broadcastable_pair(), seed=st.integers(0, 2**31 - 1))
def test_property_shape_and_sum_are_preserved(pair, seed):
    result_shape, operand_shape = pair
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(result_shape)

    out = unbroadcast(g, operand_shape)

    I_UB_SHAPE.equal(out.shape, operand_shape, detail=f"from {result_shape}")
    I_UB_SUM.close(
        float(out.sum()),
        float(g.sum()),
        rtol=1e-9,
        atol=1e-12,
        detail=f"{result_shape} -> {operand_shape}",
    )


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(pair=broadcastable_pair(), seed=st.integers(0, 2**31 - 1))
def test_property_unbroadcast_is_the_adjoint_of_broadcast(pair, seed):
    """``<broadcast(x), g> == <x, unbroadcast(g)>`` for all x and g.

    This is the real statement: unbroadcast is the *adjoint* of broadcasting, and the adjoint
    of a linear map is exactly what a VJP is. If this identity holds, the sum-preservation
    property above is a corollary -- and every broadcast gradient in the library is correct by
    construction rather than by inspection.
    """
    result_shape, operand_shape = pair
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(operand_shape)
    g = rng.standard_normal(result_shape)

    lhs = float(np.sum(np.broadcast_to(x, result_shape) * g))
    rhs = float(np.sum(x * unbroadcast(g, operand_shape)))

    I_UB_SUM.close(
        lhs,
        rhs,
        rtol=1e-9,
        atol=1e-12,
        detail=f"adjoint identity failed for {operand_shape} -> {result_shape}",
    )

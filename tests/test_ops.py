"""Milestone 2 -- forward values and the registry, for the ops referenced from docstrings.

The gradient rules themselves are asserted in ``test_tensor_autograd.py`` and swept in
``test_gradcheck.py``. What is left here is the part neither of those covers: that the forward
value is right, and that the registry stays honest about what it contains.
"""

from __future__ import annotations

import numpy as np
import pytest

from rubric import invariant
from tensorkit.tensor import Tensor

# No module-level mark: most of this file is M2, but the two stability tests at the bottom
# exercise softmax (M4) and log_softmax (M6). A module-level m2 would pull them into the
# M2 gate and fail a student who correctly stayed in scope.

I_REGISTRY = invariant(
    "I-REGISTRY",
    "every differentiable op in tensorkit.ops is decorated with @register",
    "gradcheck sweeps the registry. An op that is not in it is an op nobody checked, and the "
    "sweep goes green anyway.",
    "01-tensorkit/SPEC.md section 2",
)


def _t(data):
    return Tensor(np.asarray(data, dtype=np.float64), requires_grad=True, dtype=np.float64)


@pytest.mark.m2
def test_registry_is_populated_and_well_formed():
    from tensorkit.ops import PRIMITIVES

    I_REGISTRY.check(PRIMITIVES, "tensorkit.ops.PRIMITIVES is empty; no op called @register")
    for name, prim in PRIMITIVES.items():
        I_REGISTRY.equal(prim.name, name, detail="registry key must match the Primitive's name")
        I_REGISTRY.check(prim.arity >= 1, f"{name} declares arity {prim.arity}")
        I_REGISTRY.check(1 <= prim.milestone <= 8, f"{name} declares milestone {prim.milestone}")


@pytest.mark.m2
def test_add_broadcast_backward():
    a, b = _t(np.ones((4, 3))), _t([1.0, 2.0, 3.0])
    out = a + b
    out.sum().backward()

    np.testing.assert_allclose(out.numpy(), np.ones((4, 3)) + np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(b.grad, [4.0, 4.0, 4.0])


@pytest.mark.m2
def test_mul_broadcast_backward():
    a, b = _t(np.full((2, 3, 4), 2.0)), _t(np.arange(4, dtype=np.float64))
    (a * b).sum().backward()
    np.testing.assert_allclose(b.grad, np.full(4, 12.0))


@pytest.mark.m2
def test_div_backward_has_both_terms():
    a, b = _t([6.0]), _t([3.0])
    (a / b).backward()
    np.testing.assert_allclose(a.grad, [1 / 3])
    np.testing.assert_allclose(b.grad, [-6 / 9])


@pytest.mark.m2
def test_matmul_batched_backward():
    rng = np.random.default_rng(11)
    a = Tensor(rng.standard_normal((2, 3, 4)), requires_grad=True, dtype=np.float64)
    b = Tensor(rng.standard_normal((4, 5)), requires_grad=True, dtype=np.float64)
    (a @ b).sum().backward()
    assert a.grad.shape == (2, 3, 4)
    assert b.grad.shape == (4, 5)


@pytest.mark.m2
def test_sum_backward_keepdims_false():
    x = _t(np.ones((3, 4)))
    x.sum(axis=0).sum().backward()
    assert x.grad.shape == (3, 4)


@pytest.mark.m4
def test_softmax_stability():
    """softmax([1000, 1001]) must be finite. The naive form overflows to inf/inf = nan."""
    from tensorkit.ops import softmax

    out = softmax(_t([[1000.0, 1001.0]]))
    values = out.numpy()
    assert np.all(np.isfinite(values)), f"softmax produced {values}; subtract the max first"
    np.testing.assert_allclose(values.sum(axis=-1), [1.0], rtol=1e-9)


@pytest.mark.m6
def test_log_softmax_extreme_logits():
    """log_softmax must be computed as x - logsumexp(x), never as log(softmax(x))."""
    from tensorkit.ops import log_softmax

    out = log_softmax(_t([[-1000.0, 0.0, 1000.0]]))
    assert np.all(np.isfinite(out.numpy())), (
        f"log_softmax produced {out.numpy()}; softmax underflowed to 0 and log(0) is -inf"
    )

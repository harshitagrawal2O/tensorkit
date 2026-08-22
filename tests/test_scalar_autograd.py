"""Milestone 1 — scalar reverse-mode autodiff.

The tests are the specification. Each asserts a named invariant from
``01-tensorkit/SPEC.md`` section 3.1, and each failure message says which contract broke and
what it costs you to break it.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rubric import assert_raises_clearly, invariant
from tensorkit.scalar import Value

pytestmark = pytest.mark.m1

SPEC = "01-tensorkit/SPEC.md section 3.1"

I_ACCUM = invariant(
    "I-ACCUM",
    "_backward accumulates into .grad with +=; it never assigns",
    "A value consumed k times downstream receives k contributions, and their sum is the total "
    "derivative. Assignment keeps whichever fired last. Nothing raises, nothing NaNs -- the "
    "network simply trains with the gradient of every shared subgraph silently truncated.",
    SPEC,
)
I_ONCE = invariant(
    "I-ONCE",
    "one backward() call runs each reachable node's _backward exactly once",
    "Twice double-counts that node's contribution. Zero times means it was unreachable and "
    "should not have been in the walk at all. A recursive descent through a chain of diamonds "
    "runs some nodes exponentially often, which is a hang rather than a wrong answer.",
    SPEC,
)
I_READY = invariant(
    "I-READY",
    "a node's _backward runs only after every consumer of it has run",
    "Otherwise it propagates a partial gradient. Reverse topological order is precisely the "
    "schedule that guarantees this; a DFS that recurses into children on the way down does not.",
    SPEC,
)
I_ITERATIVE = invariant(
    "I-ONCE/iterative",
    "the topological sort is iterative, not recursive",
    "A 30-layer network with a few hundred ops per layer passes sys.getrecursionlimit(). A "
    "RecursionError in the backward pass of a model that trains fine at depth 4 is a nasty "
    "thing to debug at depth 40.",
    SPEC,
)


# ---------------------------------------------------------------------------
# Local gradient rules
# ---------------------------------------------------------------------------


def test_add_backward():
    a, b = Value(2.0), Value(-3.0)
    c = a + b
    c.backward()

    assert c.data == pytest.approx(-1.0)
    I_ACCUM.close(a.grad, 1.0, detail="d(a+b)/da is 1")
    I_ACCUM.close(b.grad, 1.0, detail="d(a+b)/db is 1")


def test_mul_backward():
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    c.backward()

    assert c.data == pytest.approx(-6.0)
    I_ACCUM.close(a.grad, -3.0, detail="d(ab)/da is b")
    I_ACCUM.close(b.grad, 2.0, detail="d(ab)/db is a")


def test_pow_backward():
    a = Value(3.0)
    c = a**4
    c.backward()

    assert c.data == pytest.approx(81.0)
    I_ACCUM.close(a.grad, 4 * 3.0**3, detail="d(a^n)/da is n*a^(n-1)")


def test_pow_with_value_exponent_raises():
    """A Value exponent needs the a**x * ln(a) term, which M1 does not implement.

    Raising beats computing something plausible: a wrong gradient here is invisible.
    """
    with assert_raises_clearly(TypeError, "exponent"):
        _ = Value(2.0) ** Value(3.0)  # type: ignore[operator]


@pytest.mark.parametrize(("x", "expected_grad"), [(2.5, 1.0), (-2.5, 0.0)])
def test_relu_backward(x, expected_grad):
    a = Value(x)
    out = a.relu()
    out.backward()

    assert out.data == pytest.approx(max(x, 0.0))
    I_ACCUM.close(a.grad, expected_grad, detail=f"ReLU gradient at x={x}")


def test_exp_backward():
    a = Value(0.7)
    out = a.exp()
    out.backward()

    assert out.data == pytest.approx(math.exp(0.7))
    I_ACCUM.close(a.grad, math.exp(0.7), detail="d(e^x)/dx is the output itself")


def test_log_backward():
    a = Value(4.0)
    out = a.log()
    out.backward()

    assert out.data == pytest.approx(math.log(4.0))
    I_ACCUM.close(a.grad, 0.25, detail="d(ln x)/dx is 1/x")


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_log_of_non_positive_raises(bad):
    """``log(0)`` must not silently return -inf.

    An -inf gradient propagates to NaN through the first multiplication and destroys every
    weight in the batch, ten layers away from the actual mistake.
    """
    with assert_raises_clearly(ValueError, "log"):
        Value(bad).log()


def test_tanh_backward():
    a = Value(0.8)
    out = a.tanh()
    out.backward()

    assert out.data == pytest.approx(math.tanh(0.8))
    I_ACCUM.close(a.grad, 1 - math.tanh(0.8) ** 2, detail="d(tanh)/dx is 1 - tanh^2")


def test_division_and_reflected_operators():
    a, b = Value(6.0), Value(3.0)
    out = (a / b) + (10.0 - a) + (2.0 * b) + (12.0 / b)
    out.backward()

    # d/da: 1/b - 1 = 1/3 - 1
    I_ACCUM.close(a.grad, 1 / 3 - 1, rtol=1e-9, detail="reflected __rsub__ must contribute -1")
    # d/db: -a/b^2 + 2 - 12/b^2 = -6/9 + 2 - 12/9
    I_ACCUM.close(b.grad, -6 / 9 + 2 - 12 / 9, rtol=1e-9, detail="__rtruediv__ term is -12/b^2")


def test_radd_is_reached_and_correct():
    """``float + Value`` goes through ``__radd__``, which nothing else here exercises.

    Found by mutation testing: every binary-operator mutation of ``__radd__``'s body survived
    the suite, which can only happen if no test reaches it. The reflected-operator test above
    covers ``__rsub__`` and ``__rtruediv__`` but never puts a float on the left of a ``+``.

    Addition is commutative, so a wrong ``__radd__`` is invisible in the forward value of a
    symmetric expression -- it has to be asserted directly.
    """
    a = Value(4.0)
    out = 10.0 + a

    assert out.data == pytest.approx(14.0), "10.0 + Value(4.0) must be 14.0"
    out.backward()
    I_ACCUM.close(a.grad, 1.0, detail="d(c + a)/da is 1 for any constant c")


def test_rsub_forward_value_not_only_its_gradient():
    """``__rsub__`` is ``(-self) + other``, and only its *gradient* was ever asserted.

    Mutating the ``+`` to ``-`` leaves the gradient at -1 either way, so the mutant survived.
    The forward value is what distinguishes them: ``10 - 4`` is 6, while ``(-4) - 10`` is -14.
    """
    a = Value(4.0)
    out = 10.0 - a

    assert out.data == pytest.approx(6.0), "10.0 - Value(4.0) must be 6.0, not -14.0"
    out.backward()
    I_ACCUM.close(a.grad, -1.0, detail="d(c - a)/da is -1")


def test_relu_subgradient_at_exactly_zero():
    """The documented convention is that the subgradient at 0 is 0. Nothing tested it.

    ``Value.relu``'s docstring states the convention and names this file as its test, but the
    parametrisation only covered 2.5 and -2.5. At exactly 0 the forward value is 0.0 under
    either ``>`` or ``>=``, so only the gradient separates them: ``>`` gives 0.0 and ``>=``
    gives 1.0. Mutation testing flipped the comparison and nothing failed.

    Any value in [0, 1] is a valid subgradient. The point is not that 0 is the only defensible
    choice -- it is that the choice is written down, so it should be pinned down.
    """
    a = Value(0.0)
    out = a.relu()
    out.backward()

    assert out.data == pytest.approx(0.0), "relu(0) is 0"
    I_ACCUM.close(
        a.grad, 0.0, detail="the documented subgradient at the kink is 0, matching PyTorch"
    )


def test_chain_rule_through_a_product_whose_output_grad_is_not_one():
    """Every backward here was previously tested with an incoming gradient of exactly 1.0.

    That is the single largest gap mutation testing found. When ``out.grad == 1.0``, the
    expressions ``self.data * out.grad``, ``self.data / out.grad`` and ``self.data ** out.grad``
    all agree, so mutating the multiply in *any* ``_backward`` survived: ``__mul__``, ``exp``,
    ``tanh`` and ``relu`` were all affected.

    Every product in the old suite was either the root of the graph or fed an addition, and
    both hand a gradient of 1.0 to the multiply. Nesting the product inside a non-linear
    operation is what makes the incoming gradient something else.

    ``out = exp(a * b)`` with a=2, b=3:
        out          = e^6
        d(out)/d(ab) = e^6                  <- the incoming gradient, not 1
        d(out)/da    = b * e^6 = 3 * e^6
        d(out)/db    = a * e^6 = 2 * e^6
    """
    a, b = Value(2.0), Value(3.0)
    out = (a * b).exp()
    out.backward()

    e6 = math.exp(6.0)
    assert out.data == pytest.approx(e6)
    I_ACCUM.close(a.grad, 3.0 * e6, rtol=1e-9, detail="d(e^(ab))/da is b*e^(ab), not b")
    I_ACCUM.close(b.grad, 2.0 * e6, rtol=1e-9, detail="d(e^(ab))/db is a*e^(ab), not a")


# ---------------------------------------------------------------------------
# Graph semantics — the part that is actually hard
# ---------------------------------------------------------------------------


def test_diamond_accumulates():
    """``d = (a+b) * (a-b)`` is d(a^2 - b^2), so a.grad == 2a and b.grad == -2b.

    The canonical I-ACCUM test: ``a`` reaches ``d`` by two paths. Assignment instead of
    accumulation gives whichever path ran last, which is a plausible-looking wrong number.
    """
    a, b = Value(5.0), Value(3.0)
    d = (a + b) * (a - b)
    d.backward()

    assert d.data == pytest.approx(16.0)
    I_ACCUM.close(a.grad, 10.0, detail="two paths contribute (a-b) and (a+b); 2+8 = 10")
    I_ACCUM.close(b.grad, -6.0, detail="two paths contribute (a-b) and -(a+b); 2-8 = -6")


def test_self_addition_accumulates():
    """``a + a`` is the smallest possible shared subgraph."""
    a = Value(3.0)
    out = a + a
    out.backward()

    assert out.data == pytest.approx(6.0)
    I_ACCUM.close(a.grad, 2.0, detail="both operands are the same node; 1 + 1 = 2")


def test_backward_visits_each_node_once(count_backward_calls):
    a, b = Value(2.0), Value(3.0)
    shared = a * b
    out = (shared + a) * (shared - b)

    counts = count_backward_calls(out)
    out.backward()

    for node_id, n in counts.items():
        I_ONCE.equal(n, 1, detail=f"node {node_id} fired {n} times")


def test_topological_order_is_iterative():
    """A 10,000-node chain must not raise RecursionError."""
    node = Value(1.0)
    for _ in range(10_000):
        node = node + 1.0

    try:
        order = node.topological_order()
    except RecursionError:  # pragma: no cover - the failure this test exists for
        I_ITERATIVE.check(False, "topological_order() raised RecursionError on a 10k chain")
        raise

    I_ITERATIVE.check(len(order) >= 10_000, f"order has {len(order)} nodes, expected >= 10000")
    node.backward()


def test_topological_order_puts_parents_before_children():
    a, b = Value(1.0), Value(2.0)
    mid = a * b
    out = mid + a

    order = out.topological_order()
    position = {id(v): i for i, v in enumerate(order)}

    for node in order:
        for parent in node._prev:
            I_READY.check(
                position[id(parent)] < position[id(node)],
                f"input {parent!r} appears after its consumer {node!r} in the topological order",
            )


def test_backward_twice_accumulates_and_is_documented():
    """Calling backward() twice doubles the gradients. That is the contract, not a bug.

    It is what gradient accumulation across microbatches relies on. The test exists so the
    behaviour cannot be "fixed" into an implicit zeroing that would silently break that use.
    """
    a = Value(3.0)
    out = a * a
    out.backward()
    first = a.grad
    out.backward()

    I_ACCUM.close(first, 6.0, detail="d(a^2)/da = 2a")
    I_ACCUM.close(a.grad, 12.0, detail="a second backward() must add to .grad, not reset it")


def test_leaf_starts_with_zero_gradient():
    a = Value(1.5)
    assert a.grad == 0.0, "a Value must start with grad 0.0 so accumulation has a base"
    assert a._prev == (), "a Value with no children is a leaf"


def test_single_node_backward():
    """Edge case: backward() on a leaf. The seed is d(self)/d(self) = 1."""
    a = Value(7.0)
    a.backward()
    I_ACCUM.close(a.grad, 1.0, detail="the seed gradient of the root is 1.0")


def test_value_is_not_iterable():
    with pytest.raises(TypeError):
        list(Value(1.0))  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Property: any expression DAG, every node fires exactly once
# ---------------------------------------------------------------------------


@st.composite
def expression_dag(draw):
    """Build a random DAG by combining a pool of existing nodes.

    Reusing nodes from the pool is what makes it a DAG rather than a tree, and shared nodes are
    exactly where I-ACCUM and I-ONCE break. A tree-shaped generator would pass on a broken
    implementation.
    """
    pool = [Value(float(draw(st.integers(min_value=1, max_value=5)))) for _ in range(3)]
    for _ in range(draw(st.integers(min_value=1, max_value=12))):
        left = pool[draw(st.integers(min_value=0, max_value=len(pool) - 1))]
        right = pool[draw(st.integers(min_value=0, max_value=len(pool) - 1))]
        op = draw(st.sampled_from(["add", "mul", "sub"]))
        pool.append({"add": left + right, "mul": left * right, "sub": left - right}[op])
    return pool[-1]


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(root=expression_dag())
def test_property_every_node_fires_exactly_once(root):
    counts: dict[int, int] = {}
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        counts[id(node)] = 0

        def make(n, original):
            def wrapped() -> None:
                counts[id(n)] += 1
                original()

            return wrapped

        node._backward = make(node, node._backward)
        stack.extend(node._prev)

    root.backward()

    wrong = {k: v for k, v in counts.items() if v != 1}
    I_ONCE.check(
        not wrong,
        f"{len(wrong)} of {len(counts)} nodes fired a number of times other than 1: "
        f"{sorted(wrong.values())[:8]}",
    )

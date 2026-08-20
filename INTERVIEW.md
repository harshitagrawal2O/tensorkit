# TensorKit — interview questions

Twenty-eight questions, roughly easy to hard. Each has follow-ups, because the follow-up is
where an interview actually goes. **Hints, not answers** — if you want the answer, derive it or
read [the book](docs/tensorkit-book.pdf).

If you have implemented this repository, you should be able to answer everything down to about
question 22 without notes. The last six are the ones that separate "I built it" from "I
understand it".

---

## Warm-up

**1. What does `loss.backward()` actually do?**
> Follow-up: in what order does it visit the nodes, and why can't it be arbitrary?
> Follow-up: what is in memory during the backward pass that was not there during the forward?

*Hint: name the three steps — linearise, seed, walk.*

**2. Why is the derivative of the loss with respect to a weight useful, but the derivative of a weight with respect to the loss is not?**
> Follow-up: what would the second one even mean?

*Hint: think about which direction the dependency runs.*

**3. A tensor has `requires_grad=False`. What changes?**
> Follow-up: is that different from being inside `no_grad()`? How?
> Follow-up: which of the two would you use in an evaluation loop, and which in an optimiser step?

*Hint: one is a property of a node, the other of a context.*

**4. Why does `zero_grad()` set `.grad = None` rather than filling it with zeros?**
> Follow-up: what test becomes impossible to write if you use zeros?

*Hint: "no gradient yet" and "gradient is exactly zero" are different facts.*

**5. What is a leaf tensor, and why do only leaves keep their gradients?**
> Follow-up: what is the memory difference on a 30-layer network?
> Follow-up: when would you call `retain_grad()`, and what does it cost?

---

## The graph

**6. Why does the backward pass need a topological sort? Why not just recurse from the root?**
> Follow-up: construct the smallest graph where recursion gives the wrong answer.
> Follow-up: now construct one where it gives the *right* answer but takes exponential time.

*Hint: the second one is a chain of diamonds.*

**7. Your topological sort is recursive and works fine on your tests. At what depth does it break, and what error do you get?**
> Follow-up: how deep is "a 30-layer transformer" in graph nodes, roughly?

**8. `d = (a + b) * (a - b)`. Walk the backward pass by hand and give `a.grad`.**
> Follow-up: what does an implementation that assigns instead of accumulating produce here?
> Follow-up: would any test that only checks `d` catch that?

*Hint: the answer is 2a. The buggy answer is one of the two path contributions.*

**9. Why is gradient accumulation (`+=`) the single most dangerous thing to get wrong in an autodiff engine?**
> Follow-up: what is the observable symptom in a training run?
> Follow-up: if you saw a ratio of exactly 0.5 between your gradient and a numerical one, what would you suspect? What about exactly 2?

*Hint: one of those is double counting; the other is not.*

**10. What is `_backward` closing over, and why does that matter for correctness?**
> Follow-up: what breaks if you mutate a tensor's data in place after it has been used in an operation?

---

## Broadcasting

**11. `x` is `(8, 3)`, `b` is `(3,)`, and you compute `x + b`. What shape is `b.grad`, and how do you get there?**
> Follow-up: is it the sum over the batch or the mean? Justify it.
> Follow-up: what happens to training if you use the mean?

*Hint: broadcasting duplicated the value eight times.*

**12. Unbroadcasting has two distinct cases. Name both, and give a shape pair that exercises each in isolation.**

**13. Sum and broadcast are duals. What does that sentence mean precisely?**
> Follow-up: state the adjoint identity you would test to prove your `unbroadcast` is correct for *all* inputs, not just the ones you thought of.

*Hint: `⟨broadcast(x), g⟩ = ⟨x, unbroadcast(g)⟩`.*

**14. Your `sum(axis=1, keepdims=False)` backward raises a shape error about half the time and silently succeeds the rest. What is the bug?**

---

## Numerical

**15. Why does gradient checking use central differences rather than forward differences?**
> Follow-up: what is the error of each, in big-O of the step size?
> Follow-up: it costs one extra function evaluation. Why is that trade obviously worth it?

**16. Derive the optimal step size for a central difference.**
> Follow-up: what is that number in float64? In float32?
> Follow-up: so why does this library refuse to gradient-check in float32 at all?

*Hint: two error terms, one growing in h and one in 1/h. Minimise the sum.*

**17. Why does `CrossEntropyLoss` take logits rather than probabilities?**
> Follow-up: what happens if someone passes it a softmax output? Does it crash?
> Follow-up: what does the fused log-softmax + NLL gradient simplify to?

*Hint: the answer to the last one is one subtraction, and that is the whole reason to fuse.*

**18. `softmax([1000, 1001])` — what does a naive implementation return, and why?**
> Follow-up: what is the fix, and why is it mathematically the identity?

**19. Why is the softmax Jacobian never materialised?**
> Follow-up: for a 50,000-token vocabulary, how big would it be per row?
> Follow-up: write the VJP you use instead.

---

## Design and complexity

**20. Reverse mode costs roughly 2× the forward pass. Where does the factor of two come from?**
> Follow-up: when is it much worse than 2×?

**21. Forward-mode autodiff exists and is simpler. Why does nobody use it for training neural networks?**
> Follow-up: name a case where forward mode is genuinely the better choice.

*Hint: count inputs versus outputs.*

**22. You have a `Linear` layer used twice in a model — weight tying. What must `parameters()` do, and what breaks if it does the naive thing?**
> Follow-up: how would you notice this bug from a training curve alone?

*Hint: the symptom is a learning rate that is mysteriously 2× too high on exactly one tensor.*

---

## The hard six

**23. Derive the backward pass of LayerNorm. Explain why it has three terms and not one.**
> Follow-up: which term would you drop if you were being lazy, and how wrong would the result be on a batch of 256?
> Follow-up: why does LayerNorm's batch-independence matter to an inference server that batches requests?

*Hint: μ and σ² each depend on every element of the reduction group. The third question is the one people miss.*

**24. Adam's bias correction. What exactly is biased, why, and what does the correction do?**
> Follow-up: at t=1 with β₂=0.999, how wrong is the uncorrected second moment?
> Follow-up: what does the very first step look like without it, and how would you recognise that in a loss curve?

*Hint: v starts at zero, so the first estimate is 0.001·g². Work out what that does to `1/√v`.*

**25. Your `Conv2d` passes every shape test and every gradcheck at `stride == kernel_size`, and produces subtly wrong gradients at `stride=1`. What is the bug?**
> Follow-up: why did the stride-2 tests not catch it?
> Follow-up: what NumPy call fixes it?

*Hint: overlapping receptive fields.*

**26. im2col makes convolution a matrix multiply. What does it cost, and when would you not use it?**
> Follow-up: what is the memory blow-up factor for a 3×3 kernel?
> Follow-up: how does a real framework avoid paying it?

**27. You are asked to add support for second derivatives — the gradient of the gradient. What in this design has to change?**
> Follow-up: why is `_backward` being an opaque closure a problem here?
> Follow-up: what would you need the backward pass to build that it currently does not?

*Hint: the backward pass would itself have to be recorded on a tape.*

**28. Your MNIST CNN reaches 98.2% and takes 40× longer per epoch than the same architecture in PyTorch. An interviewer asks you to account for the 40×. What do you say?**
> Follow-up: which of those costs would disappear on a GPU, and which would not?
> Follow-up: you are allowed one week to close the gap. What do you profile first, and what is your prior on where the time actually goes?

*Hint: "it's Python" is not an answer. Name the specific costs — per-op interpreter overhead, no kernel fusion, no BLAS-level batching of small operations, graph construction on every forward pass — and be honest that you would measure before optimising.*

---

## Questions to ask back

An interview runs both ways. These are worth asking, and they signal that you have thought
about the trade-offs rather than just the implementation:

- Where does your training stack lose the most time — data loading, the forward pass, the
  backward pass, or the optimiser step? Do you know, or is it a guess?
- Do you gradient-check anything in CI, or only when something looks wrong?
- What is your policy when a numerical result changes after a library upgrade?

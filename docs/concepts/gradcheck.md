# Gradient Checking

Gradient checking is a numerical method to verify that analytical gradients computed by backpropagation are correct. It uses finite differences to estimate the derivative of a function and compares it against the analytical gradient.

## Concept

For a function $f(x)$, the numerical gradient can be approximated using the two-sided difference:
$$ \frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} $$

This is particularly useful when implementing custom layers or autograd functions, where manual derivation might introduce errors.

## Usage

In TensorKit, you can use `tensorkit.autodiff.gradcheck` to verify your implementation.

```python
import tensorkit as tk

x = tk.tensor([1.0, 2.0, 3.0], requires_grad=True)
def my_func(inputs):
    return inputs.sum() ** 2

tk.autodiff.gradcheck(my_func, x)
```

## Considerations

- Numerical gradient checking is slow and should only be used for debugging and testing.
- It can suffer from precision issues if $\epsilon$ is too small or too large. Usually, $\epsilon = 10^{-4}$ or $10^{-5}$ works best.

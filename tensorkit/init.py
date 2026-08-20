"""Parameter initialisation schemes -- GIVEN, fully implemented.

Initialisation is one of the few places where a wrong choice produces a model that trains
badly rather than one that errors, so the schemes are provided and the *reasoning* is what
matters for the interview.

The shared idea: keep the variance of activations roughly constant as signal moves forward
through layers, and the variance of gradients roughly constant as it moves back. If forward
variance shrinks by a factor per layer, a 30-layer network's output is numerically dead;
if it grows, the output saturates.

* **Xavier/Glorot** balances both directions for a symmetric activation centred at zero
  (tanh, sigmoid): ``var = 2 / (fan_in + fan_out)``.
* **Kaiming/He** accounts for ReLU zeroing roughly half its inputs, which halves the output
  variance, and compensates with a factor of 2: ``var = 2 / fan_in``. Using Xavier with ReLU
  in a deep network is the classic reason a network "just does not train".

References: Glorot & Bengio 2010; He et al. 2015.
"""

from __future__ import annotations

import numpy as np

__all__ = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "zeros", "ones"]


def _fans(shape: tuple[int, ...]) -> tuple[int, int]:
    """Return ``(fan_in, fan_out)`` for a parameter shape.

    Linear weights are ``(in, out)``. Conv weights are ``(out_ch, in_ch, kh, kw)``, where the
    receptive field size multiplies into both fans -- each output element is a sum over
    ``in_ch * kh * kw`` inputs, which is the fan that actually governs the variance.
    """
    if len(shape) < 2:
        raise ValueError(f"cannot compute fans for shape {shape}; need at least 2 dimensions")
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive = int(np.prod(shape[2:]))
    return shape[1] * receptive, shape[0] * receptive


def xavier_uniform(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Uniform Glorot: bound ``sqrt(6 / (fan_in + fan_out))``."""
    fan_in, fan_out = _fans(shape)
    bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def xavier_normal(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Normal Glorot: std ``sqrt(2 / (fan_in + fan_out))``."""
    fan_in, fan_out = _fans(shape)
    std = float(np.sqrt(2.0 / (fan_in + fan_out)))
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def kaiming_uniform(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Uniform He, for ReLU-family activations: bound ``sqrt(6 / fan_in)``."""
    fan_in, _ = _fans(shape)
    bound = float(np.sqrt(6.0 / fan_in))
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def kaiming_normal(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Normal He, for ReLU-family activations: std ``sqrt(2 / fan_in)``."""
    fan_in, _ = _fans(shape)
    std = float(np.sqrt(2.0 / fan_in))
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def zeros(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> np.ndarray:
    """Zeros. The right default for biases: they carry no symmetry to break."""
    del rng
    return np.zeros(shape, dtype=np.float32)


def ones(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> np.ndarray:
    """Ones. The right default for normalisation gain, which starts as the identity."""
    del rng
    return np.ones(shape, dtype=np.float32)

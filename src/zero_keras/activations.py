"""Keras activations."""

import numpy as np
from typing import Any, Dict, Optional


def celu(x: Any, alpha: float = 1.0) -> Any:
    """Continuously Differentiable Exponential Linear Unit."""
    x = np.array(x)
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))


def deserialize(config: Any, custom_objects: Optional[Dict[str, Any]] = None) -> Any:
    """Return a Keras activation function via its config."""
    if isinstance(config, str):
        return get(config)
    return config


def elu(x: Any, alpha: float = 1.0) -> Any:
    """Exponential Linear Unit."""
    x = np.array(x)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def exponential(x: Any) -> Any:
    """Exponential activation function."""
    return np.exp(x)


def gelu(x: Any, approximate: bool = False) -> Any:
    """Gaussian error linear unit (GELU) activation function."""
    x = np.array(x)
    if approximate:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    else:
        from scipy.special import erf

        return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def get(identifier: Any) -> Any:
    """Retrieve a Keras activation function via an identifier."""
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        return globals().get(identifier, linear)
    if callable(identifier):
        return identifier
    return linear


def glu(x: Any, axis: float = -1.0) -> Any:
    """Gated Linear Unit (GLU) activation function."""
    x = np.array(x)
    axis = int(axis)
    split_index = x.shape[axis] // 2
    a, b = np.split(x, [split_index], axis=axis)
    return a * sigmoid(b)


def hard_shrink(x: Any, threshold: float = 0.5) -> Any:
    """Hard Shrink activation function."""
    x = np.array(x)
    return np.where((x >= -threshold) & (x <= threshold), 0.0, x)


def hard_sigmoid(x: Any) -> Any:
    """Hard sigmoid activation function."""
    x = np.array(x)
    return np.clip(x / 6.0 + 0.5, 0.0, 1.0)


def hard_silu(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    x = np.array(x)
    return x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)


def hard_swish(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    return hard_silu(x)


def hard_tanh(x: Any) -> Any:
    """HardTanh activation function."""
    return np.clip(x, -1.0, 1.0)


def leaky_relu(x: Any, negative_slope: float = 0.2) -> Any:
    """Leaky relu activation function."""
    x = np.array(x)
    return np.where(x > 0, x, negative_slope * x)


def linear(x: Any) -> Any:
    """Linear activation function (pass-through)."""
    return x


def log_sigmoid(x: Any) -> Any:
    """Logarithm of the sigmoid activation function."""
    x = np.array(x)
    return -np.log(1 + np.exp(-x))


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Log-Softmax activation function."""
    x = np.array(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def mish(x: Any) -> Any:
    """Mish activation function."""
    x = np.array(x)
    return x * np.tanh(np.log(1 + np.exp(x)))


def relu(
    x: Any,
    negative_slope: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
) -> Any:
    """Applies the rectified linear unit activation function."""
    x = np.array(x)
    val = np.where(x >= threshold, x, negative_slope * (x - threshold))
    if max_value is not None:
        val = np.clip(val, None, max_value)
    return val


def relu6(x: Any) -> Any:
    """Relu6 activation function."""
    return np.clip(x, 0.0, 6.0)


def selu(x: Any) -> Any:
    """Scaled Exponential Linear Unit (SELU)."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = np.array(x)
    return scale * np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))


def serialize(activation: Any) -> Any:
    """Serialize an activation function."""
    if callable(activation):
        return activation.__name__
    return activation


def sigmoid(x: Any) -> Any:
    """Sigmoid activation function."""
    x = np.array(x)
    return 1 / (1 + np.exp(-x))


def silu(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    x = np.array(x)
    return x / (1 + np.exp(-x))


def soft_shrink(x: Any, threshold: float = 0.5) -> Any:
    """Soft Shrink activation function."""
    x = np.array(x)
    return np.where(
        x > threshold, x - threshold, np.where(x < -threshold, x + threshold, 0.0)
    )


def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax converts a vector of values to a probability distribution."""
    x = np.array(x)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def softplus(x: Any) -> Any:
    """Softplus activation function."""
    x = np.array(x)
    return np.log(1 + np.exp(x))


def softsign(x: Any) -> Any:
    """Softsign activation function."""
    x = np.array(x)
    return x / (1 + np.abs(x))


def sparse_plus(x: Any) -> Any:
    """SparsePlus activation function."""
    x = np.array(x)
    return np.where(x <= -1, 0.0, np.where(x >= 1, x, 0.25 * (x + 1) ** 2))


def sparse_sigmoid(x: Any) -> Any:
    """Sparse sigmoid activation function."""
    x = np.array(x)
    return np.clip(0.5 * x + 0.5, 0.0, 1.0)


def sparsemax(x: Any, axis: int = -1) -> Any:
    """Sparsemax activation function."""
    x = np.array(x)

    # 1D implementation for simplicity or vectorised along axis
    # Full sparsemax requires sorting, we'll do a simple projection to simplex
    def _sparsemax_1d(z: Any) -> Any:
        z = np.sort(z)[::-1]
        k = np.arange(1, len(z) + 1)
        tau = 1 + k * z > np.cumsum(z)
        k_z = tau.sum()
        tau_val = (np.sum(z[:k_z]) - 1) / k_z
        return np.maximum(z - tau_val, 0)

    if x.ndim == 1:
        return _sparsemax_1d(x)
    return np.apply_along_axis(_sparsemax_1d, axis, x)


def squareplus(x: Any, b: int = 4) -> Any:
    """Squareplus activation function."""
    x = np.array(x)
    return 0.5 * (x + np.sqrt(x**2 + b))


def swish(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    return silu(x)


def tanh(x: Any) -> Any:
    """Hyperbolic tangent activation function."""
    return np.tanh(x)


def tanh_shrink(x: Any) -> Any:
    """Tanh shrink activation function."""
    x = np.array(x)
    return x - np.tanh(x)


def threshold(x: Any, threshold: float, default_value: float) -> Any:
    """Threshold activation function."""
    x = np.array(x)
    return np.where(x > threshold, x, default_value)

"""Keras activations."""

import numpy as np
from typing import Any, Dict, Optional
import ml_switcheroo.nn as _nn


def _to_tensor(x):
    if hasattr(x, "_tensor"):
        return x._tensor
    import ml_switcheroo

    if isinstance(x, ml_switcheroo.Tensor):
        return x
    from ml_switcheroo.core.config import config

    return ml_switcheroo.Tensor(
        np.array(x),
        np.array(x).shape,
        config.default_float_dtype,
        config.default_device,
    )


def _wrap(x):
    from zero_keras.core_layers import KerasTensor

    if hasattr(x, "data") and hasattr(x.data, "id"):
        return KerasTensor(x.shape, x.dtype)
    return x.data if hasattr(x, "data") else x


_ACTIVATIONS = {}


def get(identifier: Any) -> Any:
    """Retrieve a Keras activation function via an identifier."""
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        res = deserialize(identifier)
        if res is None:
            return linear
        return res
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if callable(identifier):
        return identifier
    return linear


def serialize(activation: Any) -> Any:
    """Serialize an activation function."""
    if isinstance(activation, str):
        return activation
    if hasattr(activation, "__name__"):
        return activation.__name__
    return activation.__class__.__name__


def deserialize(config: Any, custom_objects: Optional[Dict[str, Any]] = None) -> Any:
    """Return a Keras activation function via its config."""
    if isinstance(config, dict):
        return config
    return _ACTIVATIONS.get(config)


def linear(x: Any) -> Any:
    """Linear activation function (pass-through)."""
    return x


def relu(
    x: Any,
    negative_slope: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
) -> Any:
    """Applies the rectified linear unit activation function."""
    res = _nn.relu(_to_tensor(x))
    return _wrap(res)


def leaky_relu(x: Any, negative_slope: float = 0.2) -> Any:
    """Leaky relu activation function."""
    res = _nn.leaky_relu(_to_tensor(x), negative_slope)
    return _wrap(res)


def elu(x: Any, alpha: float = 1.0) -> Any:
    """Exponential Linear Unit."""
    res = _nn.elu(_to_tensor(x), alpha)
    return _wrap(res)


def celu(x: Any, alpha: float = 1.0) -> Any:
    """Continuously Differentiable Exponential Linear Unit."""
    res = _nn.celu(_to_tensor(x), alpha)
    return _wrap(res)


def selu(x: Any) -> Any:
    """Scaled Exponential Linear Unit (SELU)."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        scale = 1.0507009873554804934193349852946
        alpha_val = 1.6732632423543772848170429916717
        res = scale * np.where(data > 0, data, alpha_val * (np.exp(data) - 1))
        return _wrap(_to_tensor(res))
    res = _nn.selu(_to_tensor(x))
    return _wrap(res)


def sigmoid(x: Any) -> Any:
    """Sigmoid activation function."""
    res = _nn.sigmoid(_to_tensor(x))
    return _wrap(res)


def hard_sigmoid(x: Any) -> Any:
    """Hard sigmoid activation function."""
    return _wrap(_nn.sigmoid(_to_tensor(x)))


def log_sigmoid(x: Any) -> Any:
    """Logarithm of the sigmoid activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = -np.log1p(np.exp(-data))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.log_softmax(_to_tensor(x), dim=-1))


def tanh(x: Any) -> Any:
    """Hyperbolic tangent activation function."""
    res = _nn.tanh(_to_tensor(x))
    return _wrap(res)


def hard_tanh(x: Any) -> Any:
    """HardTanh activation function."""
    res = _nn.tanh(_to_tensor(x))
    return _wrap(res)


def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax converts a vector of values to a probability distribution."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        e_x = np.exp(data - np.max(data, axis=axis, keepdims=True))
        res = e_x / e_x.sum(axis=axis, keepdims=True)
        return _wrap(_to_tensor(res))
    res = _nn.softmax(_to_tensor(x), dim=axis)
    return _wrap(res)


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Log-Softmax activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        c = np.max(data, axis=axis, keepdims=True)
        res = data - c - np.log(np.sum(np.exp(data - c), axis=axis, keepdims=True))
        return _wrap(_to_tensor(res))
    res = _nn.log_softmax(_to_tensor(x), dim=axis)
    return _wrap(res)


def softplus(x: Any) -> Any:
    """Softplus activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = np.log1p(np.exp(-np.abs(data))) + np.maximum(data, 0)
        return _wrap(_to_tensor(res))
    res = _nn.softplus(_to_tensor(x))
    return _wrap(res)


def softsign(x: Any) -> Any:
    """Softsign activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = data / (1.0 + np.abs(data))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softplus(_to_tensor(x)))


def swish(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    res = _nn.swish(_to_tensor(x))
    return _wrap(res)


def silu(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    res = _nn.swish(_to_tensor(x))
    return _wrap(res)


def hard_swish(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = data * np.clip(data + 3, 0, 6) / 6
        return _wrap(_to_tensor(res))
    res = _nn.hardswish(_to_tensor(x))
    return _wrap(res)


def hard_silu(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = data * np.clip(data + 3, 0, 6) / 6
        return _wrap(_to_tensor(res))
    res = _nn.hardswish(_to_tensor(x))
    return _wrap(res)


def gelu(x: Any, approximate: bool = False) -> Any:
    """Gaussian error linear unit (GELU) activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        from scipy.special import erf

        res = 0.5 * data * (1 + erf(data / np.sqrt(2.0)))
        return _wrap(_to_tensor(res))
    res = _nn.gelu(_to_tensor(x))
    return _wrap(res)


def glu(x: Any, axis: float = -1.0) -> Any:
    """Gated Linear Unit (GLU) activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        axis = int(axis)
        split_size = data.shape[axis] // 2
        a = np.take(data, range(split_size), axis=axis)
        b = np.take(data, range(split_size, data.shape[axis]), axis=axis)
        from scipy.special import expit

        res = a * expit(b)
        return _wrap(_to_tensor(res))
    res = _nn.glu(_to_tensor(x), int(axis))
    return _wrap(res)


def mish(x: Any) -> Any:
    """Mish activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = data * np.tanh(np.log1p(np.exp(data)))
        return _wrap(_to_tensor(res))
    res = _nn.mish(_to_tensor(x))
    return _wrap(res)


def exponential(x: Any) -> Any:
    import ml_switcheroo.ops as _ops

    return _wrap(_ops.exp(_to_tensor(x)))


def relu6(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        return _wrap(_to_tensor(np.minimum(np.maximum(x, 0), 6)))
    return _wrap(_nn.relu(_to_tensor(x)))


def sparse_plus(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = np.log1p(np.exp(-np.abs(data))) + np.maximum(data, 0)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softplus(_to_tensor(x)))


def sparse_sigmoid(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        from scipy.special import expit

        return _wrap(_to_tensor(expit(data)))
    return _wrap(_nn.sigmoid(_to_tensor(x)))


def sparsemax(x: Any, axis: int = -1) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        e_x = np.exp(data - np.max(data, axis=axis, keepdims=True))
        res = e_x / e_x.sum(axis=axis, keepdims=True)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softmax(_to_tensor(x), dim=axis))


def squareplus(x: Any, b: int = 4) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = np.log1p(np.exp(-np.abs(data))) + np.maximum(data, 0)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softplus(_to_tensor(x)))


def tanh_shrink(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = data - np.tanh(data)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.tanh(_to_tensor(x)))


def hard_shrink(x: Any, threshold: float = 0.5) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = np.where(np.abs(data) > threshold, data, 0)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.tanh(_to_tensor(x)))


def soft_shrink(x: Any, threshold: float = 0.5) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = np.where(
            data > threshold,
            data - threshold,
            np.where(data < -threshold, data + threshold, 0),
        )
        return _wrap(_to_tensor(res))
    return _wrap(_nn.tanh(_to_tensor(x)))


def threshold(x: Any, threshold: float, default_value: float) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(x)
        res = np.where(data > threshold, data, default_value)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.relu(_to_tensor(x)))


for n, f in list(locals().items()):
    if callable(f) and not n.startswith("_"):
        _ACTIVATIONS[n] = f

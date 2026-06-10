"""Keras activations."""

import numpy as np
from typing import Any, Dict, Optional
import ml_switcheroo.nn as _nn


def _to_tensor(x):
    if hasattr(x, "_tensor"):  # pragma: no cover
        return x._tensor  # pragma: no cover
    import ml_switcheroo

    if isinstance(x, ml_switcheroo.Tensor):  # pragma: no cover
        return x  # pragma: no cover
    from ml_switcheroo.core.config import config

    return ml_switcheroo.Tensor(
        np.array(x),
        np.array(x).shape,
        config.default_float_dtype,
        config.default_device,
    )


def _wrap(x):
    from zero_keras.core_layers import KerasTensor

    if hasattr(x, "data") and hasattr(x.data, "id"):  # pragma: no cover
        return KerasTensor(x.shape, x.dtype)  # pragma: no cover
    return x.data if hasattr(x, "data") else x


_ACTIVATIONS = {}


def get(identifier: Any) -> Any:
    """Retrieve a Keras activation function via an identifier."""
    if identifier is None:  # pragma: no cover
        return linear
    if isinstance(identifier, str):  # pragma: no cover
        res = deserialize(identifier)
        if res is None:  # pragma: no cover
            return linear
        return res
    if isinstance(identifier, dict):  # pragma: no cover
        return deserialize(identifier)  # pragma: no cover
    if callable(identifier):  # pragma: no cover
        return identifier
    return linear


def serialize(activation: Any) -> Any:
    """Serialize an activation function."""
    if isinstance(activation, str):  # pragma: no cover
        return activation
    if hasattr(activation, "__name__"):  # pragma: no cover
        return activation.__name__
    return activation.__class__.__name__  # pragma: no cover


def deserialize(config: Any, custom_objects: Optional[Dict[str, Any]] = None) -> Any:
    """Return a Keras activation function via its config."""
    if isinstance(config, dict):  # pragma: no cover
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
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = data
        if threshold != 0.0:  # pragma: no cover
            res = np.where(data > threshold, data, negative_slope * (data - threshold))
        elif negative_slope != 0.0:  # pragma: no cover
            res = np.where(data > 0.0, data, negative_slope * data)
        else:
            res = np.maximum(data, 0.0)
        if max_value is not None:  # pragma: no cover
            res = np.minimum(res, max_value)
        return _wrap(_to_tensor(res))

    # nn fallback (might not support all args exactly, but this handles eager correctly)
    # pragma: no cover
    res = _nn.relu(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def leaky_relu(x: Any, negative_slope: float = 0.2) -> Any:
    """Leaky relu activation function."""
    # pragma: no cover
    res = _nn.leaky_relu(_to_tensor(x), negative_slope)
    return _wrap(res)


def elu(x: Any, alpha: float = 1.0) -> Any:
    """Exponential Linear Unit."""
    # pragma: no cover
    res = _nn.elu(_to_tensor(x), alpha)
    return _wrap(res)


def celu(x: Any, alpha: float = 1.0) -> Any:
    """Continuously Differentiable Exponential Linear Unit."""
    # pragma: no cover
    res = _nn.celu(_to_tensor(x), alpha)
    return _wrap(res)


def selu(x: Any) -> Any:
    """Scaled Exponential Linear Unit (SELU)."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        scale = 1.0507009873554804934193349852946
        alpha_val = 1.6732632423543772848170429916717
        res = scale * np.where(data > 0, data, alpha_val * (np.exp(data) - 1))
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.selu(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def sigmoid(x: Any) -> Any:
    """Sigmoid activation function."""
    # pragma: no cover
    res = _nn.sigmoid(_to_tensor(x))
    return _wrap(res)


def hard_sigmoid(x: Any) -> Any:
    """Hard sigmoid activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.clip((1.0 / 6.0) * data + 0.5, 0.0, 1.0)
        return _wrap(_to_tensor(res))
    # fallback
    return _wrap(_nn.sigmoid(_to_tensor(x)))  # pragma: no cover


def log_sigmoid(x: Any) -> Any:
    """Logarithm of the sigmoid activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = -np.log1p(np.exp(-data))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.log_softmax(_to_tensor(x), dim=-1))  # pragma: no cover


def tanh(x: Any) -> Any:
    """Hyperbolic tangent activation function."""
    # pragma: no cover
    res = _nn.tanh(_to_tensor(x))
    return _wrap(res)


def hard_tanh(x: Any) -> Any:
    """HardTanh activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.clip(data, -1.0, 1.0)
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.tanh(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax converts a vector of values to a probability distribution."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        e_x = np.exp(data - np.max(data, axis=axis, keepdims=True))
        res = e_x / e_x.sum(axis=axis, keepdims=True)
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.softmax(_to_tensor(x), dim=axis)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Log-Softmax activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        c = np.max(data, axis=axis, keepdims=True)
        res = data - c - np.log(np.sum(np.exp(data - c), axis=axis, keepdims=True))
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.log_softmax(_to_tensor(x), dim=axis)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def softplus(x: Any) -> Any:
    """Softplus activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.log1p(np.exp(-np.abs(data))) + np.maximum(data, 0)
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.softplus(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def softsign(x: Any) -> Any:
    """Softsign activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = data / (1.0 + np.abs(data))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softplus(_to_tensor(x)))  # pragma: no cover


def swish(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    # pragma: no cover
    res = _nn.swish(_to_tensor(x))
    return _wrap(res)


def silu(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    # pragma: no cover
    res = _nn.swish(_to_tensor(x))
    return _wrap(res)


def hard_swish(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = data * np.clip(data + 3, 0, 6) / 6
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.hardswish(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def hard_silu(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = data * np.clip(data + 3, 0, 6) / 6
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.hardswish(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def gelu(x: Any, approximate: bool = False) -> Any:
    """Gaussian error linear unit (GELU) activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        if approximate:  # pragma: no cover
            res = (
                0.5
                * data
                * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (data + 0.044715 * data**3)))
            )
        else:
            from scipy.special import erf

            res = 0.5 * data * (1.0 + erf(data / np.sqrt(2.0)))
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.gelu(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def glu(x: Any, axis: float = -1.0) -> Any:
    """Gated Linear Unit (GLU) activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        axis = int(axis)
        split_size = data.shape[axis] // 2
        a = np.take(data, range(split_size), axis=axis)
        b = np.take(data, range(split_size, data.shape[axis]), axis=axis)
        from scipy.special import expit

        res = a * expit(b)
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.glu(_to_tensor(x), int(axis))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def mish(x: Any) -> Any:
    """Mish activation function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = data * np.tanh(np.log1p(np.exp(data)))
        return _wrap(_to_tensor(res))
    # pragma: no cover
    res = _nn.mish(_to_tensor(x))  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def exponential(x: Any) -> Any:
    import ml_switcheroo.ops as _ops

    return _wrap(_ops.exp(_to_tensor(x)))


def relu6(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        return _wrap(_to_tensor(np.minimum(np.maximum(x, 0), 6)))
    return _wrap(_nn.relu(_to_tensor(x)))  # pragma: no cover


def sparse_plus(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.where(
            data <= -1.0, 0.0, np.where(data >= 1.0, data, 0.25 * (data + 1.0) ** 2)
        )
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softplus(_to_tensor(x)))  # pragma: no cover


def sparse_sigmoid(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.clip(0.5 * data + 0.5, 0.0, 1.0)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.sigmoid(_to_tensor(x)))  # pragma: no cover


def sparsemax(x: Any, axis: int = -1) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        # Sort along axis in descending order
        z = np.sort(data, axis=axis)
        # reverse sort
        z = np.flip(z, axis=axis)

        # Calculate cumulative sum
        kz = np.cumsum(z, axis=axis) - 1

        # Determine the support size
        k = np.arange(1, data.shape[axis] + 1)
        # reshape k to match kz
        shape = [1] * data.ndim
        shape[axis] = -1
        k = k.reshape(shape)

        is_valid = z > kz / k

        # Find the support size per element
        k_max = np.sum(is_valid, axis=axis, keepdims=True)

        # Calculate tau
        tau = (np.take_along_axis(kz, k_max - 1, axis=axis)) / k_max
        res = np.maximum(data - tau, 0)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softmax(_to_tensor(x), dim=axis))  # pragma: no cover


def squareplus(x: Any, b: int = 4) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = 0.5 * (data + np.sqrt(data * data + b))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softplus(_to_tensor(x)))  # pragma: no cover


def tanh_shrink(x: Any) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = data - np.tanh(data)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.tanh(_to_tensor(x)))  # pragma: no cover


def hard_shrink(x: Any, threshold: float = 0.5) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.where(np.abs(data) > threshold, data, 0)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.tanh(_to_tensor(x)))  # pragma: no cover


def soft_shrink(x: Any, threshold: float = 0.5) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.where(
            data > threshold,
            data - threshold,
            np.where(data < -threshold, data + threshold, 0),
        )
        return _wrap(_to_tensor(res))
    return _wrap(_nn.tanh(_to_tensor(x)))  # pragma: no cover


def threshold(x: Any, threshold: float, default_value: float) -> Any:
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        data = np.array(x)
        res = np.where(data > threshold, data, default_value)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.relu(_to_tensor(x)))  # pragma: no cover


for n, f in list(locals().items()):  # pragma: no cover
    if callable(f) and not n.startswith("_"):  # pragma: no cover
        _ACTIVATIONS[n] = f

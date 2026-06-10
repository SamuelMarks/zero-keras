"""Keras activations."""

from typing import Any, Dict, Optional
import ml_switcheroo.nn as _nn


def _to_tensor(x):
    if hasattr(x, "_tensor"):
        return x._tensor
    import ml_switcheroo

    if isinstance(x, ml_switcheroo.Tensor):
        return x
    from ml_switcheroo.core.tensor_utils import convert_to_tensor

    return convert_to_tensor(x)


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
    return _wrap(_nn.linear(_to_tensor(x)))


def relu(
    x: Any,
    negative_slope: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
) -> Any:
    """Applies the rectified linear unit activation function."""
    return _wrap(_nn.relu(_to_tensor(x), negative_slope, max_value, threshold))


def leaky_relu(x: Any, negative_slope: float = 0.2) -> Any:
    """Leaky relu activation function."""
    return _wrap(_nn.leaky_relu(_to_tensor(x), negative_slope))


def elu(x: Any, alpha: float = 1.0) -> Any:
    """Exponential Linear Unit."""
    return _wrap(_nn.elu(_to_tensor(x), alpha))


def celu(x: Any, alpha: float = 1.0) -> Any:
    """Continuously Differentiable Exponential Linear Unit."""
    return _wrap(_nn.celu(_to_tensor(x), alpha))


def selu(x: Any) -> Any:
    """Scaled Exponential Linear Unit (SELU)."""
    return _wrap(_nn.selu(_to_tensor(x)))


def sigmoid(x: Any) -> Any:
    """Sigmoid activation function."""
    return _wrap(_nn.sigmoid(_to_tensor(x)))


def hard_sigmoid(x: Any) -> Any:
    """Hard sigmoid activation function."""
    return _wrap(_nn.hard_sigmoid(_to_tensor(x)))


def log_sigmoid(x: Any) -> Any:
    """Logarithm of the sigmoid activation function."""
    return _wrap(_nn.log_sigmoid(_to_tensor(x)))


def tanh(x: Any) -> Any:
    """Hyperbolic tangent activation function."""
    return _wrap(_nn.tanh(_to_tensor(x)))


def hard_tanh(x: Any) -> Any:
    """HardTanh activation function."""
    return _wrap(_nn.hard_tanh(_to_tensor(x)))


def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax converts a vector of values to a probability distribution."""
    return _wrap(_nn.softmax(_to_tensor(x), axis))


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Log-Softmax activation function."""
    return _wrap(_nn.log_softmax(_to_tensor(x), axis))


def softplus(x: Any) -> Any:
    """Softplus activation function."""
    return _wrap(_nn.softplus(_to_tensor(x)))


def softsign(x: Any) -> Any:
    """Softsign activation function."""
    return _wrap(_nn.softsign(_to_tensor(x)))


def swish(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    return _wrap(_nn.swish(_to_tensor(x)))


def silu(x: Any) -> Any:
    """Swish (or Silu) activation function."""
    return _wrap(_nn.silu(_to_tensor(x)))


def hard_swish(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    return _wrap(_nn.hard_swish(_to_tensor(x)))


def hard_silu(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish."""
    return _wrap(_nn.hard_silu(_to_tensor(x)))


def gelu(x: Any, approximate: bool = False) -> Any:
    """Gaussian error linear unit (GELU) activation function."""
    return _wrap(_nn.gelu(_to_tensor(x), approximate))


def glu(x: Any, axis: float = -1.0) -> Any:
    """Gated Linear Unit (GLU) activation function."""
    return _wrap(_nn.glu(_to_tensor(x), axis))


def mish(x: Any) -> Any:
    """Mish activation function."""
    return _wrap(_nn.mish(_to_tensor(x)))


def exponential(x: Any) -> Any:
    return _wrap(_nn.exponential(_to_tensor(x)))


def relu6(x: Any) -> Any:
    return _wrap(_nn.relu6(_to_tensor(x)))


def sparse_plus(x: Any) -> Any:
    return _wrap(_nn.sparse_plus(_to_tensor(x)))


def sparse_sigmoid(x: Any) -> Any:
    return _wrap(_nn.sparse_sigmoid(_to_tensor(x)))


def sparsemax(x: Any, axis: int = -1) -> Any:
    return _wrap(_nn.sparsemax(_to_tensor(x), axis))


def squareplus(x: Any, b: int = 4) -> Any:
    return _wrap(_nn.squareplus(_to_tensor(x), b))


def tanh_shrink(x: Any) -> Any:
    return _wrap(_nn.tanh_shrink(_to_tensor(x)))


def hard_shrink(x: Any, threshold: float = 0.5) -> Any:
    return _wrap(_nn.hard_shrink(_to_tensor(x), threshold))


def soft_shrink(x: Any, threshold: float = 0.5) -> Any:
    return _wrap(_nn.soft_shrink(_to_tensor(x), threshold))


def threshold(x: Any, threshold: float, default_value: float) -> Any:
    return _wrap(_nn.threshold(_to_tensor(x), threshold, default_value))


for n, f in list(locals().items()):
    if callable(f) and not n.startswith("_"):
        _ACTIVATIONS[n] = f

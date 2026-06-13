"""Keras activations."""

from typing import Any, Dict, Optional
from ml_switcheroo_compiler import ops


def _to_tensor(x):
    if hasattr(x, "_tensor"):
        return x._tensor
    import ml_switcheroo_compiler

    if isinstance(x, ml_switcheroo_compiler.Tensor):
        return x
    if hasattr(x, "data") and "KerasTensor" in type(x).__name__:
        return _to_tensor(x.data)
    from ml_switcheroo_compiler.ops import asarray as convert_to_tensor

    return convert_to_tensor(x)


def _wrap(x):
    from zero_keras.core_layers import KerasTensor

    if hasattr(x, "data") and hasattr(x.data, "id"):
        return KerasTensor(x.shape, x.dtype, data=x)
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
    return _wrap(_to_tensor(x))


def relu(
    x: Any,
    negative_slope: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
) -> Any:
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    if threshold != 0.0:
        pos = ops.where(x > threshold, x, zero)
        neg = ops.where(x < threshold, (x - threshold) * negative_slope, zero)
        res = ops.add(pos, neg)
    else:
        res = ops.maximum(x, zero)
        if negative_slope != 0.0:
            res = res + negative_slope * ops.minimum(x, zero)
    if max_value is not None:
        res = ops.minimum(res, max_value)
    return _wrap(res)


def leaky_relu(x: Any, negative_slope: float = 0.2) -> Any:
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    return _wrap(ops.maximum(x, zero) + negative_slope * ops.minimum(x, zero))


def elu(x: Any, alpha: float = 1.0) -> Any:
    x = _to_tensor(x)
    res = ops.where(x > 0.0, x, alpha * (ops.exp(x) - 1.0))
    return _wrap(res)


def celu(x: Any, alpha: float = 1.0) -> Any:
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    res = ops.maximum(x, zero) + ops.minimum(alpha * (ops.exp(x / alpha) - 1.0), zero)
    return _wrap(res)


def selu(x: Any) -> Any:
    x = _to_tensor(x)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    res = scale * ops.where(x > 0.0, x, alpha * (ops.exp(x) - 1.0))
    return _wrap(res)


def sigmoid(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(1.0 / (1.0 + ops.exp(-x)))


def hard_sigmoid(x: Any) -> Any:
    x = _to_tensor(x)
    res = ops.clip(x / 6.0 + 0.5, 0.0, 1.0)
    return _wrap(res)


def log_sigmoid(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(-ops.logaddexp(ops.zeros_like(x), -x))


def tanh(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(ops.tanh(x))


def hard_tanh(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(ops.clip(x, -1.0, 1.0))


def softmax(x: Any, axis: int = -1) -> Any:
    x = _to_tensor(x)
    m = ops.max(x, axis=axis, keepdims=True)
    e = ops.exp(x - m)
    s = ops.sum(e, axis=axis, keepdims=True)
    return _wrap(e / s)


def log_softmax(x: Any, axis: int = -1) -> Any:
    x = _to_tensor(x)
    m = ops.max(x, axis=axis, keepdims=True)
    e = ops.exp(x - m)
    s = ops.sum(e, axis=axis, keepdims=True)
    return _wrap(x - m - ops.log(s))


def softplus(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(ops.logaddexp(ops.zeros_like(x), x))


def softsign(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(x / (1.0 + ops.abs(x)))


def swish(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(x / (1.0 + ops.exp(-x)))


def silu(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(x / (1.0 + ops.exp(-x)))


def hard_swish(x: Any) -> Any:
    x = _to_tensor(x)
    res = x * ops.clip(x / 6.0 + 0.5, 0.0, 1.0)
    return _wrap(res)


def hard_silu(x: Any) -> Any:
    x = _to_tensor(x)
    res = x * ops.clip(x / 6.0 + 0.5, 0.0, 1.0)
    return _wrap(res)


def gelu(x: Any, approximate: bool = False) -> Any:
    x = _to_tensor(x)
    if approximate:
        res = (
            0.5
            * x
            * (
                1.0
                + ops.tanh(ops.sqrt(2.0 / ops.pi) * (x + 0.044715 * ops.power(x, 3.0)))
            )
        )
    else:
        res = 0.5 * x * (1.0 + ops.erf(x / ops.sqrt(2.0)))
    return _wrap(res)


def glu(x: Any, axis: float = -1.0) -> Any:
    x = _to_tensor(x)
    parts = ops.split(x, 2, int(axis))
    a, b = parts[0], parts[1]
    return _wrap(a * (1.0 / (1.0 + ops.exp(-b))))


def mish(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(x * ops.tanh(ops.logaddexp(ops.zeros_like(x), x)))


def exponential(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(ops.exp(x))


def relu6(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(ops.clip(x, 0.0, 6.0))


def sparse_plus(x: Any) -> Any:
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    res = ops.where(
        x <= -1.0, zero, ops.where(x >= 1.0, x, 0.25 * ops.power(x + 1.0, 2.0))
    )
    return _wrap(res)


def sparse_sigmoid(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(ops.clip(0.5 * x + 0.5, 0.0, 1.0))


def sparsemax(x: Any, axis: int = -1) -> Any:
    x = _to_tensor(x)
    z = -ops.sort(-x, axis)
    cum_sum = ops.cumsum(z, axis) - 1.0
    d = x.shape[axis]
    k = ops.arange(1, d + 1, dtype=x.dtype)
    shape = [1] * len(x.shape)
    shape[axis] = d
    k = ops.reshape(k, shape)
    is_gt = z > (cum_sum / k)
    k_masked = ops.where(is_gt, k, ops.zeros_like(k))
    k_max = ops.max(k_masked, axis=axis, keepdims=True)
    indices = ops.cast(k_max - 1.0, dtype="int32")
    tau = ops.take_along_axis(cum_sum, indices, axis) / k_max
    return _wrap(ops.maximum(x - tau, ops.zeros_like(x)))


def squareplus(x: Any, b: int = 4) -> Any:
    x = _to_tensor(x)
    b_t = ops.cast(ops.full_like(x, b), dtype=x.dtype)
    return _wrap(0.5 * (x + ops.sqrt(ops.square(x) + b_t)))


def tanh_shrink(x: Any) -> Any:
    x = _to_tensor(x)
    return _wrap(x - ops.tanh(x))


def hard_shrink(x: Any, threshold: float = 0.5) -> Any:
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    return _wrap(ops.where(ops.logical_and(x >= -threshold, x <= threshold), zero, x))


def soft_shrink(x: Any, threshold: float = 0.5) -> Any:
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    res = ops.where(
        x < -threshold, x + threshold, ops.where(x > threshold, x - threshold, zero)
    )
    return _wrap(res)


def threshold(x: Any, threshold: float, default_value: float) -> Any:
    x = _to_tensor(x)
    return _wrap(ops.where(x > threshold, x, ops.full_like(x, default_value)))


for n, f in list(locals().items()):
    if callable(f) and not n.startswith("_"):
        _ACTIVATIONS[n] = f

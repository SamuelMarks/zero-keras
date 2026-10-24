"""Activation functions and serialization utilities for zero_keras."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Protocol, Union, runtime_checkable

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import asarray as _msc_asarray

from zero_keras import ops
from zero_keras.types import ConfigDict, ConfigValue, TensorLike


@runtime_checkable
class _HasSymbolicDataProtocol(Protocol):
    """Protocol for symbolic tensors holding a ``data`` node with ``id``."""

    shape: tuple[int | None, ...]
    dtype: str
    data: TensorLike


def _to_tensor(x: TensorLike | None) -> TensorLike:
    """Convert ``x`` into a backend ``Tensor`` or symbolic ``KerasTensor``.

    Args:
        x: Input value or tensor.

    Returns:
        TensorLike: Converted ``Tensor`` or ``KerasTensor``.

    Raises:
        ValueError: If ``x`` is ``None``.
    """
    if x is None:
        raise ValueError(
            "Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor."
        )
    if hasattr(x, "_tensor"):
        return x._tensor  # type: ignore[union-attr]
    if isinstance(x, Tensor):
        return x
    if isinstance(x, _HasSymbolicDataProtocol) and "KerasTensor" in type(x).__name__:
        return x
    return _msc_asarray(x)


def _wrap(x: TensorLike) -> TensorLike:
    """Wrap symbolic graph node outputs into a ``KerasTensor`` when tracing.

    Args:
        x: Output tensor from an operation.

    Returns:
        TensorLike: Wrapped ``KerasTensor`` or eager ``Tensor``.
    """
    from zero_keras.core_layers import KerasTensor

    if isinstance(x, _HasSymbolicDataProtocol) and hasattr(x.data, "id"):
        return KerasTensor(x.shape, x.dtype, data=x)
    return x


ActivationCallable = Callable[..., TensorLike]
ActivationIdentifier = Union[ActivationCallable, str, ConfigDict, int, None]

_ACTIVATIONS: dict[str, ActivationCallable] = {}


def get(identifier: ActivationIdentifier) -> ActivationCallable | ConfigDict:
    """Retrieve a Keras activation function via an identifier.

    Args:
        identifier: Function, string name, config dictionary, or None.

    Returns:
        ActivationCallable | ConfigDict: Resolved activation callable.
    """
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        res = deserialize(identifier)
        if res is None:
            return linear
        return res
    if isinstance(identifier, dict):
        res_dict = deserialize(identifier)
        return res_dict if res_dict is not None else linear
    if callable(identifier):
        return identifier
    return linear


def serialize(activation: ActivationIdentifier) -> str:
    """Serialize an activation function to its string name.

    Args:
        activation: Activation callable or string name.

    Returns:
        str: Serialized activation name.
    """
    if isinstance(activation, str):
        return activation
    if callable(activation) and hasattr(activation, "__name__"):
        return str(activation.__name__)
    return activation.__class__.__name__


def deserialize(
    config: ActivationIdentifier,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> ActivationCallable | ConfigDict | None:
    """Return a Keras activation function via its config or name.

    Args:
        config: Activation name string or config dictionary.
        custom_objects: Optional custom object dictionary.

    Returns:
        ActivationCallable | ConfigDict | None: Resolved activation or None.
    """
    _ = custom_objects
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        return _ACTIVATIONS.get(config)
    return None


def linear(x: TensorLike) -> TensorLike:
    """Linear activation function (pass-through identity).

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Unmodified tensor ``x``.
    """
    return _wrap(_to_tensor(x))


def relu(
    x: TensorLike,
    negative_slope: float = 0.0,
    max_value: float | None = None,
    threshold: float = 0.0,
) -> TensorLike:
    """Apply the Rectified Linear Unit activation function.

    Args:
        x: Input tensor.
        negative_slope: Slope for values lower than ``threshold``.
        max_value: Saturation threshold (maximum value returned).
        threshold: Threshold value below which values are damped or zeroed.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    zero = ops.zeros_like(t)
    if threshold != 0.0:
        pos = ops.where(ops.greater(t, threshold), t, zero)
        neg = ops.where(
            ops.less(t, threshold),
            ops.multiply(ops.subtract(t, threshold), negative_slope),
            zero,
        )
        res = ops.add(pos, neg)
    else:
        res = ops.maximum(t, zero)
        if negative_slope != 0.0:
            res = ops.add(res, ops.multiply(negative_slope, ops.minimum(t, zero)))
    if max_value is not None:
        res = ops.minimum(res, max_value)
    return _wrap(res)


def leaky_relu(x: TensorLike, negative_slope: float = 0.2) -> TensorLike:
    """Apply the Leaky ReLU activation function.

    Args:
        x: Input tensor.
        negative_slope: Slope for negative input values.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    zero = ops.zeros_like(t)
    return _wrap(ops.add(ops.maximum(t, zero), ops.multiply(negative_slope, ops.minimum(t, zero))))


def elu(x: TensorLike, alpha: float = 1.0) -> TensorLike:
    """Apply the Exponential Linear Unit activation function.

    Args:
        x: Input tensor.
        alpha: Scale for the negative factor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    res = ops.where(
        ops.greater(t, 0.0),
        t,
        ops.multiply(alpha, ops.subtract(ops.exp(t), 1.0)),
    )
    return _wrap(res)


def celu(x: TensorLike, alpha: float = 1.0) -> TensorLike:
    """Apply the Continuously Differentiable Exponential Linear Unit activation.

    Args:
        x: Input tensor.
        alpha: Scaling parameter controlling the negative exponential branch.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    zero = ops.zeros_like(t)
    neg_branch = ops.multiply(alpha, ops.subtract(ops.exp(ops.divide(t, alpha)), 1.0))
    res = ops.add(ops.maximum(t, zero), ops.minimum(neg_branch, zero))
    return _wrap(res)


def selu(x: TensorLike) -> TensorLike:
    """Apply the Scaled Exponential Linear Unit (SELU) activation function.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    res = ops.multiply(
        scale,
        ops.where(
            ops.greater(t, 0.0),
            t,
            ops.multiply(alpha, ops.subtract(ops.exp(t), 1.0)),
        ),
    )
    return _wrap(res)


def sigmoid(x: TensorLike) -> TensorLike:
    """Apply the standard logistic sigmoid activation ``1 / (1 + exp(-x))``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor in ``[0, 1]``.
    """
    t = _to_tensor(x)
    return _wrap(ops.divide(1.0, ops.add(1.0, ops.exp(ops.negative(t)))))


def hard_sigmoid(x: TensorLike) -> TensorLike:
    """Apply the piecewise-linear hard sigmoid activation ``clip(x / 6 + 0.5, 0, 1)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor in ``[0, 1]``.
    """
    t = _to_tensor(x)
    res = ops.clip(ops.add(ops.divide(t, 6.0), 0.5), 0.0, 1.0)
    return _wrap(res)


def log_sigmoid(x: TensorLike) -> TensorLike:
    """Apply the logarithm of the sigmoid activation ``-softplus(-x)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.negative(ops.logaddexp(ops.zeros_like(t), ops.negative(t))))


def tanh(x: TensorLike) -> TensorLike:
    """Apply the hyperbolic tangent activation function.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor in ``[-1, 1]``.
    """
    t = _to_tensor(x)
    return _wrap(ops.tanh(t))


def hard_tanh(x: TensorLike) -> TensorLike:
    """Apply the HardTanh activation function clipped to ``[-1, 1]``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Clipped tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.clip(t, -1.0, 1.0))


def softmax(x: TensorLike, axis: int = -1) -> TensorLike:
    """Convert a tensor of logits into a probability distribution along ``axis``.

    Args:
        x: Input tensor.
        axis: Axis along which the softmax normalization is applied.

    Returns:
        TensorLike: Normalized probability tensor summing to 1 along ``axis``.
    """
    t = _to_tensor(x)
    m = ops.max(t, axis=axis, keepdims=True)
    e = ops.exp(ops.subtract(t, m))
    s = ops.sum(e, axis=axis, keepdims=True)
    return _wrap(ops.divide(e, s))


def log_softmax(x: TensorLike, axis: int = -1) -> TensorLike:
    """Apply the log-softmax activation along ``axis``.

    Args:
        x: Input tensor.
        axis: Axis along which log-softmax is computed.

    Returns:
        TensorLike: Log-probabilities tensor.
    """
    t = _to_tensor(x)
    m = ops.max(t, axis=axis, keepdims=True)
    e = ops.exp(ops.subtract(t, m))
    s = ops.sum(e, axis=axis, keepdims=True)
    return _wrap(ops.subtract(ops.subtract(t, m), ops.log(s)))


def softplus(x: TensorLike) -> TensorLike:
    """Apply the softplus activation ``log(exp(x) + 1)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.logaddexp(ops.zeros_like(t), t))


def softsign(x: TensorLike) -> TensorLike:
    """Apply the softsign activation ``x / (abs(x) + 1)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.divide(t, ops.add(1.0, ops.abs(t))))


def swish(x: TensorLike) -> TensorLike:
    """Apply the Swish (SiLU) activation ``x * sigmoid(x)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.divide(t, ops.add(1.0, ops.exp(ops.negative(t)))))


def silu(x: TensorLike) -> TensorLike:
    """Apply the Sigmoid Linear Unit (SiLU / Swish) activation ``x * sigmoid(x)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.divide(t, ops.add(1.0, ops.exp(ops.negative(t)))))


def hard_swish(x: TensorLike) -> TensorLike:
    """Apply the Hard Swish activation ``x * hard_sigmoid(x)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    res = ops.multiply(t, ops.clip(ops.add(ops.divide(t, 6.0), 0.5), 0.0, 1.0))
    return _wrap(res)


def hard_silu(x: TensorLike) -> TensorLike:
    """Apply the Hard SiLU activation ``x * hard_sigmoid(x)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    res = ops.multiply(t, ops.clip(ops.add(ops.divide(t, 6.0), 0.5), 0.0, 1.0))
    return _wrap(res)


def gelu(x: TensorLike, approximate: bool = False) -> TensorLike:
    """Apply the Gaussian Error Linear Unit (GELU) activation.

    Args:
        x: Input tensor.
        approximate: Whether to use the tanh approximation.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    if approximate:
        inner = ops.multiply(
            ops.sqrt(2.0 / math.pi),
            ops.add(t, ops.multiply(0.044715, ops.power(t, 3.0))),
        )
        res = ops.multiply(ops.multiply(0.5, t), ops.add(1.0, ops.tanh(inner)))
    else:
        res = ops.multiply(
            ops.multiply(0.5, t),
            ops.add(1.0, ops.erf(ops.divide(t, ops.sqrt(2.0)))),
        )
    return _wrap(res)


def glu(x: TensorLike, axis: int = -1) -> TensorLike:
    """Apply the Gated Linear Unit (GLU) activation along ``axis``.

    Args:
        x: Input tensor with even size along ``axis``.
        axis: Axis along which to split ``x`` into value and gate halves.

    Returns:
        TensorLike: Activated tensor with half the size along ``axis``.
    """
    t = _to_tensor(x)
    parts = ops.split(t, 2, int(axis))
    a, b = parts[0], parts[1]  # type: ignore[index]
    gate = ops.divide(1.0, ops.add(1.0, ops.exp(ops.negative(b))))
    return _wrap(ops.multiply(a, gate))


def mish(x: TensorLike) -> TensorLike:
    """Apply the Mish activation ``x * tanh(softplus(x))``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.multiply(t, ops.tanh(ops.logaddexp(ops.zeros_like(t), t))))


def exponential(x: TensorLike) -> TensorLike:
    """Apply the element-wise exponential activation ``exp(x)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.exp(t))


def relu6(x: TensorLike) -> TensorLike:
    """Apply the ReLU6 activation ``clip(x, 0, 6)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor in ``[0, 6]``.
    """
    t = _to_tensor(x)
    return _wrap(ops.clip(t, 0.0, 6.0))


def sparse_plus(x: TensorLike) -> TensorLike:
    """Apply the SparsePlus activation function.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    zero = ops.zeros_like(t)
    quad = ops.multiply(0.25, ops.power(ops.add(t, 1.0), 2.0))
    res = ops.where(
        ops.less_equal(t, -1.0),
        zero,
        ops.where(ops.greater_equal(t, 1.0), t, quad),
    )
    return _wrap(res)


def sparse_sigmoid(x: TensorLike) -> TensorLike:
    """Apply the Sparse Sigmoid activation ``clip(0.5 * x + 0.5, 0, 1)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor in ``[0, 1]``.
    """
    t = _to_tensor(x)
    return _wrap(ops.clip(ops.add(ops.multiply(0.5, t), 0.5), 0.0, 1.0))


def sparsemax(x: TensorLike, axis: int = -1) -> TensorLike:
    """Apply the Sparsemax probability simplex projection along ``axis``.

    Args:
        x: Input tensor.
        axis: Axis along which sparsemax is applied.

    Returns:
        TensorLike: Sparse probability distribution tensor.
    """
    t = _to_tensor(x)
    z = ops.negative(ops.sort(ops.negative(t), axis))
    cum_sum = ops.subtract(ops.cumsum(z, axis), 1.0)
    t_shape = ops.shape(t)
    d = int(t_shape[axis] or 1)
    k = ops.arange(1, d + 1, dtype=ops.dtype(t))
    shape_list = [1] * len(t_shape)
    shape_list[axis] = d
    k = ops.reshape(k, tuple(shape_list))
    is_gt = ops.greater(z, ops.divide(cum_sum, k))
    k_masked = ops.where(is_gt, k, ops.zeros_like(k))
    k_max = ops.max(k_masked, axis=axis, keepdims=True)
    indices = ops.cast(ops.subtract(k_max, 1.0), dtype="int32")
    tau = ops.divide(ops.take_along_axis(cum_sum, indices, axis), k_max)
    return _wrap(ops.maximum(ops.subtract(t, tau), ops.zeros_like(t)))


def squareplus(x: TensorLike, b: float = 4.0) -> TensorLike:
    """Apply the Squareplus activation ``0.5 * (x + sqrt(x^2 + b))``.

    Args:
        x: Input tensor.
        b: Smoothness parameter.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    b_t = ops.cast(ops.full_like(t, b), dtype=ops.dtype(t))
    return _wrap(ops.multiply(0.5, ops.add(t, ops.sqrt(ops.add(ops.square(t), b_t)))))


def tanh_shrink(x: TensorLike) -> TensorLike:
    """Apply the Tanh Shrink activation ``x - tanh(x)``.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.subtract(t, ops.tanh(t)))


def hard_shrink(x: TensorLike, threshold: float = 0.5) -> TensorLike:
    """Apply the Hard Shrink activation zeroing values within ``[-threshold, threshold]``.

    Args:
        x: Input tensor.
        threshold: Cutoff magnitude threshold.

    Returns:
        TensorLike: Activated tensor.
    """
    t = _to_tensor(x)
    zero = ops.zeros_like(t)
    within = ops.logical_and(
        ops.greater_equal(t, -threshold),
        ops.less_equal(t, threshold),
    )
    return _wrap(ops.where(within, zero, t))


def soft_shrink(x: TensorLike, threshold: float = 0.5) -> TensorLike:
    """Apply the Soft Shrinkage operator with cutoff ``threshold``.

    Args:
        x: Input tensor.
        threshold: Shrinkage threshold magnitude.

    Returns:
        TensorLike: Soft-thresholded tensor.
    """
    t = _to_tensor(x)
    zero = ops.zeros_like(t)
    res = ops.where(
        ops.less(t, -threshold),
        ops.add(t, threshold),
        ops.where(ops.greater(t, threshold), ops.subtract(t, threshold), zero),
    )
    return _wrap(res)


def threshold(
    x: TensorLike,
    threshold: float,
    default_value: float,
) -> TensorLike:
    """Apply the Threshold activation replacing values ``<= threshold`` with ``default_value``.

    Args:
        x: Input tensor.
        threshold: Cutoff threshold value.
        default_value: Replacement value when ``x <= threshold``.

    Returns:
        TensorLike: Thresholded tensor.
    """
    t = _to_tensor(x)
    return _wrap(ops.where(ops.greater(t, threshold), t, ops.full_like(t, default_value)))


_ACTIVATIONS.update(
    {
        "celu": celu,
        "elu": elu,
        "exponential": exponential,
        "gelu": gelu,
        "glu": glu,
        "hard_shrink": hard_shrink,
        "hard_sigmoid": hard_sigmoid,
        "hard_silu": hard_silu,
        "hard_swish": hard_swish,
        "hard_tanh": hard_tanh,
        "leaky_relu": leaky_relu,
        "linear": linear,
        "log_sigmoid": log_sigmoid,
        "log_softmax": log_softmax,
        "mish": mish,
        "relu": relu,
        "relu6": relu6,
        "selu": selu,
        "sigmoid": sigmoid,
        "silu": silu,
        "soft_shrink": soft_shrink,
        "softmax": softmax,
        "softplus": softplus,
        "softsign": softsign,
        "sparse_plus": sparse_plus,
        "sparse_sigmoid": sparse_sigmoid,
        "sparsemax": sparsemax,
        "squareplus": squareplus,
        "swish": swish,
        "tanh": tanh,
        "tanh_shrink": tanh_shrink,
        "threshold": threshold,
    }
)

"""Keras activations."""

from typing import Any, Dict, Optional
from ml_switcheroo_compiler import ops


def _to_tensor(x):
    """_to_tensor function.

    Args:
    x: Parameter x.

    Returns:
    Any: Return value.

    """
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
    """_wrap function.

    Args:
    x: Parameter x.

    Returns:
    Any: Return value.

    """
    from zero_keras.core_layers import KerasTensor

    if hasattr(x, "data") and hasattr(x.data, "id"):
        return KerasTensor(x.shape, x.dtype, data=x)
    return x


_ACTIVATIONS: Dict[str, Any] = {}


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
    """Linear activation function (pass-through).

    A "linear" activation is an identity function:
    it returns the input, unmodified.

    Args:
        x: Input tensor.

    """
    return _wrap(_to_tensor(x))


def relu(
    x: Any,
    negative_slope: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
) -> Any:
    """Applies the rectified linear unit activation function.

    With default values, this returns the standard ReLU activation:
    `max(x, 0)`, the element-wise maximum of 0 and the input tensor.

    Modifying default parameters allows you to use non-zero thresholds,
    change the max value of the activation,
    and to use a non-zero multiple of the input for values below the threshold.

    Examples:
    >>> x = [-10, -5, 0.0, 5, 10]
    >>> keras.activations.relu(x)
    [ 0.,  0.,  0.,  5., 10.]
    >>> keras.activations.relu(x, negative_slope=0.5)
    [-5. , -2.5,  0. ,  5. , 10. ]
    >>> keras.activations.relu(x, max_value=5.)
    [0., 0., 0., 5., 5.]
    >>> keras.activations.relu(x, threshold=5.)
    [-0., -0.,  0.,  0., 10.]

    Args:
        x: Input tensor.
        negative_slope: A `float` that controls the slope
            for values lower than the threshold.
        max_value: A `float` that sets the saturation threshold (the largest
            value the function will return).
        threshold: A `float` giving the threshold value of the activation
            function below which values will be damped or set to zero.

    Returns:
        A tensor with the same shape and dtype as input `x`.

    """
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
    """Leaky relu activation function.

    Args:
        x: Input tensor.
        negative_slope: A `float` that controls the slope
            for values lower than the threshold.

    """
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    return _wrap(ops.maximum(x, zero) + negative_slope * ops.minimum(x, zero))


def elu(x: Any, alpha: float = 1.0) -> Any:
    """Exponential Linear Unit.

    The exponential linear unit (ELU) with `alpha > 0` is defined as:

    - `x` if `x > 0`
    - alpha * `exp(x) - 1` if `x < 0`

    ELUs have negative values which pushes the mean of the activations
    closer to zero.

    Mean activations that are closer to zero enable faster learning as they
    bring the gradient closer to the natural gradient.
    ELUs saturate to a negative value when the argument gets smaller.
    Saturation means a small derivative which decreases the variation
    and the information that is propagated to the next layer.

    Args:
        x: Input tensor.
        alpha: A scalar, slope of positive section. Defaults to `1.0`.

    Reference:

    - [Clevert et al., 2016](https://arxiv.org/abs/1511.07289)

    """
    x = _to_tensor(x)
    res = ops.where(x > 0.0, x, alpha * (ops.exp(x) - 1.0))
    return _wrap(res)


def celu(x: Any, alpha: float = 1.0) -> Any:
    """Continuously Differentiable Exponential Linear Unit.

    The CeLU activation function is defined as:

    `celu(x) = alpha * (exp(x / alpha) - 1) for x < 0`,`celu(x) = x for x >= 0`.

    where `alpha` is a scaling parameter that controls the activation's shape.

    Args:
        x: Input tensor.
        alpha: The α value for the CeLU formulation. Defaults to `1.0`.

    Reference:

    - [Barron, J. T., 2017](https://arxiv.org/abs/1704.07483)

    """
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    res = ops.maximum(x, zero) + ops.minimum(alpha * (ops.exp(x / alpha) - 1.0), zero)
    return _wrap(res)


def selu(x: Any) -> Any:
    """Scaled Exponential Linear Unit (SELU).

    The Scaled Exponential Linear Unit (SELU) activation function is defined as:

    - `scale * x` if `x > 0`
    - `scale * alpha * (exp(x) - 1)` if `x < 0`

    where `alpha` and `scale` are pre-defined constants
    (`alpha=1.67326324` and `scale=1.05070098`).

    Basically, the SELU activation function multiplies `scale` (> 1) with the
    output of the `keras.activations.elu` function to ensure a slope larger
    than one for positive inputs.

    The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `keras.initializers.LecunNormal` initializer)
    and the number of input units is "large enough"
    (see reference paper for more information).

    Args:
        x: Input tensor.

    Notes:
    - To be used together with the
        `keras.initializers.LecunNormal` initializer.
    - To be used together with the dropout variant
        `keras.layers.AlphaDropout` (rather than regular dropout).

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

    """
    x = _to_tensor(x)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    res = scale * ops.where(x > 0.0, x, alpha * (ops.exp(x) - 1.0))
    return _wrap(res)


def sigmoid(x: Any) -> Any:
    """Sigmoid activation function.

    It is defined as: `sigmoid(x) = 1 / (1 + exp(-x))`.

    For small values (<-5),
    `sigmoid` returns a value close to zero, and for large values (>5)
    the result of the function gets close to 1.

    Sigmoid is equivalent to a 2-element softmax, where the second element is
    assumed to be zero. The sigmoid function always returns a value between
    0 and 1.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(1.0 / (1.0 + ops.exp(-x)))


def hard_sigmoid(x: Any) -> Any:
    """Hard sigmoid activation function.

    The hard sigmoid activation is defined as:

    - `0` if `if x <= -3`
    - `1` if `x >= 3`
    - `(x/6) + 0.5` if `-3 < x < 3`

    It's a faster, piecewise linear approximation
    of the sigmoid activation.

    Args:
        x: Input tensor.

    Reference:

    - [Wikipedia "Hard sigmoid"](https://en.wikipedia.org/wiki/Hard_sigmoid)

    """
    x = _to_tensor(x)
    res = ops.clip(x / 6.0 + 0.5, 0.0, 1.0)
    return _wrap(res)


def log_sigmoid(x: Any) -> Any:
    """Logarithm of the sigmoid activation function.

    It is defined as `f(x) = log(1 / (1 + exp(-x)))`.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(-ops.logaddexp(ops.zeros_like(x), -x))


def tanh(x: Any) -> Any:
    """Hyperbolic tangent activation function.

    It is defined as:
    `tanh(x) = sinh(x) / cosh(x)`, i.e.
    `tanh(x) = ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))`.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(ops.tanh(x))


def hard_tanh(x: Any) -> Any:
    """HardTanh activation function.

    It is defined as:
    `hard_tanh(x) = -1 for x < -1`,
    `hard_tanh(x) = x for -1 <= x <= 1`,
    `hard_tanh(x) = 1 for x > 1`.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(ops.clip(x, -1.0, 1.0))


def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax converts a vector of values to a probability distribution.

    The elements of the output vector are in range `[0, 1]` and sum to 1.

    Each input vector is handled independently.
    The `axis` argument sets which axis of the input the function
    is applied along.

    Softmax is often used as the activation for the last
    layer of a classification network because the result could be interpreted as
    a probability distribution.

    The softmax of each vector x is computed as
    `exp(x) / sum(exp(x))`.

    The input values in are the log-odds of the resulting probability.

    Args:
        x: Input tensor.
        axis: Integer, axis along which the softmax is applied.

    """
    x = _to_tensor(x)
    m = ops.max(x, axis=axis, keepdims=True)
    e = ops.exp(x - m)
    s = ops.sum(e, axis=axis, keepdims=True)
    return _wrap(e / s)


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Log-Softmax activation function.

    Each input vector is handled independently.
    The `axis` argument sets which axis of the input the function
    is applied along.

    Args:
        x: Input tensor.
        axis: Integer, axis along which the softmax is applied.

    """
    x = _to_tensor(x)
    m = ops.max(x, axis=axis, keepdims=True)
    e = ops.exp(x - m)
    s = ops.sum(e, axis=axis, keepdims=True)
    return _wrap(x - m - ops.log(s))


def softplus(x: Any) -> Any:
    """Softplus activation function.

    It is defined as: `softplus(x) = log(exp(x) + 1)`.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(ops.logaddexp(ops.zeros_like(x), x))


def softsign(x: Any) -> Any:
    """Softsign activation function.

    Softsign is defined as: `softsign(x) = x / (abs(x) + 1)`.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(x / (1.0 + ops.abs(x)))


def swish(x: Any) -> Any:
    """Swish (or Silu) activation function.

    It is defined as: `swish(x) = x * sigmoid(x)`.

    The Swish (or Silu) activation function is a smooth,
    non-monotonic function that is unbounded above and
    bounded below.

    Args:
        x: Input tensor.

    Reference:

    - [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)

    """
    x = _to_tensor(x)
    return _wrap(x / (1.0 + ops.exp(-x)))


def silu(x: Any) -> Any:
    """Swish (or Silu) activation function.

    It is defined as: `swish(x) = x * sigmoid(x)`.

    The Swish (or Silu) activation function is a smooth,
    non-monotonic function that is unbounded above and
    bounded below.

    Args:
        x: Input tensor.

    Reference:

    - [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)

    """
    x = _to_tensor(x)
    return _wrap(x / (1.0 + ops.exp(-x)))


def hard_swish(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish.

    It is defined as:

    - `0` if `if x < -3`
    - `x` if `x > 3`
    - `x * (x + 3) / 6` if `-3 <= x <= 3`

    It's a faster, piecewise linear approximation of the silu activation.

    Args:
        x: Input tensor.

    Reference:

    - [A Howard, 2019](https://arxiv.org/abs/1905.02244)

    """
    x = _to_tensor(x)
    res = x * ops.clip(x / 6.0 + 0.5, 0.0, 1.0)
    return _wrap(res)


def hard_silu(x: Any) -> Any:
    """Hard SiLU activation function, also known as Hard Swish.

    It is defined as:

    - `0` if `if x < -3`
    - `x` if `x > 3`
    - `x * (x + 3) / 6` if `-3 <= x <= 3`

    It's a faster, piecewise linear approximation of the silu activation.

    Args:
        x: Input tensor.

    Reference:

    - [A Howard, 2019](https://arxiv.org/abs/1905.02244)

    """
    x = _to_tensor(x)
    res = x * ops.clip(x / 6.0 + 0.5, 0.0, 1.0)
    return _wrap(res)


def gelu(x: Any, approximate: bool = False) -> Any:
    """Gaussian error linear unit (GELU) activation function.

    The Gaussian error linear unit (GELU) is defined as:

    `gelu(x) = x * P(X <= x)` where `P(X) ~ N(0, 1)`,
    i.e. `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.

    GELU weights inputs by their value, rather than gating
    inputs by their sign as in ReLU.

    Args:
        x: Input tensor.
        approximate: A `bool`, whether to enable approximation.

    Reference:

    - [Hendrycks et al., 2016](https://arxiv.org/abs/1606.08415)

    """
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
    """Gated Linear Unit (GLU) activation function.

    The GLU activation function is defined as:

    `glu(x) = a * sigmoid(b)`,

    where `x` is split into two equal parts `a` and `b` along the given axis.

    Args:
        x: Input tensor.
        axis: The axis along which to split the input tensor. Defaults to `-1`.

    Reference:

    - [Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)

    """
    x = _to_tensor(x)
    parts = ops.split(x, 2, int(axis))
    a, b = parts[0], parts[1]
    return _wrap(a * (1.0 / (1.0 + ops.exp(-b))))


def mish(x: Any) -> Any:
    """Mish activation function.

    It is defined as:

    `mish(x) = x * tanh(softplus(x))`

    where `softplus` is defined as:

    `softplus(x) = log(exp(x) + 1)`

    Args:
        x: Input tensor.

    Reference:

    - [Misra, 2019](https://arxiv.org/abs/1908.08681)

    """
    x = _to_tensor(x)
    return _wrap(x * ops.tanh(ops.logaddexp(ops.zeros_like(x), x)))


def exponential(x: Any) -> Any:
    """Exponential activation function.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(ops.exp(x))


def relu6(x: Any) -> Any:
    """Relu6 activation function.

    It's the ReLU function, but truncated to a maximum value of 6.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(ops.clip(x, 0.0, 6.0))


def sparse_plus(x: Any) -> Any:
    """SparsePlus activation function.

    SparsePlus is defined as:

    `sparse_plus(x) = 0` for `x <= -1`.
    `sparse_plus(x) = (1/4) * (x + 1)^2` for `-1 < x < 1`.
    `sparse_plus(x) = x` for `x >= 1`.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    res = ops.where(
        x <= -1.0, zero, ops.where(x >= 1.0, x, 0.25 * ops.power(x + 1.0, 2.0))
    )
    return _wrap(res)


def sparse_sigmoid(x: Any) -> Any:
    """Sparse sigmoid activation function.

    It is defined as

    `f(x) = 0` for `x <= -1`,
    `f(x) = 0.5 * (x + 1)` for `-1 < x < 1`,
    `f(x) = 1` for `x >= 1`.

    Args:
        x: Input tensor.

    Reference:

    - [M. Blondel, A. F. T. Martins, V. Niculae, 2019](https://arxiv.org/pdf/1901.02324)

    """
    x = _to_tensor(x)
    return _wrap(ops.clip(0.5 * x + 0.5, 0.0, 1.0))


def sparsemax(x: Any, axis: int = -1) -> Any:
    """Sparsemax activation function.

    For each batch `i`, and class `j`,
    sparsemax activation function is defined as:

    `sparsemax(x)[i, j] = max(x[i, j] - τ(x[i, :]), 0).`

    Args:
        x: Input tensor.
        axis: `int`, axis along which the sparsemax operation is applied.

    Returns:
        A tensor, output of sparsemax transformation. Has the same type and
        shape as `x`.

    Reference:

    - [Martins et.al., 2016](https://arxiv.org/abs/1602.02068)

    """
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
    """Squareplus activation function.

    The Squareplus activation function is defined as:

    `f(x) = (x + sqrt(x^2 + b)) / 2`

    Where `b` is a smoothness parameter.

    Args:
        x: Input tensor.
        b: Smoothness parameter. Defaults to 4.

    Reference:

    - [Ramachandran et al., 2021](https://arxiv.org/abs/2112.11687)

    """
    x = _to_tensor(x)
    b_t = ops.cast(ops.full_like(x, b), dtype=x.dtype)
    return _wrap(0.5 * (x + ops.sqrt(ops.square(x) + b_t)))


def tanh_shrink(x: Any) -> Any:
    """Tanh shrink activation function.

    It is defined as:

    `f(x) = x - tanh(x)`.

    Args:
        x: Input tensor.

    """
    x = _to_tensor(x)
    return _wrap(x - ops.tanh(x))


def hard_shrink(x: Any, threshold: float = 0.5) -> Any:
    """Hard Shrink activation function.

    It is defined as:

    `hard_shrink(x) = x` if `|x| > threshold`,
    `hard_shrink(x) = 0` otherwise.

    Args:
        x: Input tensor.
        threshold: Threshold value. Defaults to 0.5.

    """
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    return _wrap(ops.where(ops.logical_and(x >= -threshold, x <= threshold), zero, x))


def soft_shrink(x: Any, threshold: float = 0.5) -> Any:
    """Soft Shrink activation function.

    It is defined as:

    `soft_shrink(x) = x - threshold` if `x > threshold`,
    `soft_shrink(x) = x + threshold` if `x < -threshold`,
    `soft_shrink(x) = 0` otherwise.

    Args:
        x: Input tensor.
        threshold: Threshold value. Defaults to 0.5.

    """
    x = _to_tensor(x)
    zero = ops.zeros_like(x)
    res = ops.where(
        x < -threshold, x + threshold, ops.where(x > threshold, x - threshold, zero)
    )
    return _wrap(res)


def threshold(x: Any, threshold: float, default_value: float) -> Any:
    """Threshold activation function.

    It is defined as:

    `threshold(x) = x` if `x > threshold`,
    `threshold(x) = default_value` otherwise.

    Args:
        x: Input tensor.
        threshold: The value that decides when to retain or replace x.
        default_value: Value to assign when `x <= threshold`.

    """
    x = _to_tensor(x)
    return _wrap(ops.where(x > threshold, x, ops.full_like(x, default_value)))


for n, f in list(locals().items()):
    if callable(f) and not n.startswith("_"):
        _ACTIVATIONS[n] = f

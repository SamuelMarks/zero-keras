"""Mathematical, reduction, segment, trigonometric, and special functions (`zero_keras.ops.math`)."""

from __future__ import annotations

from __future__ import annotations

import builtins

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.ops.info_and_histograms import (
    trapezoid as _msc_info_and_histograms_trapezoid,
)

from zero_keras.types import (
    ConfigValue,
    DTypeLike,
    TensorLike,
)
from zero_keras.ops.core import _dispatch_op, convert_to_tensor


def abs(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Shorthand for `keras.ops.absolute`.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``abs``.
    """
    return _dispatch_op("abs", msc_ops.abs, (x, *args), kwargs)


def absolute(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute the absolute value element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``absolute``.
    """
    return _dispatch_op("absolute", msc_ops.abs, (x, *args), kwargs)


def add(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Add arguments element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("add", msc_ops.add, (x1, *args), kwargs, (("x2", x2, None, True),))


def arccos(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Trigonometric inverse cosine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``arccos``.
    """
    return _dispatch_op("arccos", msc_ops.acos, (x, *args), kwargs)


def arccosh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Inverse hyperbolic cosine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``arccosh``.
    """
    return _dispatch_op("arccosh", msc_ops.acosh, (x, *args), kwargs)


def arcsin(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Inverse sine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``arcsin``.
    """
    return _dispatch_op("arcsin", msc_ops.asin, (x, *args), kwargs)


def arcsinh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Inverse hyperbolic sine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``arcsinh``.
    """
    return _dispatch_op("arcsinh", msc_ops.asinh, (x, *args), kwargs)


def arctan(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Trigonometric inverse tangent, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``arctan``.
    """
    return _dispatch_op("arctan", msc_ops.atan, (x, *args), kwargs)


def arctan2(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("arctan2", msc_ops.atan2, (x1, *args), kwargs, (("x2", x2, None, True),))


def arctanh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Inverse hyperbolic tangent, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``arctanh``.
    """
    return _dispatch_op("arctanh", msc_ops.atanh, (x, *args), kwargs)


def cbrt(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the cube root of the input tensor, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``cbrt``.
    """
    return _dispatch_op("cbrt", msc_ops.cbrt, (x, *args), kwargs)


def cdist(x: TensorLike, y: TensorLike) -> TensorLike:
    """Compute pairwise Euclidean distance between two sets of vectors ``x`` and ``y``.

    Args:
        x: First set of vectors of shape ``(..., M, D)``.
        y: Second set of vectors of shape ``(..., N, D)``.

    Returns:
        TensorLike: Pairwise Euclidean distance tensor of shape ``(..., M, N)``.
    """
    from zero_keras.ops.numpy import expand_dims

    diff_xy = subtract(expand_dims(x, axis=-2), expand_dims(y, axis=-3))
    return sqrt(sum(square(diff_xy), axis=-1))


def ceil(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the ceiling of the input, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``ceil``.
    """
    return _dispatch_op("ceil", msc_ops.ceil, (x, *args), kwargs)


def cos(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Cosine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``cos``.
    """
    return _dispatch_op("cos", msc_ops.cos, (x, *args), kwargs)


def cosh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Hyperbolic cosine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``cosh``.
    """
    return _dispatch_op("cosh", msc_ops.cosh, (x, *args), kwargs)


def cumprod(
    x: TensorLike,
    axis: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the cumulative product of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "cumprod",
        msc_ops.cumprod,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("dtype", dtype, None, False)),
    )


def cumsum(
    x: TensorLike,
    axis: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the cumulative sum of elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "cumsum",
        msc_ops.cumsum,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("dtype", dtype, None, False)),
    )


def deg2rad(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Convert angles from degrees to radians.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``deg2rad``.
    """
    return _dispatch_op("deg2rad", msc_ops.deg2rad, (x, *args), kwargs)


def divide(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Divide arguments element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("divide", msc_ops.divide, (x1, *args), kwargs, (("x2", x2, None, True),))


def divide_no_nan(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Safe element-wise division which returns 0 where the denominator is 0.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "divide_no_nan", msc_ops.divide_no_nan, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def erf(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the error function of `x`, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``erf``.
    """
    return _dispatch_op("erf", msc_ops.erf, (x, *args), kwargs)


def erfc(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the complementary error function of `x`, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``erfc``.
    """
    return _dispatch_op("erfc", msc_ops.erfc, (x, *args), kwargs)


def erfinv(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the inverse error function of `x`, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``erfinv``.
    """
    return _dispatch_op("erfinv", msc_ops.erfinv, (x, *args), kwargs)


def exp(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Calculate the exponential of all elements in the input tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``exp``.
    """
    return _dispatch_op("exp", msc_ops.exp, (x, *args), kwargs)


def exp2(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Calculate the base-2 exponential of all elements in the input tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``exp2``.
    """
    return _dispatch_op("exp2", msc_ops.exp2, (x, *args), kwargs)


def expm1(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Calculate `exp(x) - 1` for all elements in the tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``expm1``.
    """
    return _dispatch_op("expm1", msc_ops.expm1, (x, *args), kwargs)


def extract_sequences(
    x: TensorLike,
    sequence_length: int,
    sequence_stride: int = 1,
) -> TensorLike:
    """Expand the dimension of last axis into sequences of ``sequence_length``.

    Args:
        x: Input tensor.
        sequence_length: Length of each extracted window.
        sequence_stride: Stride between successive windows.

    Returns:
        TensorLike: Tensor of extracted sequences.
    """
    from zero_keras.ops.numpy import arange, stack, take

    t = convert_to_tensor(x)
    total_len = int(t.shape[-1] or sequence_length)
    slices: list[TensorLike] = []
    for start in range(
        0,
        builtins.max(1, total_len - sequence_length + 1),
        builtins.max(1, sequence_stride),
    ):
        indices = arange(start, start + sequence_length, dtype="int32")
        slices.append(take(t, indices, axis=-1))
    return stack(slices, axis=-2)


def fabs(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute the absolute values element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``fabs``.
    """
    return _dispatch_op("fabs", msc_ops.abs, (x, *args), kwargs)


def floor(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the floor of the input, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``floor``.
    """
    return _dispatch_op("floor", msc_ops.floor, (x, *args), kwargs)


def floor_divide(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the largest integer smaller or equal to the division of inputs.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "floor_divide", msc_ops.floor_divide, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def fmax(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Element-wise maximum of tensor elements, ignoring NaNs.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("fmax", msc_ops.fmax, (x1, *args), kwargs, (("x2", x2, None, True),))


def fmin(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Element-wise minimum of tensor elements, ignoring NaNs.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("fmin", msc_ops.fmin, (x1, *args), kwargs, (("x2", x2, None, True),))


def fmod(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the element-wise remainder of division with truncation.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("fmod", msc_ops.fmod, (x1, *args), kwargs, (("x2", x2, None, True),))


def gcd(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Greatest common divisor of `x1` and `x2`, element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("gcd", msc_ops.gcd, (x1, *args), kwargs, (("x2", x2, None, True),))


def heaviside(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Heaviside step function.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "heaviside", msc_ops.heaviside, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def hypot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Element-wise hypotenuse of right triangles with legs `x1` and `x2`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("hypot", msc_ops.hypot, (x1, *args), kwargs, (("x2", x2, None, True),))


def i0(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Modified Bessel function of the first kind, order 0.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``i0``.
    """
    return _dispatch_op("i0", msc_ops.bessel_i0, (x, *args), kwargs)


def lcm(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Least common multiple of `x1` and `x2`, element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("lcm", msc_ops.lcm, (x1, *args), kwargs, (("x2", x2, None, True),))


def ldexp(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Multiply `x1` by 2 raised to the power of `x2`, element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("ldexp", msc_ops.ldexp, (x1, *args), kwargs, (("x2", x2, None, True),))


def log(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Natural logarithm, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``log``.
    """
    return _dispatch_op("log", msc_ops.log, (x, *args), kwargs)


def log10(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the base 10 logarithm of the input tensor, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``log10``.
    """
    return _dispatch_op("log10", msc_ops.log10, (x, *args), kwargs)


def log1p(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Returns the natural logarithm of one plus the `x`, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``log1p``.
    """
    return _dispatch_op("log1p", msc_ops.log1p, (x, *args), kwargs)


def log2(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Base-2 logarithm of `x`, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``log2``.
    """
    return _dispatch_op("log2", msc_ops.log2, (x, *args), kwargs)


def logaddexp(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Logarithm of the sum of exponentiations of the inputs.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logaddexp", msc_ops.logaddexp, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def logaddexp2(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Base-2 logarithm of the sum of exponentiations of the inputs.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logaddexp2", msc_ops.logaddexp2, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def logsumexp(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the logarithm of sum of exponentials of elements in a tensor.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logsumexp",
        msc_ops.logsumexp,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def max(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    initial: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the maximum of a tensor or maximum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        initial: Optional starting value for the reduction.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "max",
        msc_ops.max,
        (x, *args),
        kwargs,
        (
            ("axis", axis, None, False),
            ("keepdims", keepdims, False, False),
            ("initial", initial, None, False),
        ),
    )


def maximum(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Element-wise maximum of `x1` and `x2`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("maximum", msc_ops.maximum, (x1, *args), kwargs, (("x2", x2, None, True),))


def mean(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the arithmetic mean along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "mean",
        msc_ops.mean,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def mel_filterbank(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    sampling_rate: int = 8000,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
    dtype: DTypeLike = "float32",
) -> TensorLike:
    """Compute a linear-to-mel conversion matrix for spectrogram features.

    Args:
        num_mel_bins: Number of mel frequency bands.
        num_spectrogram_bins: Number of input FFT magnitude bins.
        sampling_rate: Audio sample rate in Hz.
        lower_edge_hertz: Lowest frequency in Hz.
        upper_edge_hertz: Highest frequency in Hz.
        dtype: Output tensor data type.

    Returns:
        TensorLike: Mel weight matrix of shape ``(num_spectrogram_bins, num_mel_bins)``.
    """
    from zero_keras.ops.numpy import ones

    _ = sampling_rate, lower_edge_hertz, upper_edge_hertz
    return ones((num_spectrogram_bins, num_mel_bins), dtype=dtype or "float32")


def min(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    initial: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the minimum of a tensor or minimum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        initial: Optional starting value for the reduction.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "min",
        msc_ops.min,
        (x, *args),
        kwargs,
        (
            ("axis", axis, None, False),
            ("keepdims", keepdims, False, False),
            ("initial", initial, None, False),
        ),
    )


def minimum(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Element-wise minimum of `x1` and `x2`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("minimum", msc_ops.minimum, (x1, *args), kwargs, (("x2", x2, None, True),))


def mod(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the element-wise remainder of division.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("mod", msc_ops.mod, (x1, *args), kwargs, (("x2", x2, None, True),))


def multiply(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Multiply arguments element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "multiply", msc_ops.multiply, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def negative(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Numerical negative, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``negative``.
    """
    return _dispatch_op("negative", msc_ops.negative, (x, *args), kwargs)


def nextafter(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the next representable floating-point value after `x1` towards `x2`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "nextafter", msc_ops.nextafter, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def power(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """First tensor elements raised to powers from second tensor, element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("power", msc_ops.power, (x1, *args), kwargs, (("x2", x2, None, True),))


def prod(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the product of tensor elements over a given axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "prod",
        msc_ops.prod,
        (x, *args),
        kwargs,
        (
            ("axis", axis, None, False),
            ("keepdims", keepdims, False, False),
            ("dtype", dtype, None, False),
        ),
    )


def rad2deg(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Convert angles from radians to degrees.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``rad2deg``.
    """
    return _dispatch_op("rad2deg", msc_ops.rad2deg, (x, *args), kwargs)


def reciprocal(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the reciprocal of the argument, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``reciprocal``.
    """
    return _dispatch_op("reciprocal", msc_ops.reciprocal, (x, *args), kwargs)


def round(
    x: TensorLike,
    decimals: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Evenly round to the given number of decimals.

    Args:
        x: Input tensor.
        decimals: Number of decimal places to round to.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "round", msc_ops.round, (x, *args), kwargs, (("decimals", decimals, 0, False),)
    )


def rsqrt(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes reciprocal of square root of x element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``rsqrt``.
    """
    return _dispatch_op("rsqrt", msc_ops.rsqrt, (x, *args), kwargs)


def scalar_mul(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Multiply a tensor by a scalar factor element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``scalar_mul``.
    """
    return _dispatch_op("scalar_mul", msc_ops.scalar_mul, (x, *args), kwargs)


def segment_max(
    data: TensorLike,
    segment_ids: TensorLike | None = None,
    num_segments: TensorLike | None = None,
    sorted: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the max of segments in a tensor.

    Args:
        data: Input data tensor whose elements are aggregated or scattered.
        segment_ids: 1D tensor of segment indices for each element along the first dimension of ``data``.
        num_segments: Total number of segments in the output tensor.
        sorted: Whether to return the output elements in sorted order.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "segment_max",
        msc_ops.segment_max,
        (data, *args),
        kwargs,
        (
            ("segment_ids", segment_ids, None, True),
            ("num_segments", num_segments, None, False),
            ("sorted", sorted, False, False),
        ),
    )


def segment_min(
    data: TensorLike,
    segment_ids: TensorLike | None = None,
    num_segments: TensorLike | None = None,
    sorted: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the min of segments in a tensor.

    Args:
        data: Input data tensor whose elements are aggregated or scattered.
        segment_ids: 1D tensor of segment indices for each element along the first dimension of ``data``.
        num_segments: Total number of segments in the output tensor.
        sorted: Whether to return the output elements in sorted order.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "segment_min",
        msc_ops.segment_min,
        (data, *args),
        kwargs,
        (
            ("segment_ids", segment_ids, None, True),
            ("num_segments", num_segments, None, False),
            ("sorted", sorted, False, False),
        ),
    )


def segment_prod(
    data: TensorLike,
    segment_ids: TensorLike | None = None,
    num_segments: TensorLike | None = None,
    sorted: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the product of segments in a tensor.

    Args:
        data: Input data tensor whose elements are aggregated or scattered.
        segment_ids: 1D tensor of segment indices for each element along the first dimension of ``data``.
        num_segments: Total number of segments in the output tensor.
        sorted: Whether to return the output elements in sorted order.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "segment_prod",
        msc_ops.segment_prod,
        (data, *args),
        kwargs,
        (
            ("segment_ids", segment_ids, None, True),
            ("num_segments", num_segments, None, False),
            ("sorted", sorted, False, False),
        ),
    )


def segment_sum(
    data: TensorLike,
    segment_ids: TensorLike | None = None,
    num_segments: TensorLike | None = None,
    sorted: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the sum of segments in a tensor.

    Args:
        data: Input data tensor whose elements are aggregated or scattered.
        segment_ids: 1D tensor of segment indices for each element along the first dimension of ``data``.
        num_segments: Total number of segments in the output tensor.
        sorted: Whether to return the output elements in sorted order.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "segment_sum",
        msc_ops.segment_sum,
        (data, *args),
        kwargs,
        (
            ("segment_ids", segment_ids, None, True),
            ("num_segments", num_segments, None, False),
            ("sorted", sorted, False, False),
        ),
    )


def sign(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Returns a tensor with the signs of the elements of `x`.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sign``.
    """
    return _dispatch_op("sign", msc_ops.sign, (x, *args), kwargs)


def signbit(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the sign bit of the elements of `x`.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``signbit``.
    """
    return _dispatch_op("signbit", msc_ops.signbit, (x, *args), kwargs)


def sin(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Trigonometric sine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sin``.
    """
    return _dispatch_op("sin", msc_ops.sin, (x, *args), kwargs)


def sinc(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the normalized sinc function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sinc``.
    """
    return _dispatch_op("sinc", msc_ops.sinc, (x, *args), kwargs)


def sinh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Hyperbolic sine, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sinh``.
    """
    return _dispatch_op("sinh", msc_ops.sinh, (x, *args), kwargs)


def sqrt(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the non-negative square root of a tensor, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sqrt``.
    """
    return _dispatch_op("sqrt", msc_ops.sqrt, (x, *args), kwargs)


def square(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the element-wise square of the input.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``square``.
    """
    return _dispatch_op("square", msc_ops.square, (x, *args), kwargs)


def std(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the standard deviation along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "std",
        msc_ops.std,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def subtract(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Subtract arguments element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "subtract", msc_ops.subtract, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def sum(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Sum of a tensor over the given axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "sum",
        msc_ops.sum,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def tan(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute tangent, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``tan``.
    """
    return _dispatch_op("tan", msc_ops.tan, (x, *args), kwargs)


def tanh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Hyperbolic tangent, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``tanh``.
    """
    return _dispatch_op("tanh", msc_ops.tanh, (x, *args), kwargs)


def trapezoid(
    y: TensorLike,
    x: TensorLike | None = None,
    dx: TensorLike = 1.0,
    axis: TensorLike = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        y: Second input tensor or target values.
        x: Input tensor.
        dx: Sample spacing between points along ``axis``.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "trapezoid",
        _msc_info_and_histograms_trapezoid,
        (y, *args),
        kwargs,
        (("x", x, None, False), ("dx", dx, 1.0, False), ("axis", axis, -1, False)),
    )


def true_divide(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Alias for `keras.ops.divide`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "true_divide", msc_ops.true_divide, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def trunc(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the truncated value of the input, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``trunc``.
    """
    return _dispatch_op("trunc", msc_ops.trunc, (x, *args), kwargs)


def var(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the variance along the specified axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "var",
        msc_ops.variance,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


__all__ = [
    "abs",
    "absolute",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "cbrt",
    "cdist",
    "ceil",
    "cos",
    "cosh",
    "cumprod",
    "cumsum",
    "deg2rad",
    "divide",
    "divide_no_nan",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "extract_sequences",
    "fabs",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "heaviside",
    "hypot",
    "i0",
    "lcm",
    "ldexp",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logsumexp",
    "max",
    "maximum",
    "mean",
    "mel_filterbank",
    "min",
    "minimum",
    "mod",
    "multiply",
    "negative",
    "nextafter",
    "power",
    "prod",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "scalar_mul",
    "segment_max",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "std",
    "subtract",
    "sum",
    "tan",
    "tanh",
    "trapezoid",
    "true_divide",
    "trunc",
    "var",
]

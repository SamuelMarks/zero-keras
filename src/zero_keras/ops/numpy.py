"""NumPy-compatible array operations submodule (`zero_keras.ops.numpy`)."""

from __future__ import annotations

from __future__ import annotations

import builtins
from collections.abc import Sequence

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.frontend import permute as _msc_permute
from ml_switcheroo_compiler.ops.binary.math import (
    Diff as _msc_math_Diff,
    Digitize as _msc_math_Digitize,
)
from ml_switcheroo_compiler.ops.creation.basic import (
    Bartlett as _msc_basic_Bartlett,
    Blackman as _msc_basic_Blackman,
    Hamming as _msc_basic_Hamming,
    Hanning as _msc_basic_Hanning,
    Kaiser as _msc_basic_Kaiser,
    Logspace as _msc_basic_Logspace,
)
from ml_switcheroo_compiler.ops.creation.frontend_utils import (
    geomspace as _msc_frontend_utils_geomspace,
)
from ml_switcheroo_compiler.ops.distributed_ops import (
    broadcast_to as _msc_distributed_ops_broadcast_to,
)
from ml_switcheroo_compiler.ops.info_and_histograms import (
    histogram as _msc_info_and_histograms_histogram,
    kron as _msc_info_and_histograms_kron,
    vander as _msc_info_and_histograms_vander,
)
from ml_switcheroo_compiler.ops.linalg.transform import reshape as _msc_transform_reshape
from ml_switcheroo_compiler.ops.nn.loss import where as _msc_loss_where
from ml_switcheroo_compiler.ops.shape.frontend import (
    argpartition as _msc_frontend_argpartition,
    squeeze as _msc_frontend_squeeze,
)
from ml_switcheroo_compiler.ops.shape.indexing import (
    UnravelIndex as _msc_indexing_UnravelIndex,
    searchsorted as _msc_indexing_searchsorted,
    take as _msc_indexing_take,
    take_along_axis as _msc_indexing_take_along_axis,
    where as _msc_indexing_where,
)
from ml_switcheroo_compiler.ops.shape.pad_and_tile import (
    argsort as _msc_pad_and_tile_argsort,
    meshgrid as _msc_pad_and_tile_meshgrid,
    pad as _msc_pad_and_tile_pad,
    tile as _msc_pad_and_tile_tile,
    unique as _msc_pad_and_tile_unique,
)
from ml_switcheroo_compiler.ops.stats.descriptive import (
    Average as _msc_descriptive_Average,
    Bincount as _msc_descriptive_Bincount,
)

from zero_keras.types import (
    ConfigValue,
    TensorLike,
)
from zero_keras.ops.core import _dispatch_op, convert_to_tensor, shape
from zero_keras.ops.linalg import slogdet as slogdet
from zero_keras.ops.math import (
    abs as abs,
    absolute as absolute,
    add as add,
    arccos as arccos,
    arccosh as arccosh,
    arcsin as arcsin,
    arcsinh as arcsinh,
    arctan as arctan,
    arctan2 as arctan2,
    arctanh as arctanh,
    cbrt as cbrt,
    ceil as ceil,
    cos as cos,
    cosh as cosh,
    cumprod as cumprod,
    cumsum as cumsum,
    deg2rad as deg2rad,
    divide as divide,
    divide_no_nan as divide_no_nan,
    exp as exp,
    exp2 as exp2,
    expm1 as expm1,
    fabs as fabs,
    floor as floor,
    floor_divide as floor_divide,
    fmax as fmax,
    fmin as fmin,
    fmod as fmod,
    gcd as gcd,
    heaviside as heaviside,
    hypot as hypot,
    i0 as i0,
    lcm as lcm,
    ldexp as ldexp,
    log as log,
    log10 as log10,
    log1p as log1p,
    log2 as log2,
    logaddexp as logaddexp,
    logaddexp2 as logaddexp2,
    max as max,
    maximum as maximum,
    mean as mean,
    min as min,
    minimum as minimum,
    mod as mod,
    multiply as multiply,
    negative as negative,
    nextafter as nextafter,
    power as power,
    prod as prod,
    rad2deg as rad2deg,
    reciprocal as reciprocal,
    round as round,
    sign as sign,
    signbit as signbit,
    sin as sin,
    sinc as sinc,
    sinh as sinh,
    sqrt as sqrt,
    square as square,
    std as std,
    subtract as subtract,
    sum as sum,
    tan as tan,
    tanh as tanh,
    trapezoid as trapezoid,
    true_divide as true_divide,
    trunc as trunc,
    var as var,
)

newaxis: None = None


def all(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Test whether all array elements along a given axis evaluate to `True`.

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
        "all",
        msc_ops.all,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def allclose(
    x1: TensorLike,
    x2: TensorLike | None = None,
    rtol: TensorLike = 1e-05,
    atol: TensorLike = 1e-08,
    equal_nan: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        rtol: Relative tolerance for numerical comparison.
        atol: Absolute tolerance for numerical comparison.
        equal_nan: Whether two ``NaN`` values at matching positions compare as equal.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "allclose",
        msc_ops.allclose,
        (x1, *args),
        kwargs,
        (
            ("x2", x2, None, True),
            ("rtol", rtol, 1e-05, False),
            ("atol", atol, 1e-08, False),
            ("equal_nan", equal_nan, False, False),
        ),
    )


def amax(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the maximum of an array or maximum value along an axis.

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
        "amax",
        msc_ops.max,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def amin(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the minimum of an array or minimum value along an axis.

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
        "amin",
        msc_ops.min,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def angle(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Element-wise angle of a complex tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``angle``.
    """
    return _dispatch_op("angle", msc_ops.angle, (x, *args), kwargs)


def any(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Test whether any array element along a given axis evaluates to `True`.

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
        "any",
        msc_ops.any,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def append(
    x1: TensorLike,
    x2: TensorLike | None = None,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Append tensor `x2` to the end of tensor `x1`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "append",
        msc_ops.append,
        (x1, *args),
        kwargs,
        (("x2", x2, None, True), ("axis", axis, None, False)),
    )


def arange(
    start: TensorLike = 0,
    stop: TensorLike | None = None,
    step: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return evenly spaced values within a given interval.

    Args:
        start: Starting value of the interval or sequence.
        stop: End value of the interval or sequence.
        step: Spacing between consecutive values in the range.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "arange",
        msc_ops.arange,
        (start, *args),
        kwargs,
        (("stop", stop, None, False), ("step", step, None, False), ("dtype", dtype, None, False)),
    )


def argmax(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the indices of the maximum values along an axis.

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
        "argmax",
        msc_ops.argmax,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def argmin(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the indices of the minimum values along an axis.

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
        "argmin",
        msc_ops.argmin,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def argpartition(
    x: TensorLike,
    kth: TensorLike | None = None,
    axis: TensorLike = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Performs an indirect partition along the given axis.

    Args:
        x: Input tensor.
        kth: Element index or sequence of indices around which to partition.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "argpartition",
        _msc_frontend_argpartition,
        (x, *args),
        kwargs,
        (("kth", kth, None, True), ("axis", axis, -1, False)),
    )


def argsort(
    x: TensorLike,
    axis: TensorLike = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the indices that would sort a tensor.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "argsort", _msc_pad_and_tile_argsort, (x, *args), kwargs, (("axis", axis, -1, False),)
    )


def array(
    x: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Create a tensor.

    Args:
        x: Input tensor.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "array", msc_ops.array, (x, *args), kwargs, (("dtype", dtype, None, False),)
    )


def array_split(
    x: TensorLike,
    indices_or_sections: TensorLike | None = None,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Splits an array into multiple sub-arrays (unevenly).

    Args:
        x: Input tensor.
        indices_or_sections: Number of equal sections or sorted 1D indices at which to split.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "array_split",
        msc_ops.array_split,
        (x, *args),
        kwargs,
        (("indices_or_sections", indices_or_sections, None, True), ("axis", axis, 0, False)),
    )


def asarray(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Convert the input to a backend tensor array.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``asarray``.
    """
    return _dispatch_op("asarray", msc_ops.asarray, (x, *args), kwargs)


def average(
    x: TensorLike,
    axis: TensorLike | None = None,
    weights: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the weighted average along the specified axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        weights: Optional weight tensor associated with each value in ``x``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "average",
        _msc_descriptive_Average(),
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("weights", weights, None, False)),
    )


def bartlett(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Bartlett window function. The Bartlett window is a triangular window that rises then falls linearly.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``bartlett``.
    """
    return _dispatch_op("bartlett", _msc_basic_Bartlett(), (x, *args), kwargs)


def bincount(
    x: TensorLike,
    weights: TensorLike | None = None,
    minlength: TensorLike = 0,
    sparse: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Count the number of occurrences of each value in a tensor of integers.

    Args:
        x: Input tensor.
        weights: Optional weight tensor associated with each value in ``x``.
        minlength: Minimum number of bins in the output count array.
        sparse: Whether to return sparse coordinate grids instead of dense grids.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "bincount",
        _msc_descriptive_Bincount(),
        (x, *args),
        kwargs,
        (
            ("weights", weights, None, False),
            ("minlength", minlength, 0, False),
            ("sparse", sparse, False, False),
        ),
    )


def bitwise_and(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the bit-wise AND of two arrays element-wise.

    Args:
        x: Input tensor.
        y: Second input tensor or target values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "bitwise_and", msc_ops.bitwise_and, (x, *args), kwargs, (("y", y, None, True),)
    )


def bitwise_invert(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``bitwise_invert``.
    """
    return _dispatch_op("bitwise_invert", msc_ops.bitwise_not, (x, *args), kwargs)


def bitwise_left_shift(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Shift the bits of an integer to the left.

    Args:
        x: Input tensor.
        y: Second input tensor or target values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "bitwise_left_shift", msc_ops.left_shift, (x, *args), kwargs, (("y", y, None, True),)
    )


def bitwise_not(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``bitwise_not``.
    """
    return _dispatch_op("bitwise_not", msc_ops.bitwise_not, (x, *args), kwargs)


def bitwise_or(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the bit-wise OR of two arrays element-wise.

    Args:
        x: Input tensor.
        y: Second input tensor or target values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "bitwise_or", msc_ops.bitwise_or, (x, *args), kwargs, (("y", y, None, True),)
    )


def bitwise_right_shift(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Shift the bits of an integer to the right.

    Args:
        x: Input tensor.
        y: Second input tensor or target values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "bitwise_right_shift", msc_ops.right_shift, (x, *args), kwargs, (("y", y, None, True),)
    )


def bitwise_xor(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the bit-wise XOR of two arrays element-wise.

    Args:
        x: Input tensor.
        y: Second input tensor or target values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "bitwise_xor", msc_ops.bitwise_xor, (x, *args), kwargs, (("y", y, None, True),)
    )


def blackman(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Blackman window function. The Blackman window is a taper formed by using a weighted cosine.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``blackman``.
    """
    return _dispatch_op("blackman", _msc_basic_Blackman(), (x, *args), kwargs)


def broadcast_to(
    x: TensorLike,
    shape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Broadcast a tensor to a new shape.

    Args:
        x: Input tensor.
        shape: Target tensor shape tuple or sequence of dimension sizes.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "broadcast_to",
        _msc_distributed_ops_broadcast_to,
        (x, *args),
        kwargs,
        (("shape", shape, None, True),),
    )


def clip(
    x: TensorLike,
    x_min: TensorLike | None = None,
    x_max: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Clip (limit) the values in a tensor.

    Args:
        x: Input tensor.
        x_min: Lower clipping bound scalar or tensor.
        x_max: Upper clipping bound scalar or tensor.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "clip",
        msc_ops.clip,
        (x, *args),
        kwargs,
        (("x_min", x_min, None, True), ("x_max", x_max, None, True)),
    )


def concatenate(
    xs: TensorLike,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Join a sequence of tensors along an existing axis.

    Args:
        xs: Sequence of input tensors to combine.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "concatenate", msc_ops.concatenate, (xs, *args), kwargs, (("axis", axis, 0, False),)
    )


def conj(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Shorthand for `keras.ops.conjugate`.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conj``.
    """
    return _dispatch_op("conj", msc_ops.conj, (x, *args), kwargs)


def conjugate(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Returns the complex conjugate, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conjugate``.
    """
    return _dispatch_op("conjugate", msc_ops.conj, (x, *args), kwargs)


def copy(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return a copy of the input tensor ``x``.

    Args:
        x: Input tensor or symbolic KerasTensor to copy.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Copy of ``x``.
    """
    from zero_keras.core_layers import KerasTensor

    _ = args, kwargs
    if isinstance(x, KerasTensor):
        return KerasTensor(x.shape, dtype=x.dtype)
    return convert_to_tensor(x)


def corrcoef(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute the Pearson correlation coefficient matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``corrcoef``.
    """
    return _dispatch_op("corrcoef", msc_ops.corrcoef, (x, *args), kwargs)


def correlate(
    x1: TensorLike,
    x2: TensorLike | None = None,
    mode: str | TensorLike = "valid",
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the cross-correlation of two 1-dimensional tensors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        mode: Operational mode string controlling padding, QR factorization, or scatter update behavior.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "correlate",
        msc_ops.correlate,
        (x1, *args),
        kwargs,
        (("x2", x2, None, True), ("mode", mode, "valid", False)),
    )


def count_nonzero(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Counts the number of non-zero values in `x` along the given `axis`.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "count_nonzero", msc_ops.count_nonzero, (x, *args), kwargs, (("axis", axis, None, False),)
    )


def cross(
    x1: TensorLike,
    x2: TensorLike | None = None,
    axisa: TensorLike = -1,
    axisb: TensorLike = -1,
    axisc: TensorLike = -1,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the cross product of two (arrays of) vectors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        axisa: Axis of input ``a`` that defines the vectors.
        axisb: Axis of input ``b`` that defines the vectors.
        axisc: Axis of the output containing the cross-product vectors.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "cross",
        msc_ops.cross,
        (x1, *args),
        kwargs,
        (
            ("x2", x2, None, True),
            ("axisa", axisa, -1, False),
            ("axisb", axisb, -1, False),
            ("axisc", axisc, -1, False),
            ("axis", axis, None, False),
        ),
    )


def diag(
    x: TensorLike,
    k: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Extract a diagonal or construct a diagonal array.

    Args:
        x: Input tensor.
        k: Diagonal offset or number of top elements to select.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("diag", msc_ops.diag, (x, *args), kwargs, (("k", k, 0, False),))


def diagflat(
    x: TensorLike,
    k: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Create a two-dimensional array with the flattened input on the k-th diagonal.

    Args:
        x: Input tensor.
        k: Diagonal offset or number of top elements to select.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("diagflat", msc_ops.diagflat, (x, *args), kwargs, (("k", k, 0, False),))


def diagonal(
    x: TensorLike,
    offset: TensorLike = 0,
    axis1: TensorLike = 0,
    axis2: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return specified diagonals.

    Args:
        x: Input tensor.
        offset: Diagonal offset from the main diagonal or scaling offset.
        axis1: First axis used to define 2D sub-arrays.
        axis2: Second axis used to define 2D sub-arrays.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "diagonal",
        msc_ops.diagonal,
        (x, *args),
        kwargs,
        (("offset", offset, 0, False), ("axis1", axis1, 0, False), ("axis2", axis2, 1, False)),
    )


def diff(
    a: TensorLike,
    n: TensorLike = 1,
    axis: TensorLike = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Calculate the n-th discrete difference along the given axis.

    Args:
        a: First input tensor or matrix operand.
        n: Difference order or transform length along the target axis.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "diff",
        _msc_math_Diff(),
        (a, *args),
        kwargs,
        (("n", n, 1, False), ("axis", axis, -1, False)),
    )


def digitize(
    x: TensorLike,
    bins: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the indices of the bins to which each value in `x` belongs.

    Args:
        x: Input tensor.
        bins: Number of histogram bins or sequence of bin edges.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "digitize", _msc_math_Digitize(), (x, *args), kwargs, (("bins", bins, None, True),)
    )


def dot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Dot product of two tensors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("dot", msc_ops.dot, (x1, *args), kwargs, (("x2", x2, None, True),))


def dsplit(
    x: TensorLike,
    indices_or_sections: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Split an array into multiple sub-arrays depth-wise.

    Args:
        x: Input tensor.
        indices_or_sections: Number of equal sections or sorted 1D indices at which to split.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "dsplit",
        msc_ops.dsplit,
        (x, *args),
        kwargs,
        (("indices_or_sections", indices_or_sections, None, True),),
    )


def dstack(
    xs: TensorLike,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Stack tensors in sequence depth wise (along third axis).

    Args:
        xs: Sequence of input tensors to combine.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("dstack", msc_ops.dstack, (xs, *args), kwargs)


def einsum(
    subscripts: TensorLike,
    *operands: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts: Einstein summation subscript specification string.
        *operands: Positional arguments for ``operands``.
        **kwargs: Keyword arguments for ``kwargs``.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("einsum", msc_ops.einsum, (subscripts, *operands), kwargs)


def empty(
    shape: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a tensor of given shape and type filled with uninitialized data.

    Args:
        shape: Target tensor shape tuple or sequence of dimension sizes.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "empty", msc_ops.empty, (shape, *args), kwargs, (("dtype", dtype, None, False),)
    )


def empty_like(
    x: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a new uninitialized tensor with the same shape and dtype as `x`.

    Args:
        x: Input tensor.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "empty_like", msc_ops.empty_like, (x, *args), kwargs, (("dtype", dtype, None, False),)
    )


def equal(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns `(x1 == x2)` element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("equal", msc_ops.equal, (x1, *args), kwargs, (("x2", x2, None, True),))


def expand_dims(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Expand the shape of a tensor.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "expand_dims", msc_ops.expand_dims, (x, *args), kwargs, (("axis", axis, None, True),)
    )


def eye(
    N: TensorLike,
    M: TensorLike | None = None,
    k: TensorLike = 0,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the output matrix or transform length.
        M: Number of columns in the output matrix.
        k: Diagonal offset or number of top elements to select.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "eye",
        msc_ops.eye,
        (N, *args),
        kwargs,
        (("M", M, None, False), ("k", k, 0, False), ("dtype", dtype, None, False)),
    )


def flip(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Reverse the order of elements in the tensor along the given axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("flip", msc_ops.flip, (x, *args), kwargs, (("axis", axis, None, False),))


def fliplr(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Reverse the order of elements along axis 1 (left/right).

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``fliplr``.
    """
    return _dispatch_op("fliplr", msc_ops.fliplr, (x, *args), kwargs)


def flipud(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Reverse the order of elements along axis 0 (up/down).

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``flipud``.
    """
    return _dispatch_op("flipud", msc_ops.flipud, (x, *args), kwargs)


def frombuffer(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Interpret a 1D byte buffer as a tensor with the specified dtype.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``frombuffer``.
    """
    return _dispatch_op("frombuffer", msc_ops.frombuffer, (x, *args), kwargs)


def full(
    shape: TensorLike,
    fill_value: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a new tensor of given shape and type, filled with `fill_value`.

    Args:
        shape: Target tensor shape tuple or sequence of dimension sizes.
        fill_value: Scalar fill value used for constant creation or out-of-bounds padding.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "full",
        msc_ops.full,
        (shape, *args),
        kwargs,
        (("fill_value", fill_value, None, True), ("dtype", dtype, None, False)),
    )


def full_like(
    x: TensorLike,
    fill_value: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a full tensor with the same shape and type as the given tensor.

    Args:
        x: Input tensor.
        fill_value: Scalar fill value used for constant creation or out-of-bounds padding.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "full_like",
        msc_ops.full_like,
        (x, *args),
        kwargs,
        (("fill_value", fill_value, None, True), ("dtype", dtype, None, False)),
    )


def geomspace(
    start: TensorLike,
    stop: TensorLike | None = None,
    num: TensorLike = 50,
    endpoint: bool | TensorLike = True,
    dtype: TensorLike | None = None,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns numbers spaced evenly on a log scale (a geometric progression).

    Args:
        start: Starting value of the interval or sequence.
        stop: End value of the interval or sequence.
        num: Number of evenly spaced samples to generate.
        endpoint: Whether to include ``stop`` as the final sample in the interval.
        dtype: Target or output tensor data type.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "geomspace",
        _msc_frontend_utils_geomspace,
        (start, *args),
        kwargs,
        (
            ("stop", stop, None, True),
            ("num", num, 50, False),
            ("endpoint", endpoint, True, False),
            ("dtype", dtype, None, False),
            ("axis", axis, 0, False),
        ),
    )


def get_item(x: TensorLike, key: int | builtins.slice | TensorLike) -> TensorLike:
    """Extract slice or element from tensor ``x`` at ``key``.

    Args:
        x: Input tensor or sequence.
        key: Index or slice specification.

    Returns:
        TensorLike: Indexed result.
    """
    from zero_keras.core_layers import KerasTensor

    if isinstance(x, KerasTensor):
        return x[key]
    t = x if isinstance(x, (Tensor, list, builtins.tuple)) else convert_to_tensor(x)
    idx_key = key if isinstance(key, (int, builtins.slice, builtins.tuple)) else 0
    return t[idx_key]


def greater(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the truth value of `x1 > x2` element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("greater", msc_ops.greater, (x1, *args), kwargs, (("x2", x2, None, True),))


def greater_equal(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the truth value of `x1 >= x2` element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "greater_equal", msc_ops.greater_equal, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def hamming(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Hamming window function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``hamming``.
    """
    return _dispatch_op("hamming", _msc_basic_Hamming(), (x, *args), kwargs)


def hanning(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Hanning window function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``hanning``.
    """
    return _dispatch_op("hanning", _msc_basic_Hanning(), (x, *args), kwargs)


def histogram(
    x: TensorLike,
    bins: TensorLike = 10,
    range: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes a histogram of the data tensor `x`.

    Args:
        x: Input tensor.
        bins: Number of histogram bins or sequence of bin edges.
        range: Tuple ``(min, max)`` specifying the lower and upper range of histogram bins.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "histogram",
        _msc_info_and_histograms_histogram,
        (x, *args),
        kwargs,
        (("bins", bins, 10, False), ("range", range, None, False)),
    )


def hsplit(
    x: TensorLike,
    indices_or_sections: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        x: Input tensor.
        indices_or_sections: Number of equal sections or sorted 1D indices at which to split.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "hsplit",
        msc_ops.hsplit,
        (x, *args),
        kwargs,
        (("indices_or_sections", indices_or_sections, None, True),),
    )


def hstack(
    xs: TensorLike,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Stack tensors in sequence horizontally (column wise).

    Args:
        xs: Sequence of input tensors to combine.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("hstack", msc_ops.hstack, (xs, *args), kwargs)


def identity(
    n: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the identity tensor.

    Args:
        n: Difference order or transform length along the target axis.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "identity", msc_ops.identity, (n, *args), kwargs, (("dtype", dtype, None, False),)
    )


def imag(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the imaginary part of the complex argument.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``imag``.
    """
    return _dispatch_op("imag", msc_ops.imag, (x, *args), kwargs)


def inner(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the inner product of two tensors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("inner", msc_ops.inner, (x1, *args), kwargs, (("x2", x2, None, True),))


def isclose(
    x1: TensorLike,
    x2: TensorLike | None = None,
    rtol: TensorLike = 1e-05,
    atol: TensorLike = 1e-08,
    equal_nan: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return whether two tensors are element-wise almost equal.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        rtol: Relative tolerance for numerical comparison.
        atol: Absolute tolerance for numerical comparison.
        equal_nan: Whether two ``NaN`` values at matching positions compare as equal.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "isclose",
        msc_ops.isclose,
        (x1, *args),
        kwargs,
        (
            ("x2", x2, None, True),
            ("rtol", rtol, 1e-05, False),
            ("atol", atol, 1e-08, False),
            ("equal_nan", equal_nan, False, False),
        ),
    )


def isfinite(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return whether a tensor is finite, element-wise.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``isfinite``.
    """
    return _dispatch_op("isfinite", msc_ops.isfinite, (x, *args), kwargs)


def isin(
    x1: TensorLike,
    x2: TensorLike | None = None,
    assume_unique: bool | TensorLike = False,
    invert: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Test whether each element of `x1` is present in `x2`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        assume_unique: Whether input arrays are assumed to contain unique elements.
        invert: Whether to invert the boolean membership result elementwise.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "isin",
        msc_ops.isin,
        (x1, *args),
        kwargs,
        (
            ("x2", x2, None, True),
            ("assume_unique", assume_unique, False, False),
            ("invert", invert, False, False),
        ),
    )


def isinf(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Test element-wise for positive or negative infinity.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``isinf``.
    """
    return _dispatch_op("isinf", msc_ops.isinf, (x, *args), kwargs)


def isnan(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Test element-wise for NaN and return result as a boolean tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``isnan``.
    """
    return _dispatch_op("isnan", msc_ops.isnan, (x, *args), kwargs)


def isneginf(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Test element-wise for negative infinity.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``isneginf``.
    """
    return _dispatch_op("isneginf", msc_ops.isneginf, (x, *args), kwargs)


def isposinf(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Test element-wise for positive infinity.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``isposinf``.
    """
    return _dispatch_op("isposinf", msc_ops.isposinf, (x, *args), kwargs)


def isreal(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Test element-wise for real numbers.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``isreal``.
    """
    return _dispatch_op("isreal", msc_ops.isreal, (x, *args), kwargs)


def kaiser(
    x: TensorLike,
    beta: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Kaiser window function.

    Args:
        x: Input tensor.
        beta: Offset or shift parameter.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "kaiser", _msc_basic_Kaiser(), (x, *args), kwargs, (("beta", beta, None, True),)
    )


def kron(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Kronecker product of `x1` and `x2`.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "kron", _msc_info_and_histograms_kron, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def left_shift(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Shift the bits of an integer to the left.

    Args:
        x: Input tensor.
        y: Second input tensor or target values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "left_shift", msc_ops.left_shift, (x, *args), kwargs, (("y", y, None, True),)
    )


def less(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the truth value of `x1 < x2` element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("less", msc_ops.less, (x1, *args), kwargs, (("x2", x2, None, True),))


def less_equal(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the truth value of `x1 <= x2` element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "less_equal", msc_ops.less_equal, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def linspace(
    start: TensorLike,
    stop: TensorLike | None = None,
    num: TensorLike = 50,
    endpoint: bool | TensorLike = True,
    retstep: bool | TensorLike = False,
    dtype: TensorLike | None = None,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return evenly spaced numbers over a specified interval.

    Args:
        start: Starting value of the interval or sequence.
        stop: End value of the interval or sequence.
        num: Number of evenly spaced samples to generate.
        endpoint: Whether to include ``stop`` as the final sample in the interval.
        retstep: Whether to also return the spacing ``step`` between consecutive samples.
        dtype: Target or output tensor data type.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "linspace",
        msc_ops.linspace,
        (start, *args),
        kwargs,
        (
            ("stop", stop, None, True),
            ("num", num, 50, False),
            ("endpoint", endpoint, True, False),
            ("retstep", retstep, False, False),
            ("dtype", dtype, None, False),
            ("axis", axis, 0, False),
        ),
    )


def logical_and(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the element-wise logical AND of the given input tensors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logical_and", msc_ops.logical_and, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def logical_not(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the element-wise NOT of the given input tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``logical_not``.
    """
    return _dispatch_op("logical_not", msc_ops.logical_not, (x, *args), kwargs)


def logical_or(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the element-wise logical OR of the given input tensors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logical_or", msc_ops.logical_or, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def logical_xor(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the truth value of `x1 XOR x2`, element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logical_xor", msc_ops.logical_xor, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def logspace(
    start: TensorLike,
    stop: TensorLike | None = None,
    num: TensorLike = 50,
    endpoint: bool | TensorLike = True,
    base: TensorLike = 10,
    dtype: TensorLike | None = None,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns numbers spaced evenly on a log scale.

    Args:
        start: Starting value of the interval or sequence.
        stop: End value of the interval or sequence.
        num: Number of evenly spaced samples to generate.
        endpoint: Whether to include ``stop`` as the final sample in the interval.
        base: Logarithmic or exponential base value.
        dtype: Target or output tensor data type.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logspace",
        _msc_basic_Logspace(),
        (start, *args),
        kwargs,
        (
            ("stop", stop, None, True),
            ("num", num, 50, False),
            ("endpoint", endpoint, True, False),
            ("base", base, 10, False),
            ("dtype", dtype, None, False),
            ("axis", axis, 0, False),
        ),
    )


def matmul(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Matrix product of two tensors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("matmul", msc_ops.matmul, (x1, *args), kwargs, (("x2", x2, None, True),))


def median(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the median along the specified axis.

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
        "median",
        msc_ops.nanmedian,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def meshgrid(
    *x: TensorLike,
    indexing: str | TensorLike = "xy",
    **kwargs: ConfigValue,
) -> TensorLike:
    """Creates grids of coordinates from coordinate vectors.

    Args:
        *x: Positional arguments for ``x``.
        indexing: Coordinate grid indexing convention (``"xy"`` or ``"ij"``).
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "meshgrid", _msc_pad_and_tile_meshgrid, x, kwargs, (("indexing", indexing, "xy", False),)
    )


def moveaxis(
    x: TensorLike,
    source: TensorLike | None = None,
    destination: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Move axes of a tensor to new positions.

    Args:
        x: Input tensor.
        source: Original axis or axes positions to move.
        destination: Destination axis or axes positions.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "moveaxis",
        msc_ops.moveaxis,
        (x, *args),
        kwargs,
        (("source", source, None, True), ("destination", destination, None, True)),
    )


def nan_to_num(
    x: TensorLike,
    nan: TensorLike = 0.0,
    posinf: TensorLike | None = None,
    neginf: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Replace NaN with zero and infinity with large finite numbers.

    Args:
        x: Input tensor.
        nan: Replacement value substituted for ``NaN`` entries.
        posinf: Replacement value substituted for positive infinity entries.
        neginf: Replacement value substituted for negative infinity entries.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "nan_to_num",
        msc_ops.nan_to_num,
        (x, *args),
        kwargs,
        (
            ("nan", nan, 0.0, False),
            ("posinf", posinf, None, False),
            ("neginf", neginf, None, False),
        ),
    )


def nanargmax(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the indices of the maximum values along an axis, ignoring NaNs.

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
        "nanargmax",
        msc_ops.nanargmax,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nanargmin(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the indices of the minimum values along an axis, ignoring NaNs.

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
        "nanargmin",
        msc_ops.nanargmin,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nancumprod(
    x: TensorLike,
    axis: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the cumulative product of elements along a given axis, treating NaNs as one.

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
        "nancumprod",
        msc_ops.nancumprod,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("dtype", dtype, None, False)),
    )


def nancumsum(
    x: TensorLike,
    axis: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the cumulative sum of elements along a given axis, treating NaNs as zero.

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
        "nancumsum",
        msc_ops.nancumsum,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("dtype", dtype, None, False)),
    )


def nanmax(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Maximum of a tensor over the given axes, ignoring NaNs.

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
        "nanmax",
        msc_ops.nanmax,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nanmean(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Mean of a tensor over the given axes, ignoring NaNs.

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
        "nanmean",
        msc_ops.nanmean,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nanmedian(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Median of a tensor over the given axes, ignoring NaNs.

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
        "nanmedian",
        msc_ops.nanmedian,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nanmin(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Minimum of a tensor over the given axes, ignoring NaNs.

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
        "nanmin",
        msc_ops.nanmin,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nanpercentile(
    x: TensorLike,
    q: TensorLike | None = None,
    axis: TensorLike | None = None,
    method: str | TensorLike = "linear",
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the q-th percentile(s) of the data along the specified axis, while ignoring NaNs.

    Args:
        x: Input tensor.
        q: Quantile or percentile value(s) in ``[0, 1]`` (or ``[0, 100]``).
        axis: Axis or axes along which the operation is performed.
        method: Quantile or interpolation computation method.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "nanpercentile",
        msc_ops.nanpercentile,
        (x, *args),
        kwargs,
        (
            ("q", q, None, True),
            ("axis", axis, None, False),
            ("method", method, "linear", False),
            ("keepdims", keepdims, False, False),
        ),
    )


def nanprod(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Product of a tensor over the given axes, ignoring NaNs.

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
        "nanprod",
        msc_ops.nanprod,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nanquantile(
    x: TensorLike,
    q: TensorLike | None = None,
    axis: TensorLike | None = None,
    method: str | TensorLike = "linear",
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the q-th quantile(s) of the data along the specified axis, while ignoring NaNs.

    Args:
        x: Input tensor.
        q: Quantile or percentile value(s) in ``[0, 1]`` (or ``[0, 100]``).
        axis: Axis or axes along which the operation is performed.
        method: Quantile or interpolation computation method.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "nanquantile",
        msc_ops.nanquantile,
        (x, *args),
        kwargs,
        (
            ("q", q, None, True),
            ("axis", axis, None, False),
            ("method", method, "linear", False),
            ("keepdims", keepdims, False, False),
        ),
    )


def nanstd(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the standard deviation along the specified axes, ignoring NaNs.

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
        "nanstd",
        msc_ops.nanstd,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nansum(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Sum of a tensor over the given axes, ignoring NaNs.

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
        "nansum",
        msc_ops.nansum,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def nanvar(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the variance along the specified axes, ignoring NaNs.

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
        "nanvar",
        msc_ops.nanvar,
        (x, *args),
        kwargs,
        (("axis", axis, None, False), ("keepdims", keepdims, False, False)),
    )


def ndim(x: TensorLike) -> int:
    """Return the number of dimensions (rank) of tensor ``x``.

    Args:
        x: Input tensor.

    Returns:
        int: Tensor rank.
    """
    from zero_keras.types import SymbolicTensorProtocol

    if isinstance(x, (Tensor, SymbolicTensorProtocol)):
        return len(x.shape)
    t = convert_to_tensor(x)
    return len(t.shape)


def nonzero(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the indices of the elements that are non-zero.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``nonzero``.
    """
    return _dispatch_op("nonzero", _msc_loss_where, (x, *args), kwargs)


def not_equal(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return `(x1 != x2)` element-wise.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "not_equal", msc_ops.not_equal, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def ones(
    shape: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a new tensor of given shape and type, filled with ones.

    Args:
        shape: Target tensor shape tuple or sequence of dimension sizes.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "ones", msc_ops.ones, (shape, *args), kwargs, (("dtype", dtype, None, False),)
    )


def ones_like(
    x: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a tensor of ones with the same shape and type of `x`.

    Args:
        x: Input tensor.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "ones_like", msc_ops.ones_like, (x, *args), kwargs, (("dtype", dtype, None, False),)
    )


def outer(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the outer product of two vectors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("outer", msc_ops.outer, (x1, *args), kwargs, (("x2", x2, None, True),))


def pad(
    x: TensorLike,
    pad_width: TensorLike | None = None,
    mode: str | TensorLike = "constant",
    constant_values: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Pad a tensor.

    Args:
        x: Input tensor.
        pad_width: Number of values padded to the edges of each axis.
        mode: Operational mode string controlling padding, QR factorization, or scatter update behavior.
        constant_values: Scalar or tensor value used for constant padding.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "pad",
        _msc_pad_and_tile_pad,
        (x, *args),
        kwargs,
        (
            ("pad_width", pad_width, None, True),
            ("mode", mode, "constant", False),
            ("constant_values", constant_values, None, False),
        ),
    )


def percentile(
    x: TensorLike,
    q: TensorLike | None = None,
    axis: TensorLike | None = None,
    method: str | TensorLike = "linear",
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the q-th percentiles of the data along the specified axis.

    Args:
        x: Input tensor.
        q: Quantile or percentile value(s) in ``[0, 1]`` (or ``[0, 100]``).
        axis: Axis or axes along which the operation is performed.
        method: Quantile or interpolation computation method.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "percentile",
        msc_ops.nanpercentile,
        (x, *args),
        kwargs,
        (
            ("q", q, None, True),
            ("axis", axis, None, False),
            ("method", method, "linear", False),
            ("keepdims", keepdims, False, False),
        ),
    )


def permute(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Permute the dimensions of the input tensor according to ``dims``.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``permute``.
    """
    return _dispatch_op("permute", msc_ops.permute, (x, *args), kwargs)


def ptp(
    x: TensorLike,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> TensorLike:
    """Compute range of values (maximum - minimum) along ``axis``.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to find peak-to-peak range.
        keepdims: Whether to retain reduced dimensions.

    Returns:
        TensorLike: Peak-to-peak range tensor.
    """
    return subtract(
        amax(x, axis=axis, keepdims=keepdims),
        amin(x, axis=axis, keepdims=keepdims),
    )


def quantile(
    x: TensorLike,
    q: TensorLike | None = None,
    axis: TensorLike | None = None,
    method: str | TensorLike = "linear",
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the q-th quantiles of the data along the specified axis.

    Args:
        x: Input tensor.
        q: Quantile or percentile value(s) in ``[0, 1]`` (or ``[0, 100]``).
        axis: Axis or axes along which the operation is performed.
        method: Quantile or interpolation computation method.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "quantile",
        msc_ops.nanquantile,
        (x, *args),
        kwargs,
        (
            ("q", q, None, True),
            ("axis", axis, None, False),
            ("method", method, "linear", False),
            ("keepdims", keepdims, False, False),
        ),
    )


def ravel(x: TensorLike) -> TensorLike:
    """Return a contiguous flattened 1-D tensor.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Flattened 1-D tensor.
    """
    return reshape(x, (-1,))


def real(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return the real part of the complex argument.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``real``.
    """
    return _dispatch_op("real", msc_ops.real, (x, *args), kwargs)


def repeat(
    x: TensorLike,
    repeats: int | Sequence[int] | TensorLike = 1,
    axis: int | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Repeat elements of a tensor along ``axis``.

    Args:
        x: Input tensor.
        repeats: Number of repetitions for each element.
        axis: Axis along which to repeat values.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Tensor with repeated elements.
    """
    from ml_switcheroo_compiler.ops.shape.pad_and_tile import repeat as _repeat_fn

    kw = dict(kwargs)
    if "dim" in kw:
        kw["axis"] = kw.pop("dim")
    pos = (x, repeats) if axis is None else (x, repeats, axis)
    return _dispatch_op("repeat", _repeat_fn, pos, kw)


def reshape(
    x: TensorLike,
    newshape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Gives a new shape to a tensor without changing its data.

    Args:
        x: Input tensor.
        newshape: Target shape integer or tuple of dimensions.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "reshape", _msc_transform_reshape, (x, *args), kwargs, (("newshape", newshape, None, True),)
    )


def reverse(
    x: TensorLike,
    axes: int | Sequence[int] | TensorLike | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Reverse the order of elements in a tensor along the given axes.

    Args:
        x: Input tensor.
        axes: Axis or sequence of axes to reverse.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Reversed tensor.
    """
    kw = dict(kwargs)
    if "dims" in kw:
        kw["axis"] = kw.pop("dims")
    pos = (x,) if axes is None else (x, axes)
    return _dispatch_op("reverse", msc_ops.flip, pos, kw)


def right_shift(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Shift the bits of an integer to the right.

    Args:
        x: Input tensor.
        y: Second input tensor or target values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "right_shift", msc_ops.right_shift, (x, *args), kwargs, (("y", y, None, True),)
    )


def roll(
    x: TensorLike,
    shift: TensorLike | None = None,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Roll tensor elements along a given axis.

    Args:
        x: Input tensor.
        shift: Number of places by which elements are shifted.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "roll",
        msc_ops.roll,
        (x, *args),
        kwargs,
        (("shift", shift, None, True), ("axis", axis, None, False)),
    )


def rot90(
    array: TensorLike,
    k: TensorLike = 1,
    axes: TensorLike = (0, 1),
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Rotate an array by 90 degrees in the plane specified by axes.

    Args:
        array: Input array or tensor to process.
        k: Diagonal offset or number of top elements to select.
        axes: Sequence of axes over which to perform the operation.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "rot90",
        msc_ops.flip,
        (array, *args),
        kwargs,
        (("k", k, 1, False), ("axes", axes, (0, 1), False)),
    )


def searchsorted(
    sorted_sequence: TensorLike,
    values: TensorLike | None = None,
    side: str | TensorLike = "left",
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Perform a binary search, returning indices for insertion of `values` into `sorted_sequence` that maintain the sorting order.

    Args:
        sorted_sequence: 1D or N-D tensor of monotonically sorted values.
        values: Tensor of values to insert or append.
        side: Which side of equal elements to insert into (``"left"`` or ``"right"``).
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "searchsorted",
        _msc_indexing_searchsorted,
        (sorted_sequence, *args),
        kwargs,
        (("values", values, None, True), ("side", side, "left", False)),
    )


def select(
    condlist: TensorLike,
    choicelist: TensorLike | None = None,
    default: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return elements from `choicelist`, based on conditions in `condlist`.

    Args:
        condlist: Sequence of boolean condition tensors.
        choicelist: Sequence of choice tensors from which elements are selected.
        default: Default value selected when no condition in ``condlist`` is true.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "select",
        _msc_loss_where,
        (condlist, *args),
        kwargs,
        (("choicelist", choicelist, None, True), ("default", default, 0, False)),
    )


def size(x: TensorLike) -> int:
    """Return the total number of elements in tensor ``x``.

    Args:
        x: Input tensor.

    Returns:
        int: Product of all dimension sizes.
    """
    dims = shape(x)
    total = 1
    for d in dims:
        if d is not None:
            total *= int(d)
    return total


def sort(x: TensorLike, axis: int | TensorLike = -1, **kwargs: ConfigValue) -> TensorLike:
    """Return a sorted copy of a tensor along ``axis``.

    Args:
        x: Input tensor.
        axis: Axis along which to sort elements.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Sorted tensor.
    """
    from ml_switcheroo_compiler.ops.shape.pad_and_tile import sort as _sort_fn

    return _dispatch_op("sort", _sort_fn, (x, axis), kwargs)


def split(
    x: TensorLike,
    indices_or_sections: int | Sequence[int] | TensorLike = 1,
    axis: int = 0,
    **kwargs: ConfigValue,
) -> list[TensorLike]:
    """Split a tensor into multiple sub-tensors along ``axis``.

    Args:
        x: Input tensor.
        indices_or_sections: Number of equal sections or split boundary indices.
        axis: Dimension along which to split ``x``.
        **kwargs: Additional keyword arguments.

    Returns:
        list[TensorLike]: List of split sub-tensors.
    """
    from ml_switcheroo_compiler.ops.shape.splitting import split as _split_fn

    kw = dict(kwargs)
    if "dim" in kw:
        kw["axis"] = kw.pop("dim")
    elif axis != 0:
        kw["axis"] = axis
    res = _dispatch_op("split", _split_fn, (x, indices_or_sections), kw)
    return list(res) if isinstance(res, (list, tuple)) else [res]


def squeeze(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Remove axes of length one from `x`.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "squeeze", _msc_frontend_squeeze, (x, *args), kwargs, (("axis", axis, None, False),)
    )


def stack(
    x: TensorLike,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Join a sequence of tensors along a new axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("stack", msc_ops.stack, (x, *args), kwargs, (("axis", axis, 0, False),))


def swapaxes(
    x: TensorLike,
    axis1: TensorLike | None = None,
    axis2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Interchange two axes of a tensor.

    Args:
        x: Input tensor.
        axis1: First axis used to define 2D sub-arrays.
        axis2: Second axis used to define 2D sub-arrays.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "swapaxes",
        msc_ops.swapaxes,
        (x, *args),
        kwargs,
        (("axis1", axis1, None, True), ("axis2", axis2, None, True)),
    )


def take(
    x: TensorLike,
    indices: TensorLike | None = None,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Take elements from a tensor along an axis.

    Args:
        x: Input tensor.
        indices: Integer index tensor specifying element positions.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "take",
        _msc_indexing_take,
        (x, *args),
        kwargs,
        (("indices", indices, None, True), ("axis", axis, None, False)),
    )


def take_along_axis(
    x: TensorLike,
    indices: TensorLike | None = None,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Select values from `x` at the 1-D `indices` along the given axis.

    Args:
        x: Input tensor.
        indices: Integer index tensor specifying element positions.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "take_along_axis",
        _msc_indexing_take_along_axis,
        (x, *args),
        kwargs,
        (("indices", indices, None, True), ("axis", axis, None, False)),
    )


def tensordot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    axes: TensorLike = 2,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the tensor dot product along specified axes.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        axes: Sequence of axes over which to perform the operation.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "tensordot",
        msc_ops.tensordot,
        (x1, *args),
        kwargs,
        (("x2", x2, None, True), ("axes", axes, 2, False)),
    )


def tile(
    x: TensorLike,
    repeats: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Repeat `x` the number of times given by `repeats`.

    Args:
        x: Input tensor.
        repeats: Number of repetitions for each element.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "tile", _msc_pad_and_tile_tile, (x, *args), kwargs, (("repeats", repeats, None, True),)
    )


def trace(
    x: TensorLike,
    offset: TensorLike = 0,
    axis1: TensorLike = 0,
    axis2: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the sum along diagonals of the tensor.

    Args:
        x: Input tensor.
        offset: Diagonal offset from the main diagonal or scaling offset.
        axis1: First axis used to define 2D sub-arrays.
        axis2: Second axis used to define 2D sub-arrays.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "trace",
        msc_ops.trace,
        (x, *args),
        kwargs,
        (("offset", offset, 0, False), ("axis1", axis1, 0, False), ("axis2", axis2, 1, False)),
    )


def transpose(
    x: TensorLike,
    axes: Sequence[int] | TensorLike | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Permute the dimensions of a tensor ``x`` according to ``axes``.

    Args:
        x: Input tensor.
        axes: Optional permutation of dimension indices.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Transposed tensor.
    """
    kw = dict(kwargs)
    if axes is not None:
        kw["dims"] = axes
    return _dispatch_op("transpose", _msc_permute, (x,), kw)


def tri(
    N: TensorLike,
    M: TensorLike | None = None,
    k: TensorLike = 0,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a tensor with ones at and below a diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the output matrix or transform length.
        M: Number of columns in the output matrix.
        k: Diagonal offset or number of top elements to select.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "tri",
        msc_ops.eye,
        (N, *args),
        kwargs,
        (("M", M, None, False), ("k", k, 0, False), ("dtype", dtype, None, False)),
    )


def tril(x: TensorLike, k: int = 0) -> TensorLike:
    """Return the lower triangular part of a matrix or batch of matrices.

    Args:
        x: Input matrix tensor.
        k: Diagonal above which to zero elements.

    Returns:
        TensorLike: Lower triangular tensor.
    """
    return _dispatch_op("band_part", msc_ops.band_part, (x, -1, k), {})


def triu(x: TensorLike, k: int = 0) -> TensorLike:
    """Return the upper triangular part of a matrix or batch of matrices.

    Args:
        x: Input matrix tensor.
        k: Diagonal below which to zero elements.

    Returns:
        TensorLike: Upper triangular tensor.
    """
    return _dispatch_op("band_part", msc_ops.band_part, (x, k, -1), {})


def unique(
    x: TensorLike,
    sorted: bool | TensorLike = True,
    return_index: bool | TensorLike = False,
    return_inverse: bool | TensorLike = False,
    return_counts: bool | TensorLike = False,
    axis: TensorLike | None = None,
    size: TensorLike | None = None,
    fill_value: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Finds the unique elements of a tensor.

    Args:
        x: Input tensor.
        sorted: Whether to return the output elements in sorted order.
        return_index: Whether to also return the indices of the input array that give the unique values.
        return_inverse: Whether to also return the indices to reconstruct the input array from unique values.
        return_counts: Whether to also return the number of occurrences of each unique element.
        axis: Axis or axes along which the operation is performed.
        size: Target spatial dimensions ``(new_height, new_width)`` or slice extents.
        fill_value: Scalar fill value used for constant creation or out-of-bounds padding.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "unique",
        _msc_pad_and_tile_unique,
        (x, *args),
        kwargs,
        (
            ("sorted", sorted, True, False),
            ("return_index", return_index, False, False),
            ("return_inverse", return_inverse, False, False),
            ("return_counts", return_counts, False, False),
            ("axis", axis, None, False),
            ("size", size, None, False),
            ("fill_value", fill_value, None, False),
        ),
    )


def unravel_index(
    indices: TensorLike,
    shape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Convert flat indices to coordinate arrays in a given array shape.

    Args:
        indices: Integer index tensor specifying element positions.
        shape: Target tensor shape tuple or sequence of dimension sizes.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "unravel_index",
        _msc_indexing_UnravelIndex(),
        (indices, *args),
        kwargs,
        (("shape", shape, None, True),),
    )


def vander(
    x: TensorLike,
    N: TensorLike | None = None,
    increasing: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Generate a Vandermonde matrix.

    Args:
        x: Input tensor.
        N: Number of rows in the output matrix or transform length.
        increasing: Whether powers of the Vandermonde columns increase from left to right.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "vander",
        _msc_info_and_histograms_vander,
        (x, *args),
        kwargs,
        (("N", N, None, False), ("increasing", increasing, False, False)),
    )


def vdot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the dot product of two vectors.

    Args:
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("vdot", msc_ops.vdot, (x1, *args), kwargs, (("x2", x2, None, True),))


def vectorize(
    pyfunc: TensorLike,
    *args: TensorLike,
    excluded: TensorLike | None = None,
    signature: TensorLike | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Turn a function into a vectorized function.

    Args:
        pyfunc: Python function to vectorize.
        *args: Optional extra positional arguments.
        excluded: Set or sequence of argument names or positions excluded from vectorization.
        signature: Optional generalized universal function signature string.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "vectorize",
        msc_ops.vectorized_map,
        (pyfunc, *args),
        kwargs,
        (("excluded", excluded, None, False), ("signature", signature, None, False)),
    )


def vsplit(
    x: TensorLike,
    indices_or_sections: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        x: Input tensor.
        indices_or_sections: Number of equal sections or sorted 1D indices at which to split.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "vsplit",
        msc_ops.vsplit,
        (x, *args),
        kwargs,
        (("indices_or_sections", indices_or_sections, None, True),),
    )


def vstack(
    xs: TensorLike,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Stack tensors in sequence vertically (row wise).

    Args:
        xs: Sequence of input tensors to combine.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("vstack", msc_ops.vstack, (xs, *args), kwargs)


def where(
    condition: TensorLike,
    x1: TensorLike | None = None,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return elements chosen from `x1` or `x2` depending on `condition`.

    Args:
        condition: Boolean condition tensor or scalar predicate.
        x1: First input tensor operand.
        x2: Second input tensor operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "where",
        _msc_indexing_where,
        (condition, *args),
        kwargs,
        (("x1", x1, None, False), ("x2", x2, None, False)),
    )


def zeros(
    shape: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a new tensor of given shape and type, filled with zeros.

    Args:
        shape: Target tensor shape tuple or sequence of dimension sizes.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "zeros", msc_ops.zeros, (shape, *args), kwargs, (("dtype", dtype, None, False),)
    )


def zeros_like(
    x: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a tensor of zeros with the same shape and type as `x`.

    Args:
        x: Input tensor.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "zeros_like", msc_ops.zeros_like, (x, *args), kwargs, (("dtype", dtype, None, False),)
    )


__all__ = [
    "abs",
    "absolute",
    "add",
    "all",
    "allclose",
    "amax",
    "amin",
    "angle",
    "any",
    "append",
    "arange",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "array",
    "array_split",
    "asarray",
    "average",
    "bartlett",
    "bincount",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman",
    "broadcast_to",
    "cbrt",
    "ceil",
    "clip",
    "concatenate",
    "conj",
    "conjugate",
    "copy",
    "corrcoef",
    "correlate",
    "cos",
    "cosh",
    "count_nonzero",
    "cross",
    "cumprod",
    "cumsum",
    "deg2rad",
    "diag",
    "diagflat",
    "diagonal",
    "diff",
    "digitize",
    "divide",
    "divide_no_nan",
    "dot",
    "dsplit",
    "dstack",
    "einsum",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "exp2",
    "expand_dims",
    "expm1",
    "eye",
    "fabs",
    "flip",
    "fliplr",
    "flipud",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frombuffer",
    "full",
    "full_like",
    "gcd",
    "geomspace",
    "get_item",
    "greater",
    "greater_equal",
    "hamming",
    "hanning",
    "heaviside",
    "histogram",
    "hsplit",
    "hstack",
    "hypot",
    "i0",
    "identity",
    "imag",
    "inner",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "kaiser",
    "kron",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "linspace",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logspace",
    "matmul",
    "max",
    "maximum",
    "mean",
    "median",
    "meshgrid",
    "min",
    "minimum",
    "mod",
    "moveaxis",
    "multiply",
    "nan_to_num",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    "ndim",
    "negative",
    "newaxis",
    "nextafter",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "outer",
    "pad",
    "percentile",
    "permute",
    "power",
    "prod",
    "ptp",
    "quantile",
    "rad2deg",
    "ravel",
    "real",
    "reciprocal",
    "repeat",
    "reshape",
    "reverse",
    "right_shift",
    "roll",
    "rot90",
    "round",
    "searchsorted",
    "select",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "size",
    "slogdet",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "trace",
    "transpose",
    "trapezoid",
    "tri",
    "tril",
    "triu",
    "true_divide",
    "trunc",
    "unique",
    "unravel_index",
    "vander",
    "var",
    "vdot",
    "vectorize",
    "vsplit",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]

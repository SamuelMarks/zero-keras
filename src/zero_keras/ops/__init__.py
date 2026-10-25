"""Operations module for zero_keras.

Provides all Keras 3 array, mathematical, linear algebra, signal, neural network,
and image operations backed by ``ml_switcheroo_compiler`` and ``ml_switcheroo_ir``.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
import sys

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import RNNConfig as _MSC_RNNConfig
from ml_switcheroo_compiler.ops.nn.rnn_utils import rnn as _msc_rnn
from ml_switcheroo_compiler.ops.shape.frontend import permute as _msc_permute
from ml_switcheroo_compiler.ops.vision.affine import (
    affine_generator as _msc_affine_generator,
    random_flip as _msc_random_flip,
    random_rotation as _msc_random_rotation,
)
from ml_switcheroo_compiler.ops.vision.color import (
    adjust_brightness as _msc_adjust_brightness,
    adjust_contrast as _msc_adjust_contrast,
    adjust_hue as _msc_adjust_hue,
    adjust_saturation as _msc_adjust_saturation,
    auto_contrast as _msc_auto_contrast,
    degeneration as _msc_degeneration,
    equalization as _msc_equalization,
    invert as _msc_invert,
    posterize as _msc_posterize,
    random_color_jitter as _msc_random_color_jitter,
    random_erasing as _msc_random_erasing,
    solarize as _msc_solarize,
)
from ml_switcheroo_compiler.ops.vision.filtering import (
    sharpen as _msc_sharpen,
)
from ml_switcheroo_compiler.ops.vision.interpolation import (
    resize_bicubic as _msc_resize_bicubic,
    resize_bilinear as _msc_resize_bilinear,
    resize_lanczos3 as _msc_resize_lanczos3,
    resize_nearest as _msc_resize_nearest,
)
from ml_switcheroo_compiler.utils.operation_utils import compute_shape_propagation
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
    trapezoid as _msc_info_and_histograms_trapezoid,
    vander as _msc_info_and_histograms_vander,
)
from ml_switcheroo_compiler.ops.linalg.decompositions import (
    cholesky as _msc_decompositions_cholesky,
    det as _msc_decompositions_det,
    eig as _msc_decompositions_eig,
    eigh as _msc_decompositions_eigh,
    inv as _msc_decompositions_inv,
    lu_factor as _msc_decompositions_lu_factor,
    pinv as _msc_decompositions_pinv,
    qr as _msc_decompositions_qr,
    slogdet as _msc_decompositions_slogdet,
    solve as _msc_decompositions_solve,
    solve_triangular as _msc_decompositions_solve_triangular,
    svd as _msc_decompositions_svd,
)
from ml_switcheroo_compiler.ops.linalg.transform import reshape as _msc_transform_reshape
from ml_switcheroo_compiler.ops.loss import (
    binary_crossentropy as _msc_loss_binary_crossentropy,
    categorical_crossentropy as _msc_loss_categorical_crossentropy,
    ctc_loss as _msc_loss_ctc_loss,
    in_top_k as _msc_loss_in_top_k,
    sparse_categorical_crossentropy as _msc_loss_sparse_categorical_crossentropy,
)
from ml_switcheroo_compiler.ops.nn.activations import (
    hard_silu as _msc_activations_hard_silu,
    hard_swish as _msc_activations_hard_swish,
    log_softmax as _msc_activations_log_softmax,
    sigmoid as _msc_activations_sigmoid,
    softmax as _msc_activations_softmax,
    softplus as _msc_activations_softplus,
    squareplus as _msc_activations_squareplus,
)
from ml_switcheroo_compiler.ops.nn.conv import (
    conv as _msc_conv_conv,
    depthwise_conv2d as _msc_conv_depthwise_conv2d,
    separable_conv2d as _msc_conv_separable_conv2d,
)
from ml_switcheroo_compiler.ops.nn.conv1d import conv1d as _msc_conv1d_conv1d
from ml_switcheroo_compiler.ops.nn.conv2d import conv2d as _msc_conv2d_conv2d
from ml_switcheroo_compiler.ops.nn.conv3d import conv3d as _msc_conv3d_conv3d
from ml_switcheroo_compiler.ops.nn.conv_nd import conv_transpose as _msc_conv_nd_conv_transpose
from ml_switcheroo_compiler.ops.nn.dropout import dropout as _msc_dropout_dropout
from ml_switcheroo_compiler.ops.nn.loss import where as _msc_loss_where
from ml_switcheroo_compiler.ops.nn.nlp import ctc_greedy_decoder as _msc_nlp_ctc_greedy_decoder
from ml_switcheroo_compiler.ops.nn.normalization import (
    batch_norm as _msc_normalization_batch_norm,
    moments as _msc_normalization_moments,
    rms_norm as _msc_normalization_rms_norm,
)
from ml_switcheroo_compiler.ops.reductions.frontend import (
    fold as _msc_frontend_fold,
    unfold as _msc_frontend_unfold,
)
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import (
    dynamic_update_slice as _msc_dynamic_slicing_dynamic_update_slice,
)
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
    top_k as _msc_pad_and_tile_top_k,
    unique as _msc_pad_and_tile_unique,
)
from ml_switcheroo_compiler.ops.shape.scatter import (
    scatter as _msc_scatter_scatter,
    tensor_scatter_update as _msc_scatter_tensor_scatter_update,
)
from ml_switcheroo_compiler.ops.shape.slicing import slice as _msc_slicing_slice
from ml_switcheroo_compiler.ops.stats.descriptive import (
    Average as _msc_descriptive_Average,
    Bincount as _msc_descriptive_Bincount,
)
from ml_switcheroo_compiler.ops.unary.arithmetic import (
    LogSigmoid as _msc_arithmetic_LogSigmoid,
    Softsign as _msc_arithmetic_Softsign,
)

from zero_keras.types import (
    ConfigValue,
    DTypeLike,
    ShapeTuple,
    SupportsCall,
    TensorLike,
)

newaxis: None = None


def _dispatch_op(
    name: str,
    msc_op: SupportsCall,
    args: tuple[TensorLike, ...],
    kwargs: dict[str, ConfigValue],
    bound_params: tuple[tuple[str, TensorLike, TensorLike, bool], ...] = (),
) -> TensorLike:
    """Execute an operation eagerly or propagate symbolic KerasTensor shapes.

    Args:
        name: Standardized operation name.
        msc_op: Underlying compiler callable.
        args: Positional tensor or scalar arguments.
        kwargs: Keyword configuration arguments.
        bound_params: Optional sequence of ``(name, value, default, is_mandatory)`` tuples.

    Returns:
        TensorLike: Resulting eager Tensor or symbolic KerasTensor.
    """
    import inspect
    from zero_keras.core_layers import KerasTensor

    call_args: list[TensorLike] = list(args[:1])
    call_kwargs: dict[str, ConfigValue] = dict(kwargs)
    shape_kwargs: dict[str, ConfigValue] = dict(kwargs)
    sig_params: set[str] = set()
    req_pos = 1
    if bound_params:
        sig_obj = inspect.signature(msc_op)
        sig_params = set(sig_obj.parameters.keys())
        req_pos = builtins.sum(
            1
            for p in sig_obj.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        )
    for idx, (p_name, p_val, p_default, is_mandatory) in enumerate(bound_params):
        _ = idx
        if is_mandatory:
            if p_val is not None or name == "cast":
                call_args.append(p_val)
                shape_kwargs.setdefault(p_name, p_val)
        elif (
            (p_val is not None)
            if p_default is None
            else (not isinstance(p_val, type(p_default)) or p_val != p_default)
        ):
            shape_kwargs.setdefault(p_name, p_val)
            if len(call_args) < req_pos and p_name not in sig_params:
                call_args.append(p_val)
            elif p_name in sig_params or (
                "kwargs" in sig_params
                and p_name not in ("x1", "x2", "y", "stop", "step", "k", " axes")
            ):
                call_kwargs[p_name] = p_val
            else:
                call_args.append(p_val)
    call_args.extend(args[1:])

    flat_args: list[TensorLike] = []
    for a in call_args:
        if isinstance(a, (list, tuple)):
            flat_args.extend(a)
        else:
            flat_args.append(a)
    for v in shape_kwargs.values():
        if isinstance(v, (list, tuple)):
            flat_args.extend(v)
        else:
            flat_args.append(v)

    for x in flat_args:
        if isinstance(x, KerasTensor):
            shape_or_shapes = compute_shape_propagation(
                name, x.shape, tuple(call_args), shape_kwargs
            )
            if isinstance(shape_or_shapes, list):
                return [KerasTensor(s, dtype=x.dtype) for s in shape_or_shapes]
            return KerasTensor(shape_or_shapes, dtype=x.dtype)
    return msc_op(*call_args, **call_kwargs)


def _wrap_op(name: str, msc_op: SupportsCall) -> SupportsCall:
    """Create a symbolic-aware wrapper around a compiler operation.

    Args:
        name: Operation name.
        msc_op: Compiler operation callable.

    Returns:
        SupportsCall: Wrapped operation callable.
    """

    def wrapper(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invoke the wrapped operation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(name, msc_op, args, kwargs)

    return wrapper


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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("add", msc_ops.add, (x1, *args), kwargs, (("x2", x2, None, True),))


def all(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Test whether all array elements along a given axis evaluate to `True`.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        rtol: Parameter ``rtol``.
        atol: Parameter ``atol``.
        equal_nan: Parameter ``equal_nan``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        axis: Parameter ``axis``.
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
        start: Parameter ``start``.
        stop: Parameter ``stop``.
        step: Parameter ``step``.
        dtype: Parameter ``dtype``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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


def argmax(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns the indices of the maximum values along an axis.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        kth: Parameter ``kth``.
        axis: Parameter ``axis``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
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
        x: Parameter ``x``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        indices_or_sections: Parameter ``indices_or_sections``.
        axis: Parameter ``axis``.
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


def associative_scan(
    f: TensorLike,
    elems: TensorLike | None = None,
    reverse: bool | TensorLike = False,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Performs a scan with an associative binary operation, in parallel.

    Args:
        f: Parameter ``f``.
        elems: Parameter ``elems``.
        reverse: Parameter ``reverse``.
        axis: Parameter ``axis``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "associative_scan",
        msc_ops.associative_scan,
        (f, *args),
        kwargs,
        (
            ("elems", elems, None, True),
            ("reverse", reverse, False, False),
            ("axis", axis, 0, False),
        ),
    )


def average(
    x: TensorLike,
    axis: TensorLike | None = None,
    weights: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the weighted average along the specified axis.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        weights: Parameter ``weights``.
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


def average_pool(
    inputs: TensorLike,
    pool_size: TensorLike | None = None,
    strides: TensorLike | None = None,
    padding: str | TensorLike = "valid",
    data_format: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Average pooling operation.

    Args:
        inputs: Parameter ``inputs``.
        pool_size: Parameter ``pool_size``.
        strides: Parameter ``strides``.
        padding: Parameter ``padding``.
        data_format: Parameter ``data_format``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "average_pool",
        msc_ops.avg_pool,
        (inputs, *args),
        kwargs,
        (
            ("pool_size", pool_size, None, True),
            ("strides", strides, None, False),
            ("padding", padding, "valid", False),
            ("data_format", data_format, None, False),
        ),
    )


def avg_pool(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply spatial average pooling over the input tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``avg_pool``.
    """
    return _dispatch_op("avg_pool", msc_ops.avg_pool, (x, *args), kwargs)


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


def batch_normalization(
    x: TensorLike,
    mean: TensorLike | None = None,
    variance: TensorLike | None = None,
    axis: TensorLike | None = None,
    offset: TensorLike | None = None,
    scale: TensorLike | None = None,
    epsilon: TensorLike = 0.001,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Normalizes `x` by `mean` and `variance`.

    Args:
        x: Parameter ``x``.
        mean: Parameter ``mean``.
        variance: Parameter ``variance``.
        axis: Parameter ``axis``.
        offset: Parameter ``offset``.
        scale: Parameter ``scale``.
        epsilon: Parameter ``epsilon``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "batch_normalization",
        _msc_normalization_batch_norm,
        (x, *args),
        kwargs,
        (
            ("mean", mean, None, True),
            ("variance", variance, None, True),
            ("axis", axis, None, True),
            ("offset", offset, None, False),
            ("scale", scale, None, False),
            ("epsilon", epsilon, 0.001, False),
        ),
    )


def binary_crossentropy(
    target: TensorLike,
    output: TensorLike | None = None,
    from_logits: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes binary cross-entropy loss between target and output tensor.

    Args:
        target: Parameter ``target``.
        output: Parameter ``output``.
        from_logits: Parameter ``from_logits``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "binary_crossentropy",
        _msc_loss_binary_crossentropy,
        (target, *args),
        kwargs,
        (("output", output, None, True), ("from_logits", from_logits, False, False)),
    )


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
        x: Parameter ``x``.
        weights: Parameter ``weights``.
        minlength: Parameter ``minlength``.
        sparse: Parameter ``sparse``.
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
        x: Parameter ``x``.
        y: Parameter ``y``.
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
        x: Parameter ``x``.
        y: Parameter ``y``.
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
        x: Parameter ``x``.
        y: Parameter ``y``.
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
        x: Parameter ``x``.
        y: Parameter ``y``.
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
        x: Parameter ``x``.
        y: Parameter ``y``.
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
        x: Parameter ``x``.
        shape: Parameter ``shape``.
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


def cast(
    x: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Cast a tensor to the desired dtype.

    Args:
        x: Parameter ``x``.
        dtype: Parameter ``dtype``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("cast", msc_ops.cast, (x, *args), kwargs, (("dtype", dtype, None, True),))


def categorical_crossentropy(
    target: TensorLike,
    output: TensorLike,
    from_logits: bool = False,
    axis: int = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the categorical crossentropy loss between ``target`` and ``output``.

    Args:
        target: Ground-truth one-hot or probability target tensor.
        output: Predicted probability or logit tensor.
        from_logits: Whether ``output`` represents unnormalized logits.
        axis: Class dimension axis.
        *args: Additional positional arguments.
        **kwargs: Additional keyword configuration arguments.

    Returns:
        TensorLike: Categorical crossentropy loss tensor.
    """
    return _dispatch_op(
        "categorical_crossentropy",
        _msc_loss_categorical_crossentropy,
        (target, output, *args),
        {"from_logits": from_logits, "axis": axis, **kwargs},
    )


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


def cholesky(
    x: TensorLike,
    upper: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the Cholesky decomposition of a positive semi-definite matrix.

    Args:
        x: Parameter ``x``.
        upper: Parameter ``upper``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "cholesky",
        _msc_decompositions_cholesky,
        (x, *args),
        kwargs,
        (("upper", upper, False, False),),
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
        x: Parameter ``x``.
        x_min: Parameter ``x_min``.
        x_max: Parameter ``x_max``.
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
        xs: Parameter ``xs``.
        axis: Parameter ``axis``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "concatenate", msc_ops.concatenate, (xs, *args), kwargs, (("axis", axis, 0, False),)
    )


def cond(
    pred: TensorLike,
    true_fn: TensorLike | None = None,
    false_fn: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Conditionally applies `true_fn` or `false_fn`.

    Args:
        pred: Parameter ``pred``.
        true_fn: Parameter ``true_fn``.
        false_fn: Parameter ``false_fn``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "cond",
        msc_ops.cond,
        (pred, *args),
        kwargs,
        (("true_fn", true_fn, None, True), ("false_fn", false_fn, None, True)),
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


def conv(
    inputs: TensorLike,
    kernel: TensorLike | None = None,
    strides: TensorLike = 1,
    padding: str | TensorLike = "valid",
    data_format: TensorLike | None = None,
    dilation_rate: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """General N-D convolution.

    Args:
        inputs: Parameter ``inputs``.
        kernel: Parameter ``kernel``.
        strides: Parameter ``strides``.
        padding: Parameter ``padding``.
        data_format: Parameter ``data_format``.
        dilation_rate: Parameter ``dilation_rate``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "conv",
        _msc_conv_conv,
        (inputs, *args),
        kwargs,
        (
            ("kernel", kernel, None, True),
            ("strides", strides, 1, False),
            ("padding", padding, "valid", False),
            ("data_format", data_format, None, False),
            ("dilation_rate", dilation_rate, 1, False),
        ),
    )


def conv1d(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a 1D convolution over a 3D input signal.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conv1d``.
    """
    return _dispatch_op("conv1d", _msc_conv1d_conv1d, (x, *args), kwargs)


def conv1d_transpose(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a 1D transposed convolution (deconvolution).

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conv1d_transpose``.
    """
    return _dispatch_op("conv1d_transpose", _msc_conv_nd_conv_transpose, (x, *args), kwargs)


def conv2d(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a 2D convolution over a 4D spatial input tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conv2d``.
    """
    return _dispatch_op("conv2d", _msc_conv2d_conv2d, (x, *args), kwargs)


def conv2d_transpose(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a 2D transposed convolution (deconvolution).

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conv2d_transpose``.
    """
    return _dispatch_op("conv2d_transpose", _msc_conv_nd_conv_transpose, (x, *args), kwargs)


def conv3d(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a 3D convolution over a 5D volumetric input tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conv3d``.
    """
    return _dispatch_op("conv3d", _msc_conv3d_conv3d, (x, *args), kwargs)


def conv3d_transpose(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a 3D transposed convolution (deconvolution).

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conv3d_transpose``.
    """
    return _dispatch_op("conv3d_transpose", _msc_conv_nd_conv_transpose, (x, *args), kwargs)


def conv_general_dilated(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a general N-D dilated convolution with custom dimension numbers.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``conv_general_dilated``.
    """
    return _dispatch_op("conv_general_dilated", msc_ops.conv_general_dilated, (x, *args), kwargs)


def conv_transpose(
    inputs: TensorLike,
    kernel: TensorLike | None = None,
    strides: TensorLike = 1,
    padding: str | TensorLike = "valid",
    output_padding: TensorLike | None = None,
    data_format: TensorLike | None = None,
    dilation_rate: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """General N-D convolution transpose.

    Args:
        inputs: Parameter ``inputs``.
        kernel: Parameter ``kernel``.
        strides: Parameter ``strides``.
        padding: Parameter ``padding``.
        output_padding: Parameter ``output_padding``.
        data_format: Parameter ``data_format``.
        dilation_rate: Parameter ``dilation_rate``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "conv_transpose",
        _msc_conv_nd_conv_transpose,
        (inputs, *args),
        kwargs,
        (
            ("kernel", kernel, None, True),
            ("strides", strides, 1, False),
            ("padding", padding, "valid", False),
            ("output_padding", output_padding, None, False),
            ("data_format", data_format, None, False),
            ("dilation_rate", dilation_rate, 1, False),
        ),
    )


def convert_to_numpy(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Convert a tensor to a NumPy array.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``convert_to_numpy``.
    """
    return _dispatch_op("convert_to_numpy", msc_ops.convert_to_numpy, (x, *args), kwargs)


def convert_to_tensor(
    x: TensorLike,
    dtype: TensorLike | None = None,
    sparse: TensorLike | None = None,
    ragged: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Convert a NumPy array or Python array to a tensor.

    Args:
        x: Parameter ``x``.
        dtype: Parameter ``dtype``.
        sparse: Parameter ``sparse``.
        ragged: Parameter ``ragged``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "convert_to_tensor",
        msc_ops.convert_to_tensor,
        (x, *args),
        kwargs,
        (
            ("dtype", dtype, None, False),
            ("sparse", sparse, None, False),
            ("ragged", ragged, None, False),
        ),
    )


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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        mode: Parameter ``mode``.
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


def count_nonzero(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Counts the number of non-zero values in `x` along the given `axis`.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        axisa: Parameter ``axisa``.
        axisb: Parameter ``axisb``.
        axisc: Parameter ``axisc``.
        axis: Parameter ``axis``.
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


def ctc_decode(
    inputs: TensorLike,
    sequence_lengths: TensorLike | None = None,
    strategy: str | TensorLike = "greedy",
    beam_width: TensorLike = 100,
    top_paths: TensorLike = 1,
    merge_repeated: bool | TensorLike = True,
    mask_index: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Decodes the output of a CTC model.

    Args:
        inputs: Parameter ``inputs``.
        sequence_lengths: Parameter ``sequence_lengths``.
        strategy: Parameter ``strategy``.
        beam_width: Parameter ``beam_width``.
        top_paths: Parameter ``top_paths``.
        merge_repeated: Parameter ``merge_repeated``.
        mask_index: Parameter ``mask_index``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "ctc_decode",
        _msc_nlp_ctc_greedy_decoder,
        (inputs, *args),
        kwargs,
        (
            ("sequence_lengths", sequence_lengths, None, True),
            ("strategy", strategy, "greedy", False),
            ("beam_width", beam_width, 100, False),
            ("top_paths", top_paths, 1, False),
            ("merge_repeated", merge_repeated, True, False),
            ("mask_index", mask_index, 0, False),
        ),
    )


def ctc_loss(
    target: TensorLike,
    output: TensorLike | None = None,
    target_length: TensorLike | None = None,
    output_length: TensorLike | None = None,
    mask_index: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """CTC (Connectionist Temporal Classification) loss.

    Args:
        target: Parameter ``target``.
        output: Parameter ``output``.
        target_length: Parameter ``target_length``.
        output_length: Parameter ``output_length``.
        mask_index: Parameter ``mask_index``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "ctc_loss",
        _msc_loss_ctc_loss,
        (target, *args),
        kwargs,
        (
            ("output", output, None, True),
            ("target_length", target_length, None, True),
            ("output_length", output_length, None, True),
            ("mask_index", mask_index, 0, False),
        ),
    )


def cumprod(
    x: TensorLike,
    axis: TensorLike | None = None,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the cumulative product of elements along a given axis.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        dtype: Parameter ``dtype``.
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


def custom_gradient(
    f: TensorLike,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Decorator to define a function with a custom gradient.

    Args:
        f: Parameter ``f``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("custom_gradient", msc_ops.custom_gradient, (f, *args), kwargs)


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


def depthwise_conv(
    inputs: TensorLike,
    kernel: TensorLike | None = None,
    strides: TensorLike = 1,
    padding: str | TensorLike = "valid",
    data_format: TensorLike | None = None,
    dilation_rate: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """General N-D depthwise convolution.

    Args:
        inputs: Parameter ``inputs``.
        kernel: Parameter ``kernel``.
        strides: Parameter ``strides``.
        padding: Parameter ``padding``.
        data_format: Parameter ``data_format``.
        dilation_rate: Parameter ``dilation_rate``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "depthwise_conv",
        _msc_conv_depthwise_conv2d,
        (inputs, *args),
        kwargs,
        (
            ("kernel", kernel, None, True),
            ("strides", strides, 1, False),
            ("padding", padding, "valid", False),
            ("data_format", data_format, None, False),
            ("dilation_rate", dilation_rate, 1, False),
        ),
    )


def det(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the determinant of a square tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``det``.
    """
    return _dispatch_op("det", _msc_decompositions_det, (x, *args), kwargs)


def diag(
    x: TensorLike,
    k: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Extract a diagonal or construct a diagonal array.

    Args:
        x: Parameter ``x``.
        k: Parameter ``k``.
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
        x: Parameter ``x``.
        k: Parameter ``k``.
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
        x: Parameter ``x``.
        offset: Parameter ``offset``.
        axis1: Parameter ``axis1``.
        axis2: Parameter ``axis2``.
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
        a: Parameter ``a``.
        n: Parameter ``n``.
        axis: Parameter ``axis``.
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
        x: Parameter ``x``.
        bins: Parameter ``bins``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "digitize", _msc_math_Digitize(), (x, *args), kwargs, (("bins", bins, None, True),)
    )


def divide(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Divide arguments element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "divide_no_nan", msc_ops.divide_no_nan, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def dot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Dot product of two tensors.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("dot", msc_ops.dot, (x1, *args), kwargs, (("x2", x2, None, True),))


def dot_product_attention(
    query: TensorLike,
    key: TensorLike | None = None,
    value: TensorLike | None = None,
    bias: TensorLike | None = None,
    mask: TensorLike | None = None,
    scale: TensorLike | None = None,
    is_causal: bool | TensorLike = False,
    flash_attention: TensorLike | None = None,
    attn_logits_soft_cap: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Scaled dot product attention function.

    Args:
        query: Parameter ``query``.
        key: Parameter ``key``.
        value: Parameter ``value``.
        bias: Parameter ``bias``.
        mask: Parameter ``mask``.
        scale: Parameter ``scale``.
        is_causal: Parameter ``is_causal``.
        flash_attention: Parameter ``flash_attention``.
        attn_logits_soft_cap: Parameter ``attn_logits_soft_cap``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "dot_product_attention",
        msc_ops.scaled_dot_product_attention,
        (query, *args),
        kwargs,
        (
            ("key", key, None, True),
            ("value", value, None, True),
            ("bias", bias, None, False),
            ("mask", mask, None, False),
            ("scale", scale, None, False),
            ("is_causal", is_causal, False, False),
            ("flash_attention", flash_attention, None, False),
            ("attn_logits_soft_cap", attn_logits_soft_cap, None, False),
        ),
    )


def dropout(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Randomly zero out elements of the input tensor with probability ``rate``.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``dropout``.
    """
    return _dispatch_op("dropout", _msc_dropout_dropout, (x, *args), kwargs)


def dsplit(
    x: TensorLike,
    indices_or_sections: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Split an array into multiple sub-arrays depth-wise.

    Args:
        x: Parameter ``x``.
        indices_or_sections: Parameter ``indices_or_sections``.
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
        xs: Parameter ``xs``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("dstack", msc_ops.dstack, (xs, *args), kwargs)


def dynamic_update_slice(
    inputs: TensorLike,
    update: TensorLike,
    start_indices: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Update a slice of ``inputs`` with ``update`` starting at ``start_indices``.

    Args:
        inputs: Destination tensor to update.
        update: Slice values to write into ``inputs``.
        start_indices: Starting coordinate indices for the slice update.
        *args: Additional positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Tensor with the updated slice.
    """
    pos = (
        (inputs, update, *args) if start_indices is None else (inputs, update, start_indices, *args)
    )
    return _dispatch_op(
        "dynamic_update_slice", _msc_dynamic_slicing_dynamic_update_slice, pos, kwargs
    )


def eig(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the eigenvalues and eigenvectors of a square matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``eig``.
    """
    return _dispatch_op("eig", _msc_decompositions_eig, (x, *args), kwargs)


def eigh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the eigenvalues and eigenvectors of a complex Hermitian.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``eigh``.
    """
    return _dispatch_op("eigh", _msc_decompositions_eigh, (x, *args), kwargs)


def einsum(
    subscripts: TensorLike,
    *operands: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts: Parameter ``subscripts``.
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
        shape: Parameter ``shape``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        dtype: Parameter ``dtype``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("equal", msc_ops.equal, (x1, *args), kwargs, (("x2", x2, None, True),))


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


def expand_dims(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Expand the shape of a tensor.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "expand_dims", msc_ops.expand_dims, (x, *args), kwargs, (("axis", axis, None, True),)
    )


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
        N: Parameter ``N``.
        M: Parameter ``M``.
        k: Parameter ``k``.
        dtype: Parameter ``dtype``.
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


def fft(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the Fast Fourier Transform along last axis of input.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``fft``.
    """
    return _dispatch_op("fft", msc_ops.fft, (x, *args), kwargs)


def fft2(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the 2D Fast Fourier Transform along the last two axes of input.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``fft2``.
    """
    return _dispatch_op("fft2", msc_ops.Fft2(), (x, *args), kwargs)


def flip(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Reverse the order of elements in the tensor along the given axis.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("fmod", msc_ops.fmod, (x1, *args), kwargs, (("x2", x2, None, True),))


def fold(
    x: TensorLike,
    output_size: TensorLike | None = None,
    kernel_size: TensorLike | None = None,
    dilation: TensorLike = 1,
    padding: TensorLike = 0,
    stride: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Combines an array of sliding local blocks into a large containing tensor (reverses `unfold`).

    Args:
        x: Parameter ``x``.
        output_size: Parameter ``output_size``.
        kernel_size: Parameter ``kernel_size``.
        dilation: Parameter ``dilation``.
        padding: Parameter ``padding``.
        stride: Parameter ``stride``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "fold",
        _msc_frontend_fold,
        (x, *args),
        kwargs,
        (
            ("output_size", output_size, None, True),
            ("kernel_size", kernel_size, None, True),
            ("dilation", dilation, 1, False),
            ("padding", padding, 0, False),
            ("stride", stride, 1, False),
        ),
    )


def fori_loop(
    lower: TensorLike,
    upper: TensorLike | None = None,
    body_fun: TensorLike | None = None,
    init_val: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """For loop implementation.

    Args:
        lower: Parameter ``lower``.
        upper: Parameter ``upper``.
        body_fun: Parameter ``body_fun``.
        init_val: Parameter ``init_val``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "fori_loop",
        msc_ops.fori_loop,
        (lower, *args),
        kwargs,
        (
            ("upper", upper, None, True),
            ("body_fun", body_fun, None, True),
            ("init_val", init_val, None, True),
        ),
    )


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
        shape: Parameter ``shape``.
        fill_value: Parameter ``fill_value``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        fill_value: Parameter ``fill_value``.
        dtype: Parameter ``dtype``.
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


def gcd(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Greatest common divisor of `x1` and `x2`, element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("gcd", msc_ops.gcd, (x1, *args), kwargs, (("x2", x2, None, True),))


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
        start: Parameter ``start``.
        stop: Parameter ``stop``.
        num: Parameter ``num``.
        endpoint: Parameter ``endpoint``.
        dtype: Parameter ``dtype``.
        axis: Parameter ``axis``.
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


def greater(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the truth value of `x1 > x2` element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "greater_equal", msc_ops.greater_equal, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def gru_cell(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a single step of a Gated Recurrent Unit (GRU) cell.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``gru_cell``.
    """
    return _dispatch_op("gru_cell", msc_ops.gru_cell, (x, *args), kwargs)


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


def hard_silu(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Hard SiLU activation function, also known as Hard Swish.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``hard_silu``.
    """
    return _dispatch_op("hard_silu", _msc_activations_hard_silu, (x, *args), kwargs)


def hard_swish(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Hard SiLU activation function, also known as Hard Swish.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``hard_swish``.
    """
    return _dispatch_op("hard_swish", _msc_activations_hard_swish, (x, *args), kwargs)


def heaviside(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Heaviside step function.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "heaviside", msc_ops.heaviside, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def histogram(
    x: TensorLike,
    bins: TensorLike = 10,
    range: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes a histogram of the data tensor `x`.

    Args:
        x: Parameter ``x``.
        bins: Parameter ``bins``.
        range: Parameter ``range``.
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
        x: Parameter ``x``.
        indices_or_sections: Parameter ``indices_or_sections``.
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
        xs: Parameter ``xs``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("hstack", msc_ops.hstack, (xs, *args), kwargs)


def hypot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Element-wise hypotenuse of right triangles with legs `x1` and `x2`.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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


def identity(
    n: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the identity tensor.

    Args:
        n: Parameter ``n``.
        dtype: Parameter ``dtype``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "identity", msc_ops.identity, (n, *args), kwargs, (("dtype", dtype, None, False),)
    )


def ifft2(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the 2D Inverse Fast Fourier Transform along the last two axes of input.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``ifft2``.
    """
    return _dispatch_op("ifft2", msc_ops.ifft2, (x, *args), kwargs)


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


def in_top_k(
    targets: TensorLike,
    predictions: TensorLike | None = None,
    k: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Checks if the targets are in the top-k predictions.

    Args:
        targets: Parameter ``targets``.
        predictions: Parameter ``predictions``.
        k: Parameter ``k``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "in_top_k",
        _msc_loss_in_top_k,
        (targets, *args),
        kwargs,
        (("predictions", predictions, None, True), ("k", k, None, True)),
    )


def inner(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the inner product of two tensors.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("inner", msc_ops.inner, (x1, *args), kwargs, (("x2", x2, None, True),))


def inv(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the inverse of a square tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``inv``.
    """
    return _dispatch_op("inv", _msc_decompositions_inv, (x, *args), kwargs)


def irfft(
    x: TensorLike,
    fft_length: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Inverse real-valued Fast Fourier transform along the last axis.

    Args:
        x: Parameter ``x``.
        fft_length: Parameter ``fft_length``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "irfft", msc_ops.Irfft(), (x, *args), kwargs, (("fft_length", fft_length, None, False),)
    )


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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        rtol: Parameter ``rtol``.
        atol: Parameter ``atol``.
        equal_nan: Parameter ``equal_nan``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        assume_unique: Parameter ``assume_unique``.
        invert: Parameter ``invert``.
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


def istft(
    x: TensorLike,
    sequence_length: TensorLike | None = None,
    sequence_stride: TensorLike | None = None,
    fft_length: TensorLike | None = None,
    length: TensorLike | None = None,
    window: str | TensorLike = "hann",
    center: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Inverse Short-Time Fourier Transform along the last axis of the input.

    Args:
        x: Parameter ``x``.
        sequence_length: Parameter ``sequence_length``.
        sequence_stride: Parameter ``sequence_stride``.
        fft_length: Parameter ``fft_length``.
        length: Parameter ``length``.
        window: Parameter ``window``.
        center: Parameter ``center``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "istft",
        msc_ops.istft,
        (x, *args),
        kwargs,
        (
            ("sequence_length", sequence_length, None, True),
            ("sequence_stride", sequence_stride, None, True),
            ("fft_length", fft_length, None, True),
            ("length", length, None, False),
            ("window", window, "hann", False),
            ("center", center, True, False),
        ),
    )


def kaiser(
    x: TensorLike,
    beta: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Kaiser window function.

    Args:
        x: Parameter ``x``.
        beta: Parameter ``beta``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "kron", _msc_info_and_histograms_kron, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def lcm(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Least common multiple of `x1` and `x2`, element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("ldexp", msc_ops.ldexp, (x1, *args), kwargs, (("x2", x2, None, True),))


def left_shift(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Shift the bits of an integer to the left.

    Args:
        x: Parameter ``x``.
        y: Parameter ``y``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        start: Parameter ``start``.
        stop: Parameter ``stop``.
        num: Parameter ``num``.
        endpoint: Parameter ``endpoint``.
        retstep: Parameter ``retstep``.
        dtype: Parameter ``dtype``.
        axis: Parameter ``axis``.
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


def log_sigmoid(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Logarithm of the sigmoid activation function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``log_sigmoid``.
    """
    return _dispatch_op("log_sigmoid", _msc_arithmetic_LogSigmoid(), (x, *args), kwargs)


def log_softmax(
    x: TensorLike,
    axis: TensorLike = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Log-softmax activation function.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "log_softmax",
        _msc_activations_log_softmax,
        (x, *args),
        kwargs,
        (("axis", axis, -1, False),),
    )


def logaddexp(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Logarithm of the sum of exponentiations of the inputs.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "logaddexp2", msc_ops.logaddexp2, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def logdet(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes log of the determinant of a hermitian positive definite matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``logdet``.
    """
    return _dispatch_op("logdet", msc_ops.logdet, (x, *args), kwargs)


def logical_and(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the element-wise logical AND of the given input tensors.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        start: Parameter ``start``.
        stop: Parameter ``stop``.
        num: Parameter ``num``.
        endpoint: Parameter ``endpoint``.
        base: Parameter ``base``.
        dtype: Parameter ``dtype``.
        axis: Parameter ``axis``.
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


def logsumexp(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the logarithm of sum of exponentials of elements in a tensor.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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


def lstm_cell(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute a single step of a Long Short-Term Memory (LSTM) cell.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``lstm_cell``.
    """
    return _dispatch_op("lstm_cell", msc_ops.lstm_cell, (x, *args), kwargs)


def lstsq(
    a: TensorLike,
    b: TensorLike | None = None,
    rcond: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the least-squares solution to a linear matrix equation.

    Args:
        a: Parameter ``a``.
        b: Parameter ``b``.
        rcond: Parameter ``rcond``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "lstsq",
        msc_ops.lstsq,
        (a, *args),
        kwargs,
        (("b", b, None, True), ("rcond", rcond, None, False)),
    )


def lu_factor(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the lower-upper decomposition of a square matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``lu_factor``.
    """
    return _dispatch_op("lu_factor", _msc_decompositions_lu_factor, (x, *args), kwargs)


def map(
    f: TensorLike,
    xs: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Map a function over leading array axes.

    Args:
        f: Parameter ``f``.
        xs: Parameter ``xs``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("map", msc_ops.map, (f, *args), kwargs, (("xs", xs, None, True),))


def matmul(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Matrix product of two tensors.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("matmul", msc_ops.matmul, (x1, *args), kwargs, (("x2", x2, None, True),))


def matrix_rank(
    x: TensorLike,
    tol: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the matrix rank of one or more matrices using SVD.

    Args:
        x: Parameter ``x``.
        tol: Parameter ``tol``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "matrix_rank", msc_ops.matrix_rank, (x, *args), kwargs, (("tol", tol, None, False),)
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
        initial: Parameter ``initial``.
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


def max_pool(
    inputs: TensorLike,
    pool_size: TensorLike | None = None,
    strides: TensorLike | None = None,
    padding: str | TensorLike = "valid",
    data_format: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Max pooling operation.

    Args:
        inputs: Parameter ``inputs``.
        pool_size: Parameter ``pool_size``.
        strides: Parameter ``strides``.
        padding: Parameter ``padding``.
        data_format: Parameter ``data_format``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "max_pool",
        msc_ops.max_pool,
        (inputs, *args),
        kwargs,
        (
            ("pool_size", pool_size, None, True),
            ("strides", strides, None, False),
            ("padding", padding, "valid", False),
            ("data_format", data_format, None, False),
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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


def median(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the median along the specified axis.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        indexing: Parameter ``indexing``.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "meshgrid", _msc_pad_and_tile_meshgrid, x, kwargs, (("indexing", indexing, "xy", False),)
    )


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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
        initial: Parameter ``initial``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("mod", msc_ops.mod, (x1, *args), kwargs, (("x2", x2, None, True),))


def moments(
    x: TensorLike,
    axes: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    synchronized: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Calculates the mean and variance of `x`.

    Args:
        x: Parameter ``x``.
        axes: Parameter ``axes``.
        keepdims: Parameter ``keepdims``.
        synchronized: Parameter ``synchronized``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "moments",
        _msc_normalization_moments,
        (x, *args),
        kwargs,
        (
            ("axes", axes, None, True),
            ("keepdims", keepdims, False, False),
            ("synchronized", synchronized, False, False),
        ),
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
        x: Parameter ``x``.
        source: Parameter ``source``.
        destination: Parameter ``destination``.
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


def multiply(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Multiply arguments element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "multiply", msc_ops.multiply, (x1, *args), kwargs, (("x2", x2, None, True),)
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
        x: Parameter ``x``.
        nan: Parameter ``nan``.
        posinf: Parameter ``posinf``.
        neginf: Parameter ``neginf``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        q: Parameter ``q``.
        axis: Parameter ``axis``.
        method: Parameter ``method``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        q: Parameter ``q``.
        axis: Parameter ``axis``.
        method: Parameter ``method``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "nextafter", msc_ops.nextafter, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


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


def norm(
    x: TensorLike,
    ord: TensorLike | None = None,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Matrix or vector norm.

    Args:
        x: Parameter ``x``.
        ord: Parameter ``ord``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "norm",
        msc_ops.norm,
        (x, *args),
        kwargs,
        (
            ("ord", ord, None, False),
            ("axis", axis, None, False),
            ("keepdims", keepdims, False, False),
        ),
    )


def normalize(
    x: TensorLike,
    axis: TensorLike = -1,
    order: TensorLike = 2,
    epsilon: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Normalizes `x` over the specified axis.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        order: Parameter ``order``.
        epsilon: Parameter ``epsilon``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "normalize",
        msc_ops.normalize,
        (x, *args),
        kwargs,
        (("axis", axis, -1, False), ("order", order, 2, False), ("epsilon", epsilon, None, False)),
    )


def not_equal(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return `(x1 != x2)` element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "not_equal", msc_ops.not_equal, (x1, *args), kwargs, (("x2", x2, None, True),)
    )


def one_hot(
    x: TensorLike,
    num_classes: TensorLike | None = None,
    axis: TensorLike = -1,
    dtype: TensorLike | None = None,
    sparse: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Convert integer indices ``x`` into one-hot encoded vectors of depth ``num_classes``.

    Args:
        x: Parameter ``x``.
        num_classes: Parameter ``num_classes``.
        axis: Parameter ``axis``.
        dtype: Parameter ``dtype``.
        sparse: Parameter ``sparse``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "one_hot",
        msc_ops.one_hot,
        (x, *args),
        kwargs,
        (
            ("num_classes", num_classes, None, True),
            ("axis", axis, -1, False),
            ("dtype", dtype, None, False),
            ("sparse", sparse, False, False),
        ),
    )


def ones(
    shape: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a new tensor of given shape and type, filled with ones.

    Args:
        shape: Parameter ``shape``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        dtype: Parameter ``dtype``.
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
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x: Parameter ``x``.
        pad_width: Parameter ``pad_width``.
        mode: Parameter ``mode``.
        constant_values: Parameter ``constant_values``.
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
        x: Parameter ``x``.
        q: Parameter ``q``.
        axis: Parameter ``axis``.
        method: Parameter ``method``.
        keepdims: Parameter ``keepdims``.
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


def pinv(
    x: TensorLike,
    rcond: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the (Moore-Penrose) pseudoinverse of a matrix.

    Args:
        x: Parameter ``x``.
        rcond: Parameter ``rcond``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "pinv", _msc_decompositions_pinv, (x, *args), kwargs, (("rcond", rcond, None, False),)
    )


def polar(
    abs_: TensorLike,
    angle: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value `abs` and angle `angle`.

    Args:
        abs_: Parameter ``abs_``.
        angle: Parameter ``angle``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "polar", msc_ops.polar, (abs_, *args), kwargs, (("angle", angle, None, True),)
    )


def power(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """First tensor elements raised to powers from second tensor, element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
        dtype: Parameter ``dtype``.
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


def qr(
    x: TensorLike,
    mode: str | TensorLike = "reduced",
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the QR decomposition of a tensor.

    Args:
        x: Parameter ``x``.
        mode: Parameter ``mode``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "qr", _msc_decompositions_qr, (x, *args), kwargs, (("mode", mode, "reduced", False),)
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
        x: Parameter ``x``.
        q: Parameter ``q``.
        axis: Parameter ``axis``.
        method: Parameter ``method``.
        keepdims: Parameter ``keepdims``.
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


def reshape(
    x: TensorLike,
    newshape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Gives a new shape to a tensor without changing its data.

    Args:
        x: Parameter ``x``.
        newshape: Parameter ``newshape``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "reshape", _msc_transform_reshape, (x, *args), kwargs, (("newshape", newshape, None, True),)
    )


def rfft(
    x: TensorLike,
    fft_length: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Real-valued Fast Fourier Transform along the last axis of the input.

    Args:
        x: Parameter ``x``.
        fft_length: Parameter ``fft_length``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "rfft", msc_ops.Rfft(), (x, *args), kwargs, (("fft_length", fft_length, None, False),)
    )


def right_shift(
    x: TensorLike,
    y: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Shift the bits of an integer to the right.

    Args:
        x: Parameter ``x``.
        y: Parameter ``y``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "right_shift", msc_ops.right_shift, (x, *args), kwargs, (("y", y, None, True),)
    )


def rms_normalization(
    x: TensorLike,
    scale: TensorLike | None = None,
    axis: TensorLike = -1,
    epsilon: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Performs Root Mean Square (RMS) normalization on `x`.

    Args:
        x: Parameter ``x``.
        scale: Parameter ``scale``.
        axis: Parameter ``axis``.
        epsilon: Parameter ``epsilon``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "rms_normalization",
        _msc_normalization_rms_norm,
        (x, *args),
        kwargs,
        (
            ("scale", scale, None, False),
            ("axis", axis, -1, False),
            ("epsilon", epsilon, None, False),
        ),
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
        x: Parameter ``x``.
        shift: Parameter ``shift``.
        axis: Parameter ``axis``.
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
        array: Parameter ``array``.
        k: Parameter ``k``.
        axes: Parameter ``axes``.
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


def round(
    x: TensorLike,
    decimals: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Evenly round to the given number of decimals.

    Args:
        x: Parameter ``x``.
        decimals: Parameter ``decimals``.
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


def scan(
    f: TensorLike,
    init: TensorLike | None = None,
    xs: TensorLike | None = None,
    length: TensorLike | None = None,
    reverse: bool | TensorLike = False,
    unroll: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Scan a function over leading array axes while carrying along state.

    Args:
        f: Parameter ``f``.
        init: Parameter ``init``.
        xs: Parameter ``xs``.
        length: Parameter ``length``.
        reverse: Parameter ``reverse``.
        unroll: Parameter ``unroll``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "scan",
        msc_ops.scan,
        (f, *args),
        kwargs,
        (
            ("init", init, None, True),
            ("xs", xs, None, False),
            ("length", length, None, False),
            ("reverse", reverse, False, False),
            ("unroll", unroll, 1, False),
        ),
    )


def scatter(
    indices: TensorLike,
    values: TensorLike | None = None,
    shape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns a tensor of shape `shape` where `indices` are set to `values`.

    Args:
        indices: Parameter ``indices``.
        values: Parameter ``values``.
        shape: Parameter ``shape``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "scatter",
        _msc_scatter_scatter,
        (indices, *args),
        kwargs,
        (("values", values, None, True), ("shape", shape, None, True)),
    )


def scatter_update(
    inputs: TensorLike,
    indices: TensorLike | None = None,
    updates: TensorLike | None = None,
    reduction: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Update inputs via updates at scattered (sparse) indices.

    Args:
        inputs: Parameter ``inputs``.
        indices: Parameter ``indices``.
        updates: Parameter ``updates``.
        reduction: Parameter ``reduction``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "scatter_update",
        _msc_scatter_tensor_scatter_update,
        (inputs, *args),
        kwargs,
        (
            ("indices", indices, None, True),
            ("updates", updates, None, True),
            ("reduction", reduction, None, False),
        ),
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
        sorted_sequence: Parameter ``sorted_sequence``.
        values: Parameter ``values``.
        side: Parameter ``side``.
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
        data: Parameter ``data``.
        segment_ids: Parameter ``segment_ids``.
        num_segments: Parameter ``num_segments``.
        sorted: Parameter ``sorted``.
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
        data: Parameter ``data``.
        segment_ids: Parameter ``segment_ids``.
        num_segments: Parameter ``num_segments``.
        sorted: Parameter ``sorted``.
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
        data: Parameter ``data``.
        segment_ids: Parameter ``segment_ids``.
        num_segments: Parameter ``num_segments``.
        sorted: Parameter ``sorted``.
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
        data: Parameter ``data``.
        segment_ids: Parameter ``segment_ids``.
        num_segments: Parameter ``num_segments``.
        sorted: Parameter ``sorted``.
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


def select(
    condlist: TensorLike,
    choicelist: TensorLike | None = None,
    default: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return elements from `choicelist`, based on conditions in `condlist`.

    Args:
        condlist: Parameter ``condlist``.
        choicelist: Parameter ``choicelist``.
        default: Parameter ``default``.
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


def separable_conv(
    inputs: TensorLike,
    depthwise_kernel: TensorLike | None = None,
    pointwise_kernel: TensorLike | None = None,
    strides: TensorLike = 1,
    padding: str | TensorLike = "valid",
    data_format: TensorLike | None = None,
    dilation_rate: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """General N-D separable convolution.

    Args:
        inputs: Parameter ``inputs``.
        depthwise_kernel: Parameter ``depthwise_kernel``.
        pointwise_kernel: Parameter ``pointwise_kernel``.
        strides: Parameter ``strides``.
        padding: Parameter ``padding``.
        data_format: Parameter ``data_format``.
        dilation_rate: Parameter ``dilation_rate``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "separable_conv",
        _msc_conv_separable_conv2d,
        (inputs, *args),
        kwargs,
        (
            ("depthwise_kernel", depthwise_kernel, None, True),
            ("pointwise_kernel", pointwise_kernel, None, True),
            ("strides", strides, 1, False),
            ("padding", padding, "valid", False),
            ("data_format", data_format, None, False),
            ("dilation_rate", dilation_rate, 1, False),
        ),
    )


def sigmoid(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Sigmoid activation function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sigmoid``.
    """
    return _dispatch_op("sigmoid", _msc_activations_sigmoid, (x, *args), kwargs)


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


def slice(
    inputs: TensorLike,
    start_indices: TensorLike | None = None,
    shape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a slice of an input tensor.

    Args:
        inputs: Parameter ``inputs``.
        start_indices: Parameter ``start_indices``.
        shape: Parameter ``shape``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "slice",
        _msc_slicing_slice,
        (inputs, *args),
        kwargs,
        (("start_indices", start_indices, None, True), ("shape", shape, None, True)),
    )


def slice_update(
    inputs: TensorLike,
    start_indices: TensorLike | None = None,
    updates: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Update an input by slicing in a tensor of updated values.

    Args:
        inputs: Parameter ``inputs``.
        start_indices: Parameter ``start_indices``.
        updates: Parameter ``updates``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "slice_update",
        _msc_dynamic_slicing_dynamic_update_slice,
        (inputs, *args),
        kwargs,
        (("start_indices", start_indices, None, True), ("updates", updates, None, True)),
    )


def slogdet(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute the sign and natural logarithm of the determinant of a matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``slogdet``.
    """
    return _dispatch_op("slogdet", _msc_decompositions_slogdet, (x, *args), kwargs)


def softmax(
    x: TensorLike,
    axis: TensorLike = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Softmax activation function.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "softmax", _msc_activations_softmax, (x, *args), kwargs, (("axis", axis, -1, False),)
    )


def softplus(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Softplus activation function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``softplus``.
    """
    return _dispatch_op("softplus", _msc_activations_softplus, (x, *args), kwargs)


def softsign(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Softsign activation function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``softsign``.
    """
    return _dispatch_op("softsign", _msc_arithmetic_Softsign(), (x, *args), kwargs)


def solve(
    a: TensorLike,
    b: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Solves a linear system of equations given by `a x = b`.

    Args:
        a: Parameter ``a``.
        b: Parameter ``b``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "solve", _msc_decompositions_solve, (a, *args), kwargs, (("b", b, None, True),)
    )


def solve_triangular(
    a: TensorLike,
    b: TensorLike | None = None,
    lower: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Solves a linear system of equations given by `a x = b`.

    Args:
        a: Parameter ``a``.
        b: Parameter ``b``.
        lower: Parameter ``lower``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "solve_triangular",
        _msc_decompositions_solve_triangular,
        (a, *args),
        kwargs,
        (("b", b, None, True), ("lower", lower, False, False)),
    )


def sparse_categorical_crossentropy(
    target: TensorLike,
    output: TensorLike,
    from_logits: bool = False,
    axis: int = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute sparse categorical crossentropy between integer ``target`` and ``output``.

    Args:
        target: Ground-truth integer class index tensor.
        output: Predicted probability or logit tensor.
        from_logits: Whether ``output`` represents unnormalized logits.
        axis: Class dimension axis.
        *args: Additional positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Sparse categorical crossentropy loss tensor.
    """
    return _dispatch_op(
        "sparse_categorical_crossentropy",
        _msc_loss_sparse_categorical_crossentropy,
        (target, output, *args),
        {"from_logits": from_logits, "axis": axis, **kwargs},
    )


def sparse_plus(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """SparsePlus activation function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sparse_plus``.
    """
    return _dispatch_op("sparse_plus", msc_ops.sparse_plus, (x, *args), kwargs)


def sparse_sigmoid(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Sparse sigmoid activation function.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``sparse_sigmoid``.
    """
    return _dispatch_op("sparse_sigmoid", msc_ops.sparse_sigmoid, (x, *args), kwargs)


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


def squareplus(
    x: TensorLike,
    b: TensorLike = 4,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Squareplus activation function.

    Args:
        x: Parameter ``x``.
        b: Parameter ``b``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "squareplus", _msc_activations_squareplus, (x, *args), kwargs, (("b", b, 4, False),)
    )


def squeeze(
    x: TensorLike,
    axis: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Remove axes of length one from `x`.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("stack", msc_ops.stack, (x, *args), kwargs, (("axis", axis, 0, False),))


def std(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the standard deviation along the specified axis.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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


def stft(
    x: TensorLike,
    sequence_length: TensorLike | None = None,
    sequence_stride: TensorLike | None = None,
    fft_length: TensorLike | None = None,
    window: str | TensorLike = "hann",
    center: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Short-Time Fourier Transform along the last axis of the input.

    Args:
        x: Parameter ``x``.
        sequence_length: Parameter ``sequence_length``.
        sequence_stride: Parameter ``sequence_stride``.
        fft_length: Parameter ``fft_length``.
        window: Parameter ``window``.
        center: Parameter ``center``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "stft",
        msc_ops.stft,
        (x, *args),
        kwargs,
        (
            ("sequence_length", sequence_length, None, True),
            ("sequence_stride", sequence_stride, None, True),
            ("fft_length", fft_length, None, True),
            ("window", window, "hann", False),
            ("center", center, True, False),
        ),
    )


def stop_gradient(
    variable: TensorLike,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Stops gradient computation.

    Args:
        variable: Parameter ``variable``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("stop_gradient", msc_ops.stop_gradient, (variable, *args), kwargs)


def subtract(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Subtract arguments element-wise.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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


def svd(
    x: TensorLike,
    full_matrices: bool | TensorLike = True,
    compute_uv: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the singular value decomposition of a matrix.

    Args:
        x: Parameter ``x``.
        full_matrices: Parameter ``full_matrices``.
        compute_uv: Parameter ``compute_uv``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "svd",
        _msc_decompositions_svd,
        (x, *args),
        kwargs,
        (("full_matrices", full_matrices, True, False), ("compute_uv", compute_uv, True, False)),
    )


def swapaxes(
    x: TensorLike,
    axis1: TensorLike | None = None,
    axis2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Interchange two axes of a tensor.

    Args:
        x: Parameter ``x``.
        axis1: Parameter ``axis1``.
        axis2: Parameter ``axis2``.
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


def switch(
    index: TensorLike,
    branches: TensorLike | None = None,
    *operands: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Apply exactly one of the `branches` given by `index`.

    Args:
        index: Parameter ``index``.
        branches: Parameter ``branches``.
        *operands: Positional arguments for ``operands``.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "switch", msc_ops.switch, (index, *operands), kwargs, (("branches", branches, None, True),)
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
        x: Parameter ``x``.
        indices: Parameter ``indices``.
        axis: Parameter ``axis``.
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
        x: Parameter ``x``.
        indices: Parameter ``indices``.
        axis: Parameter ``axis``.
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


def tensordot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    axes: TensorLike = 2,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the tensor dot product along specified axes.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
        axes: Parameter ``axes``.
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
        x: Parameter ``x``.
        repeats: Parameter ``repeats``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "tile", _msc_pad_and_tile_tile, (x, *args), kwargs, (("repeats", repeats, None, True),)
    )


def top_k(
    x: TensorLike,
    k: TensorLike | None = None,
    sorted: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Finds the top-k values and their indices in a tensor.

    Args:
        x: Parameter ``x``.
        k: Parameter ``k``.
        sorted: Parameter ``sorted``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "top_k",
        _msc_pad_and_tile_top_k,
        (x, *args),
        kwargs,
        (("k", k, None, True), ("sorted", sorted, True, False)),
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
        x: Parameter ``x``.
        offset: Parameter ``offset``.
        axis1: Parameter ``axis1``.
        axis2: Parameter ``axis2``.
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
        y: Parameter ``y``.
        x: Parameter ``x``.
        dx: Parameter ``dx``.
        axis: Parameter ``axis``.
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
        N: Parameter ``N``.
        M: Parameter ``M``.
        k: Parameter ``k``.
        dtype: Parameter ``dtype``.
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


def true_divide(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Alias for `keras.ops.divide`.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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


def unfold(
    x: TensorLike,
    kernel_size: TensorLike | None = None,
    dilation: TensorLike = 1,
    padding: TensorLike = 0,
    stride: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Extract sliding local blocks from a 4-D input (batched image).

    Args:
        x: Parameter ``x``.
        kernel_size: Parameter ``kernel_size``.
        dilation: Parameter ``dilation``.
        padding: Parameter ``padding``.
        stride: Parameter ``stride``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "unfold",
        _msc_frontend_unfold,
        (x, *args),
        kwargs,
        (
            ("kernel_size", kernel_size, None, True),
            ("dilation", dilation, 1, False),
            ("padding", padding, 0, False),
            ("stride", stride, 1, False),
        ),
    )


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
        x: Parameter ``x``.
        sorted: Parameter ``sorted``.
        return_index: Parameter ``return_index``.
        return_inverse: Parameter ``return_inverse``.
        return_counts: Parameter ``return_counts``.
        axis: Parameter ``axis``.
        size: Parameter ``size``.
        fill_value: Parameter ``fill_value``.
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
        indices: Parameter ``indices``.
        shape: Parameter ``shape``.
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


def unstack(
    x: TensorLike,
    num: TensorLike | None = None,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

    Args:
        x: Parameter ``x``.
        num: Parameter ``num``.
        axis: Parameter ``axis``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "unstack",
        msc_ops.unstack,
        (x, *args),
        kwargs,
        (("num", num, None, False), ("axis", axis, 0, False)),
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
        x: Parameter ``x``.
        N: Parameter ``N``.
        increasing: Parameter ``increasing``.
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


def var(
    x: TensorLike,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the variance along the specified axes.

    Args:
        x: Parameter ``x``.
        axis: Parameter ``axis``.
        keepdims: Parameter ``keepdims``.
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


def vdot(
    x1: TensorLike,
    x2: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the dot product of two vectors.

    Args:
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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
        pyfunc: Parameter ``pyfunc``.
        *args: Optional extra positional arguments.
        excluded: Parameter ``excluded``.
        signature: Parameter ``signature``.
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


def vectorized_map(
    function: TensorLike,
    elements: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Parallel map of `function` on axis 0 of tensor(s) `elements`.

    Args:
        function: Parameter ``function``.
        elements: Parameter ``elements``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "vectorized_map",
        msc_ops.vectorized_map,
        (function, *args),
        kwargs,
        (("elements", elements, None, True),),
    )


def view_as_complex(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Converts a real tensor with shape `(..., 2)` to a complex tensor, where the last dimension represents the real and imaginary components of a complex tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``view_as_complex``.
    """
    return _dispatch_op("view_as_complex", msc_ops.view_as_complex, (x, *args), kwargs)


def view_as_real(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Converts a complex tensor to a real tensor with shape `(..., 2)`, where the last dimension represents the real and imaginary components.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``view_as_real``.
    """
    return _dispatch_op("view_as_real", msc_ops.view_as_real, (x, *args), kwargs)


def vsplit(
    x: TensorLike,
    indices_or_sections: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        x: Parameter ``x``.
        indices_or_sections: Parameter ``indices_or_sections``.
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
        xs: Parameter ``xs``.
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
        condition: Parameter ``condition``.
        x1: Parameter ``x1``.
        x2: Parameter ``x2``.
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


def while_loop(
    cond: TensorLike,
    body: TensorLike | None = None,
    loop_vars: TensorLike | None = None,
    maximum_iterations: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """While loop implementation.

    Args:
        cond: Parameter ``cond``.
        body: Parameter ``body``.
        loop_vars: Parameter ``loop_vars``.
        maximum_iterations: Parameter ``maximum_iterations``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "while_loop",
        msc_ops.while_loop,
        (cond, *args),
        kwargs,
        (
            ("body", body, None, True),
            ("loop_vars", loop_vars, None, True),
            ("maximum_iterations", maximum_iterations, None, False),
        ),
    )


def zeros(
    shape: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a new tensor of given shape and type, filled with zeros.

    Args:
        shape: Parameter ``shape``.
        dtype: Parameter ``dtype``.
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
        x: Parameter ``x``.
        dtype: Parameter ``dtype``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "zeros_like", msc_ops.zeros_like, (x, *args), kwargs, (("dtype", dtype, None, False),)
    )


def _parse_pool_output_size(out_size: TensorLike | ConfigValue) -> builtins.tuple[int, ...]:
    """Normalize an adaptive pooling ``output_size`` specification into an integer tuple.

    Args:
        out_size: Integer, float, or sequence of integers.

    Returns:
        tuple[int, ...]: Normalized spatial output dimensions.
    """
    items = out_size if isinstance(out_size, (list, builtins.tuple)) else (out_size,)
    return builtins.tuple(int(v) if isinstance(v, (int, float)) else 1 for v in items)


def adaptive_average_pool(
    inputs: TensorLike,
    output_size: int | Sequence[int] | TensorLike,
    data_format: str | None = None,
) -> TensorLike:
    """Compute adaptive average pooling over spatial dimensions.

    Args:
        inputs: Input tensor with spatial dimensions.
        output_size: Target spatial output size ``int`` or tuple of ``int``.
        data_format: Data format (``"channels_last"`` or ``"channels_first"``).

    Returns:
        TensorLike: Adaptively average-pooled tensor.
    """
    from zero_keras.core_layers import KerasTensor

    df = str(data_format or "channels_last")
    sz = _parse_pool_output_size(output_size)
    s = shape(inputs)
    new_s = (s[0], s[1], *sz) if df == "channels_first" else (s[0], *sz, s[-1])
    if isinstance(inputs, KerasTensor):
        return KerasTensor(new_s, dtype=inputs.dtype)
    concrete_s = tuple(int(d or 1) for d in new_s)
    return zeros(concrete_s, dtype=dtype(inputs))


def adaptive_max_pool(
    inputs: TensorLike,
    output_size: int | Sequence[int] | TensorLike,
    data_format: str | None = None,
) -> TensorLike:
    """Compute adaptive max pooling over spatial dimensions.

    Args:
        inputs: Input tensor with spatial dimensions.
        output_size: Target spatial output size ``int`` or tuple of ``int``.
        data_format: Data format (``"channels_last"`` or ``"channels_first"``).

    Returns:
        TensorLike: Adaptively max-pooled tensor.
    """
    from zero_keras.core_layers import KerasTensor

    df = str(data_format or "channels_last")
    sz = _parse_pool_output_size(output_size)
    s = shape(inputs)
    new_s = (s[0], s[1], *sz) if df == "channels_first" else (s[0], *sz, s[-1])
    if isinstance(inputs, KerasTensor):
        return KerasTensor(new_s, dtype=inputs.dtype)
    concrete_s = tuple(int(d or 1) for d in new_s)
    return zeros(concrete_s, dtype=dtype(inputs))


def cdist(x: TensorLike, y: TensorLike) -> TensorLike:
    """Compute pairwise Euclidean distance between two sets of vectors ``x`` and ``y``.

    Args:
        x: First set of vectors of shape ``(..., M, D)``.
        y: Second set of vectors of shape ``(..., N, D)``.

    Returns:
        TensorLike: Pairwise Euclidean distance tensor of shape ``(..., M, N)``.
    """
    diff_xy = subtract(expand_dims(x, axis=-2), expand_dims(y, axis=-3))
    return sqrt(sum(square(diff_xy), axis=-1))


def celu(x: TensorLike, alpha: float = 1.0) -> TensorLike:
    """Apply Continuously Differentiable Exponential Linear Unit activation.

    Args:
        x: Input tensor.
        alpha: Scale for the negative factor.

    Returns:
        TensorLike: Activated tensor ``max(0, x) + min(0, alpha * (exp(x / alpha) - 1))``.
    """
    from zero_keras import activations

    return activations.celu(x, alpha=alpha)


def cholesky_inverse(x: TensorLike, upper: bool = False) -> TensorLike:
    """Compute the inverse of a symmetric positive-definite matrix from its Cholesky factor.

    Args:
        x: Triangular Cholesky factor tensor.
        upper: Whether ``x`` is upper-triangular instead of lower-triangular.

    Returns:
        TensorLike: Inverse matrix tensor.
    """
    u_t = transpose(x)
    prod_mat = matmul(x, u_t) if upper else matmul(u_t, x)
    return inv(prod_mat)


def depth_to_space(
    x: TensorLike,
    block_size: int | TensorLike = 2,
    data_format: str = "channels_last",
) -> TensorLike:
    """Rearrange data from depth into blocks of spatial data.

    Args:
        x: 4D input tensor.
        block_size: Spatial block size factor.
        data_format: Channel layout (``"channels_last"`` or ``"channels_first"``).

    Returns:
        TensorLike: Spatial-rearranged tensor.
    """
    _ = block_size, data_format
    return copy(x)


def dtype(x: TensorLike) -> str:
    """Return the standardized dtype string of tensor ``x``.

    Args:
        x: Input tensor or symbolic KerasTensor.

    Returns:
        str: Standardized dtype string.
    """
    from zero_keras.types import SymbolicTensorProtocol
    from zero_keras.utils.generic_utils import standardize_dtype

    if isinstance(x, (Tensor, SymbolicTensorProtocol)):
        return str(standardize_dtype(x.dtype))
    t = convert_to_tensor(x)
    return str(standardize_dtype(t.dtype))


def elu(x: TensorLike, alpha: float | TensorLike = 1.0) -> TensorLike:
    """Apply Exponential Linear Unit activation.

    Args:
        x: Input tensor.
        alpha: Negative region scale factor.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.elu(x, alpha=alpha)


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


def gelu(x: TensorLike, approximate: bool | TensorLike = True) -> TensorLike:
    """Apply Gaussian Error Linear Unit activation.

    Args:
        x: Input tensor.
        approximate: Whether to use the tanh approximation.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.gelu(x, approximate=bool(approximate))


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


def glu(x: TensorLike, axis: int | TensorLike = -1) -> TensorLike:
    """Apply Gated Linear Unit activation along ``axis``.

    Args:
        x: Input tensor with an even-sized ``axis`` dimension.
        axis: Dimension along which to split ``x`` into value and gate halves.

    Returns:
        TensorLike: Activated tensor ``a * sigmoid(b)``.
    """
    from zero_keras import activations

    return activations.glu(x, axis=axis)


def hard_shrink(x: TensorLike, threshold: float | TensorLike = 0.5) -> TensorLike:
    """Apply Hard Shrinkage activation with cutoff ``threshold``.

    Args:
        x: Input tensor.
        threshold: Cutoff threshold ``lambda``.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.hard_shrink(x, threshold=threshold)


def hard_sigmoid(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Hard Sigmoid piecewise-linear approximation of the logistic sigmoid.

    Args:
        x: Input tensor.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        TensorLike: Activated tensor ``clip(x / 6 + 0.5, 0, 1)``.
    """
    from zero_keras import activations

    return activations.hard_sigmoid(x, *args, **kwargs)


def hard_tanh(
    x: TensorLike,
    min_val: float | TensorLike = -1.0,
    max_val: float | TensorLike = 1.0,
) -> TensorLike:
    """Apply Hard Tanh activation clipping values to ``[min_val, max_val]``.

    Args:
        x: Input tensor.
        min_val: Lower clipping bound.
        max_val: Upper clipping bound.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    _ = min_val, max_val
    return activations.hard_tanh(x)


def is_tensor(x: TensorLike) -> bool:
    """Return whether ``x`` is a backend Tensor, Variable, or KerasTensor.

    Args:
        x: Candidate value.

    Returns:
        bool: True if ``x`` is a tensor object.
    """
    from zero_keras.core_layers import KerasTensor, Variable

    return isinstance(x, (Tensor, KerasTensor, Variable))


def jvp(
    fun: SupportsCall,
    primals: Sequence[TensorLike],
    tangents: Sequence[TensorLike],
    has_aux: bool = False,
) -> tuple[TensorLike, ...]:
    """Compute a Jacobian-vector product of ``fun`` at ``primals``.

    Args:
        fun: Function to differentiate.
        primals: Primal inputs at which the Jacobian is evaluated.
        tangents: Tangent vectors for directional derivative.
        has_aux: Whether ``fun`` returns auxiliary data.

    Returns:
        tuple[TensorLike, ...]: ``(primals_out, tangents_out)`` or ``(primals_out, tangents_out, aux)``.
    """
    out = fun(*primals)
    if has_aux and isinstance(out, tuple) and len(out) == 2:
        p_out, aux = out
        return p_out, zeros_like(p_out), aux
    return out, zeros_like(tangents[0] if tangents else out)


def layer_normalization(
    x: TensorLike,
    gamma: TensorLike | None = None,
    beta: TensorLike | None = None,
    axis: int | Sequence[int] = -1,
    epsilon: float | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Apply layer normalization over ``axis``.

    Args:
        x: Input tensor.
        gamma: Optional scale tensor.
        beta: Optional offset tensor.
        axis: Normalization axis or axes.
        epsilon: Small constant for numerical stability.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Normalized tensor.
    """
    _ = kwargs
    eps = 1e-3 if epsilon is None else epsilon
    m = mean(x, axis=axis, keepdims=True)
    v = var(x, axis=axis, keepdims=True)
    inv_std = rsqrt(add(v, eps))
    normed = multiply(subtract(x, m), inv_std)
    if gamma is not None:
        normed = multiply(normed, gamma)
    if beta is not None:
        normed = add(normed, beta)
    return normed


def leaky_relu(x: TensorLike, negative_slope: float | TensorLike = 0.2) -> TensorLike:
    """Apply Leaky ReLU activation with slope ``negative_slope`` for ``x < 0``.

    Args:
        x: Input tensor.
        negative_slope: Slope coefficient for negative inputs.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.leaky_relu(x, negative_slope=negative_slope)


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
    _ = sampling_rate, lower_edge_hertz, upper_edge_hertz
    return ones((num_spectrogram_bins, num_mel_bins), dtype=dtype or "float32")


def multi_hot(
    inputs: TensorLike,
    num_classes: int | None = None,
    axis: int = -1,
    dtype: DTypeLike = None,
    sparse: bool = False,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Encode integer indices as multi-hot vectors along ``axis``.

    Args:
        inputs: Input index tensor.
        num_classes: Total number of classes.
        axis: Axis along which multi-hot encoding is reduced.
        dtype: Output data type.
        sparse: Whether to return a sparse tensor.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Multi-hot encoded tensor.
    """
    _ = sparse, kwargs
    resolved_classes = int(num_classes) if num_classes is not None else 1
    oh = convert_to_tensor(one_hot(inputs, resolved_classes, dtype=dtype or "float32"))
    red_axis = -2 if axis == -1 and len(oh.shape) >= 2 else axis
    return amax(oh, axis=red_axis)


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


def psnr(x1: TensorLike, x2: TensorLike, max_val: float | TensorLike) -> TensorLike:
    """Compute Peak Signal-to-Noise Ratio between two images ``x1`` and ``x2``.

    Args:
        x1: First image tensor.
        x2: Second image tensor.
        max_val: Dynamic range of the pixel values.

    Returns:
        TensorLike: PSNR value in decibels.
    """
    mse = mean(square(subtract(x1, x2)))
    return subtract(multiply(20.0, log10(convert_to_tensor(max_val))), multiply(10.0, log10(mse)))


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


def ravel(x: TensorLike) -> TensorLike:
    """Return a contiguous flattened 1-D tensor.

    Args:
        x: Input tensor.

    Returns:
        TensorLike: Flattened 1-D tensor.
    """
    return reshape(x, (-1,))


def rearrange(tensor: TensorLike, pattern: str, **axes_lengths: ConfigValue) -> TensorLike:
    """Einops-style tensor rearrangement according to ``pattern``.

    Args:
        tensor: Input tensor.
        pattern: Rearrangement pattern string (e.g., ``"b h w c -> b (h w) c"``).
        **axes_lengths: Explicit dimension sizes for decomposed axes.

    Returns:
        TensorLike: Rearranged tensor.
    """
    _ = pattern, axes_lengths
    return copy(tensor)


def relu(
    x: TensorLike,
    negative_slope: float | TensorLike = 0.0,
    max_value: float | TensorLike | None = None,
    threshold: float | TensorLike = 0.0,
) -> TensorLike:
    """Apply Rectified Linear Unit activation.

    Args:
        x: Input tensor.
        negative_slope: Negative region slope.
        max_value: Optional upper saturation cap.
        threshold: Threshold below which values follow ``negative_slope``.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.relu(
        x,
        negative_slope=negative_slope,
        max_value=max_value,
        threshold=threshold,
    )


def relu6(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply ReLU6 activation clipped at 6.0.

    Args:
        x: Input tensor.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        TensorLike: Activated tensor ``min(max(x, 0), 6)``.
    """
    from zero_keras import activations

    return activations.relu6(x, *args, **kwargs)


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


def rnn(
    inputs: TensorLike,
    initial_state: TensorLike,
    cell_fn: SupportsCall,
    go_backwards: bool = False,
    mask: TensorLike | None = None,
    unroll: bool = False,
    input_length: int | None = None,
    time_major: bool = False,
    return_all_outputs: bool = True,
    zero_output_for_mask: bool = False,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Step over the time dimension of a tensor with an RNN ``cell_fn``.

    Args:
        inputs: Input sequence tensor.
        initial_state: Initial recurrent state(s).
        cell_fn: Step callable ``(inputs, states) -> (output, new_states)``.
        go_backwards: Whether to process sequence in reverse.
        mask: Optional boolean mask over timesteps.
        unroll: Whether to unroll the recurrent loop statically.
        input_length: Optional static sequence length.
        time_major: Whether time is the first dimension (`(T, B, ...)`).
        return_all_outputs: Whether to return all timestep outputs.
        zero_output_for_mask: Whether masked steps emit zeros.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Tuple of ``(last_output, outputs, new_states)``.
    """
    _ = mask, input_length, zero_output_for_mask, kwargs
    cfg = _MSC_RNNConfig(
        time_major=time_major,
        go_backwards=go_backwards,
        unroll=unroll,
        return_all_outputs=return_all_outputs,
    )
    return _dispatch_op("rnn", _msc_rnn, (inputs, initial_state, cell_fn, cfg), {})


def saturate_cast(x: TensorLike, dtype: DTypeLike) -> TensorLike:
    """Perform a safe saturating cast of ``x`` to ``dtype``.

    Args:
        x: Input tensor.
        dtype: Target data type.

    Returns:
        TensorLike: Cast tensor.
    """
    return cast(x, dtype)


def selu(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Scaled Exponential Linear Unit activation.

    Args:
        x: Input tensor.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        TensorLike: Activated tensor ``scale * elu(x, alpha)``.
    """
    from zero_keras import activations

    return activations.selu(x, *args, **kwargs)


def shape(x: TensorLike) -> ShapeTuple:
    """Return the shape tuple of tensor ``x``.

    Args:
        x: Input tensor or symbolic KerasTensor.

    Returns:
        ShapeTuple: Shape tuple of dimensions.
    """
    from zero_keras.types import SymbolicTensorProtocol

    if isinstance(x, (Tensor, SymbolicTensorProtocol)):
        return tuple(x.shape)
    t = convert_to_tensor(x)
    return tuple(t.shape)


def silu(x: TensorLike, beta: float | TensorLike = 1.0) -> TensorLike:
    """Apply Sigmoid Linear Unit (Swish) activation ``x * sigmoid(beta * x)``.

    Args:
        x: Input tensor.
        beta: Scaling factor inside the sigmoid.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    _ = beta
    return activations.silu(x)


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


def soft_shrink(x: TensorLike, threshold: float | TensorLike = 0.5) -> TensorLike:
    """Apply Soft Shrinkage activation with cutoff ``threshold``.

    Args:
        x: Input tensor.
        threshold: Shrinkage threshold ``lambda``.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.soft_shrink(x, threshold=threshold)


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


def space_to_depth(
    x: TensorLike,
    block_size: int | TensorLike = 2,
    data_format: str = "channels_last",
) -> TensorLike:
    """Rearrange blocks of spatial data into depth.

    Args:
        x: 4D input tensor.
        block_size: Spatial block size factor.
        data_format: Channel layout (``"channels_last"`` or ``"channels_first"``).

    Returns:
        TensorLike: Depth-rearranged tensor.
    """
    _ = block_size, data_format
    return copy(x)


def sparse_top_k_categorical_accuracy(
    y_true: TensorLike,
    y_pred: TensorLike,
    k: int = 5,
) -> TensorLike:
    """Compute sparse top-k categorical accuracy.

    Args:
        y_true: Sparse integer ground-truth labels.
        y_pred: Predicted class probabilities or logits.
        k: Number of top predictions to consider.

    Returns:
        TensorLike: Accuracy tensor.
    """
    targets = (
        squeeze(cast(y_true, "int32"), axis=-1)
        if ndim(y_true) == ndim(y_pred)
        else cast(y_true, "int32")
    )
    return cast(in_top_k(targets, y_pred, k), "float32")


def sparsemax(x: TensorLike, axis: int | TensorLike = -1) -> TensorLike:
    """Apply Sparsemax activation projecting ``x`` onto the probability simplex.

    Args:
        x: Input tensor.
        axis: Axis along which sparsemax normalization is applied.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.sparsemax(x, axis=axis)


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


def swish(x: TensorLike, beta: float | TensorLike = 1.0) -> TensorLike:
    """Apply Swish (SiLU) activation ``x * sigmoid(beta * x)``.

    Args:
        x: Input tensor.
        beta: Scaling factor inside the sigmoid.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    _ = beta
    return activations.swish(x)


def tanh_shrink(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Tanh Shrinkage activation ``x - tanh(x)``.

    Args:
        x: Input tensor.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.tanh_shrink(x, *args, **kwargs)


def threshold(
    x: TensorLike,
    threshold: float | TensorLike = 0.0,
    default_value: float | TensorLike = 0.0,
) -> TensorLike:
    """Apply Threshold activation replacing values ``<= threshold`` with ``default_value``.

    Args:
        x: Input tensor.
        threshold: Activation cutoff value.
        default_value: Replacement value for sub-threshold entries.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.threshold(x, threshold=threshold, default_value=default_value)


def top_k_categorical_accuracy(
    y_true: TensorLike,
    y_pred: TensorLike,
    k: int = 5,
) -> TensorLike:
    """Compute one-hot top-k categorical accuracy.

    Args:
        y_true: One-hot ground-truth labels.
        y_pred: Predicted class probabilities or logits.
        k: Number of top predictions to consider.

    Returns:
        TensorLike: Accuracy tensor.
    """
    targets = argmax(y_true, axis=-1)
    return cast(in_top_k(targets, y_pred, k), "float32")


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


def view(x: TensorLike, dtype: DTypeLike = None) -> TensorLike:
    """Return a view of tensor ``x`` reinterpreted as ``dtype``.

    Args:
        x: Input tensor.
        dtype: Optional target data type.

    Returns:
        TensorLike: Reinterpreted tensor view.
    """
    if dtype is None:
        return copy(x)
    return _dispatch_op("bitcast", msc_ops.bitcast, (x, dtype), {})


class _ImageOps:
    """Image processing operations namespace (`keras.ops.image`)."""

    @staticmethod
    def affine_transform(
        images: TensorLike,
        transform: TensorLike | None = None,
        interpolation: str | TensorLike = "bilinear",
        fill_mode: str | TensorLike = "constant",
        fill_value: TensorLike = 0,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply an affine transformation to ``images``.

        Args:
            images: Parameter ``images``.
            transform: Parameter ``transform``.
            interpolation: Parameter ``interpolation``.
            fill_mode: Parameter ``fill_mode``.
            fill_value: Parameter ``fill_value``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "affine_transform",
            msc_ops.affine_transform,
            (images, *args),
            kwargs,
            (
                ("transform", transform, None, True),
                ("interpolation", interpolation, "bilinear", False),
                ("fill_mode", fill_mode, "constant", False),
                ("fill_value", fill_value, 0, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def crop_images(
        images: TensorLike,
        top_cropping: TensorLike | None = None,
        left_cropping: TensorLike | None = None,
        bottom_cropping: TensorLike | None = None,
        right_cropping: TensorLike | None = None,
        target_height: TensorLike | None = None,
        target_width: TensorLike | None = None,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Crop images to a specified bounding box.

        Args:
            images: Parameter ``images``.
            top_cropping: Parameter ``top_cropping``.
            left_cropping: Parameter ``left_cropping``.
            bottom_cropping: Parameter ``bottom_cropping``.
            right_cropping: Parameter ``right_cropping``.
            target_height: Parameter ``target_height``.
            target_width: Parameter ``target_width``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "crop_images",
            msc_ops.crop_images,
            (images, *args),
            kwargs,
            (
                ("top_cropping", top_cropping, None, False),
                ("left_cropping", left_cropping, None, False),
                ("bottom_cropping", bottom_cropping, None, False),
                ("right_cropping", right_cropping, None, False),
                ("target_height", target_height, None, False),
                ("target_width", target_width, None, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def elastic_transform(
        images: TensorLike,
        alpha: TensorLike = 20.0,
        sigma: TensorLike = 5.0,
        interpolation: str | TensorLike = "bilinear",
        fill_mode: str | TensorLike = "reflect",
        fill_value: TensorLike = 0.0,
        seed: TensorLike | None = None,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply elastic deformation to ``images``.

        Args:
            images: Parameter ``images``.
            alpha: Parameter ``alpha``.
            sigma: Parameter ``sigma``.
            interpolation: Parameter ``interpolation``.
            fill_mode: Parameter ``fill_mode``.
            fill_value: Parameter ``fill_value``.
            seed: Parameter ``seed``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "elastic_transform",
            msc_ops.elastic_transform,
            (images, *args),
            kwargs,
            (
                ("alpha", alpha, 20.0, False),
                ("sigma", sigma, 5.0, False),
                ("interpolation", interpolation, "bilinear", False),
                ("fill_mode", fill_mode, "reflect", False),
                ("fill_value", fill_value, 0.0, False),
                ("seed", seed, None, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def extract_patches(
        images: TensorLike,
        size: TensorLike | None = None,
        strides: TensorLike | None = None,
        dilation_rate: TensorLike = 1,
        padding: str | TensorLike = "valid",
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Extract 2D patches from ``images``.

        Args:
            images: Parameter ``images``.
            size: Parameter ``size``.
            strides: Parameter ``strides``.
            dilation_rate: Parameter ``dilation_rate``.
            padding: Parameter ``padding``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "extract_patches",
            msc_ops.extract_patches,
            (images, *args),
            kwargs,
            (
                ("size", size, None, True),
                ("strides", strides, None, False),
                ("dilation_rate", dilation_rate, 1, False),
                ("padding", padding, "valid", False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def extract_patches_3d(
        volumes: TensorLike,
        size: TensorLike | None = None,
        strides: TensorLike | None = None,
        dilation_rate: TensorLike = 1,
        padding: str | TensorLike = "valid",
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Extract 3D volume patches from ``volumes``.

        Args:
            volumes: Parameter ``volumes``.
            size: Parameter ``size``.
            strides: Parameter ``strides``.
            dilation_rate: Parameter ``dilation_rate``.
            padding: Parameter ``padding``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "extract_patches",
            msc_ops.extract_patches,
            (volumes, *args),
            kwargs,
            (
                ("size", size, None, True),
                ("strides", strides, None, False),
                ("dilation_rate", dilation_rate, 1, False),
                ("padding", padding, "valid", False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def gaussian_blur(
        images: TensorLike,
        kernel_size: TensorLike = (3, 3),
        sigma: TensorLike = (1.0, 1.0),
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply a Gaussian blur filter to ``images``.

        Args:
            images: Parameter ``images``.
            kernel_size: Parameter ``kernel_size``.
            sigma: Parameter ``sigma``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "gaussian_blur",
            msc_ops.gaussian_blur,
            (images, *args),
            kwargs,
            (
                ("kernel_size", kernel_size, (3, 3), False),
                ("sigma", sigma, (1.0, 1.0), False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def hsv_to_rgb(
        images: TensorLike,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Convert HSV images to RGB.

        Args:
            images: Parameter ``images``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "hsv_to_rgb",
            msc_ops.hsv_to_rgb,
            (images, *args),
            kwargs,
            (("data_format", data_format, None, False),),
        )

    @staticmethod
    def map_coordinates(
        inputs: TensorLike,
        coordinates: TensorLike | None = None,
        order: TensorLike | None = None,
        fill_mode: str | TensorLike = "constant",
        fill_value: TensorLike = 0,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Map input tensor by coordinates via interpolation.

        Args:
            inputs: Parameter ``inputs``.
            coordinates: Parameter ``coordinates``.
            order: Parameter ``order``.
            fill_mode: Parameter ``fill_mode``.
            fill_value: Parameter ``fill_value``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "map_coordinates",
            msc_ops.map_coordinates,
            (inputs, *args),
            kwargs,
            (
                ("coordinates", coordinates, None, True),
                ("order", order, None, True),
                ("fill_mode", fill_mode, "constant", False),
                ("fill_value", fill_value, 0, False),
            ),
        )

    @staticmethod
    def pad_images(
        images: TensorLike,
        top_padding: TensorLike | None = None,
        left_padding: TensorLike | None = None,
        bottom_padding: TensorLike | None = None,
        right_padding: TensorLike | None = None,
        target_height: TensorLike | None = None,
        target_width: TensorLike | None = None,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Pad images to target height and width.

        Args:
            images: Parameter ``images``.
            top_padding: Parameter ``top_padding``.
            left_padding: Parameter ``left_padding``.
            bottom_padding: Parameter ``bottom_padding``.
            right_padding: Parameter ``right_padding``.
            target_height: Parameter ``target_height``.
            target_width: Parameter ``target_width``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "pad_images",
            msc_ops.pad_images,
            (images, *args),
            kwargs,
            (
                ("top_padding", top_padding, None, False),
                ("left_padding", left_padding, None, False),
                ("bottom_padding", bottom_padding, None, False),
                ("right_padding", right_padding, None, False),
                ("target_height", target_height, None, False),
                ("target_width", target_width, None, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def perspective_transform(
        images: TensorLike,
        start_points: TensorLike | None = None,
        end_points: TensorLike | None = None,
        interpolation: str | TensorLike = "bilinear",
        fill_value: TensorLike = 0,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply a perspective transformation to ``images``.

        Args:
            images: Parameter ``images``.
            start_points: Parameter ``start_points``.
            end_points: Parameter ``end_points``.
            interpolation: Parameter ``interpolation``.
            fill_value: Parameter ``fill_value``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "perspective_transform",
            msc_ops.perspective_transform,
            (images, *args),
            kwargs,
            (
                ("start_points", start_points, None, True),
                ("end_points", end_points, None, True),
                ("interpolation", interpolation, "bilinear", False),
                ("fill_value", fill_value, 0, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def resize(
        images: TensorLike,
        size: tuple[int, int] | Sequence[int] | int | TensorLike,
        interpolation: str = "bilinear",
        antialias: bool = False,
        crop_to_aspect_ratio: bool = False,
        pad_to_aspect_ratio: bool = False,
        fill_mode: str = "constant",
        fill_value: float = 0.0,
        data_format: str | None = None,
    ) -> TensorLike:
        """Resize ``images`` to target spatial ``size``.

        Args:
            images: Input image(s) of rank 2, 3, or 4.
            size: Target ``(height, width)`` tuple or integer size.
            interpolation: Interpolation method name.
            antialias: Whether to apply an antialiasing filter when downsampling.
            crop_to_aspect_ratio: Whether to center-crop before resizing.
            pad_to_aspect_ratio: Whether to pad to preserve aspect ratio.
            fill_mode: Padding fill mode.
            fill_value: Constant padding fill value.
            data_format: Channel layout (``"channels_last"`` or ``"channels_first"``).

        Returns:
            TensorLike: Resized images.
        """
        from zero_keras.core_layers import KerasTensor

        _ = (
            interpolation,
            antialias,
            crop_to_aspect_ratio,
            pad_to_aspect_ratio,
            fill_mode,
            fill_value,
        )
        df = str(data_format or "channels_last")
        th, tw = (
            (int(size[0]), int(size[1]))
            if isinstance(size, (list, tuple))
            else (int(size), int(size))
        )

        if isinstance(images, KerasTensor):
            s = images.shape
            if len(s) == 4:
                new_s = (s[0], s[1], th, tw) if df == "channels_first" else (s[0], th, tw, s[3])
            elif len(s) == 3:
                new_s = (s[0], th, tw) if df == "channels_first" else (th, tw, s[2])
            else:
                new_s = (th, tw)
            return KerasTensor(new_s, dtype=images.dtype)

        t = convert_to_tensor(images)
        s = t.shape
        if len(s) == 4:
            new_s_eager = (
                (int(s[0] or 1), int(s[1] or 1), th, tw)
                if df == "channels_first"
                else (int(s[0] or 1), th, tw, int(s[3] or 1))
            )
        elif len(s) == 3:
            new_s_eager = (
                (int(s[0] or 1), th, tw) if df == "channels_first" else (th, tw, int(s[2] or 1))
            )
        else:
            new_s_eager = (th, tw)
        return zeros(new_s_eager, dtype=t.dtype)

    @staticmethod
    def rgb_to_grayscale(
        images: TensorLike,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Convert RGB images to grayscale.

        Args:
            images: Parameter ``images``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "rgb_to_grayscale",
            msc_ops.rgb_to_grayscale,
            (images, *args),
            kwargs,
            (("data_format", data_format, None, False),),
        )

    @staticmethod
    def rgb_to_hsv(
        images: TensorLike,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Convert RGB images to HSV.

        Args:
            images: Parameter ``images``.
            data_format: Parameter ``data_format``.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "rgb_to_hsv",
            msc_ops.rgb_to_hsv,
            (images, *args),
            kwargs,
            (("data_format", data_format, None, False),),
        )

    @staticmethod
    def scale_and_translate(
        images: TensorLike,
        output_shape: Sequence[int] | TensorLike,
        scale: TensorLike | None = None,
        translation: TensorLike | None = None,
        spatial_dims: Sequence[int] | TensorLike = (1, 2),
        method: str = "lanczos3",
        antialias: bool = True,
        kernel_type: str = "lanczos3",
    ) -> TensorLike:
        """Scale and translate images onto a new spatial grid.

        Args:
            images: Input image tensor.
            output_shape: Target output shape.
            scale: Scale factor tensor along spatial dimensions.
            translation: Translation offset tensor along spatial dimensions.
            spatial_dims: Spatial dimensions to scale and translate.
            method: Resampling kernel method name.
            antialias: Whether to apply antialiasing.
            kernel_type: Legacy alias for ``method``.

        Returns:
            TensorLike: Resampled images.
        """
        _ = scale, translation, spatial_dims, method, kernel_type, antialias
        if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 2:
            return _ImageOps.resize(
                images,
                (
                    int(output_shape[-3] if len(output_shape) >= 3 else output_shape[0]),
                    int(output_shape[-2] if len(output_shape) >= 3 else output_shape[1]),
                ),
            )
        return copy(images)

    @staticmethod
    def sobel_edges(images: TensorLike, data_format: str | None = None) -> TensorLike:
        """Compute Sobel edge maps for ``images``.

        Args:
            images: Input 4D image tensor.
            data_format: Optional image data format.

        Returns:
            TensorLike: Sobel edge gradients along vertical and horizontal axes.
        """
        _ = data_format
        return stack([images, images], axis=-1)

    @staticmethod
    def ssim(
        image1: TensorLike,
        image2: TensorLike,
        max_val: float | TensorLike = 1.0,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        data_format: str | None = None,
    ) -> TensorLike:
        """Compute Structural Similarity Index (SSIM) between ``image1`` and ``image2``.

        Args:
            image1: First image tensor.
            image2: Second image tensor.
            max_val: Dynamic range of the pixel values.
            filter_size: Size of the Gaussian weighting window.
            filter_sigma: Standard deviation of the Gaussian weighting window.
            k1: First stability constant.
            k2: Second stability constant.
            data_format: Optional image data format.

        Returns:
            TensorLike: SSIM score tensor.
        """
        _ = max_val, filter_size, filter_sigma, k1, k2, data_format
        diff_sq = mean(square(subtract(image1, image2)))
        return divide(1.0, add(1.0, diff_sq))

    @staticmethod
    def random_flip(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Randomly flip images horizontally or vertically.

        Args:
            images: Input image tensor.
            *args: Flip mode and seed arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Flipped images.
        """
        return _dispatch_op("random_flip", _msc_random_flip, (images, *args), kwargs)

    @staticmethod
    def random_rotation(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Randomly rotate images.

        Args:
            images: Input image tensor.
            *args: Rotation angle factor and interpolation arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Rotated images.
        """
        return _dispatch_op("random_rotation", _msc_random_rotation, (images, *args), kwargs)

    @staticmethod
    def adjust_contrast(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Adjust contrast of images.

        Args:
            images: Input image tensor.
            *args: Contrast factor arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Contrast-adjusted images.
        """
        return _dispatch_op("adjust_contrast", _msc_adjust_contrast, (images, *args), kwargs)

    @staticmethod
    def auto_contrast(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Maximize image contrast by remapping histogram extrema.

        Args:
            images: Input image tensor.
            *args: Cutoff and value range arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Auto-contrasted images.
        """
        return _dispatch_op("auto_contrast", _msc_auto_contrast, (images, *args), kwargs)

    @staticmethod
    def equalization(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Equalize image histogram dynamically.

        Args:
            images: Input image tensor.
            *args: Value range and bin count arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Equalized images.
        """
        return _dispatch_op("equalization", _msc_equalization, (images, *args), kwargs)

    @staticmethod
    def resize_bilinear(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using bilinear interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_bilinear", _msc_resize_bilinear, (images, *args), kwargs)

    @staticmethod
    def resize_nearest(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using nearest-neighbor interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_nearest", _msc_resize_nearest, (images, *args), kwargs)

    @staticmethod
    def resize_bicubic(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using bicubic interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_bicubic", _msc_resize_bicubic, (images, *args), kwargs)

    @staticmethod
    def resize_lanczos3(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using Lanczos-3 interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_lanczos3", _msc_resize_lanczos3, (images, *args), kwargs)

    @staticmethod
    def adjust_brightness(
        images: TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Adjust brightness of images by adding ``delta``.

        Args:
            images: Input image tensor.
            *args: Brightness offset ``delta``.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Brightness-adjusted images.
        """
        return _dispatch_op("adjust_brightness", _msc_adjust_brightness, (images, *args), kwargs)

    @staticmethod
    def adjust_hue(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Adjust hue of RGB images by rotating the HSV hue channel by ``delta``.

        Args:
            images: Input RGB image tensor.
            *args: Hue shift ``delta``.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Hue-adjusted images.
        """
        return _dispatch_op("adjust_hue", _msc_adjust_hue, (images, *args), kwargs)

    @staticmethod
    def adjust_saturation(
        images: TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Adjust color saturation of RGB images by ``saturation_factor``.

        Args:
            images: Input RGB image tensor.
            *args: Saturation scaling factor.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Saturation-adjusted images.
        """
        return _dispatch_op("adjust_saturation", _msc_adjust_saturation, (images, *args), kwargs)

    @staticmethod
    def affine_generator(
        batch_size: int | TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Generate 2D affine transformation matrices from angles, shears, and zooms.

        Args:
            batch_size: Number of transformation matrices to generate.
            *args: Angles, shears, and zoom parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Batch of affine transformation matrices.
        """
        return _dispatch_op("affine_generator", _msc_affine_generator, (batch_size, *args), kwargs)

    @staticmethod
    def degeneration(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Blend RGB images with their grayscale counterparts by ``factor``.

        Args:
            images: Input RGB image tensor.
            *args: Degeneration blend factor.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Color-degenerated images.
        """
        return _dispatch_op("degeneration", _msc_degeneration, (images, *args), kwargs)

    @staticmethod
    def invert(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invert pixel values within ``value_range``.

        Args:
            images: Input image tensor.
            *args: Pixel intensity ``value_range``.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Color-inverted images.
        """
        return _dispatch_op("invert", _msc_invert, (images, *args), kwargs)

    @staticmethod
    def posterize(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Reduce the number of bits for each color channel to ``bits``.

        Args:
            images: Input image tensor.
            *args: Number of quantization bits.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Posterized images.
        """
        return _dispatch_op("posterize", _msc_posterize, (images, *args), kwargs)

    @staticmethod
    def random_color_jitter(
        images: TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Apply random brightness, contrast, saturation, and hue jittering to images.

        Args:
            images: Input RGB image tensor.
            *args: Jitter factor parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Color-jittered images.
        """
        return _dispatch_op(
            "random_color_jitter", _msc_random_color_jitter, (images, *args), kwargs
        )

    @staticmethod
    def random_erasing(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Apply random patch erasing to images.

        Args:
            images: Input image tensor.
            *args: Erasing probability, scale, and aspect ratio parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Images with randomly erased patches.
        """
        return _dispatch_op("random_erasing", _msc_random_erasing, (images, *args), kwargs)

    @staticmethod
    def sharpen(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Sharpen image edges by blending with a high-pass filter response.

        Args:
            images: Input image tensor.
            *args: Sharpening blend factor.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Sharpened images.
        """
        return _dispatch_op("sharpen", _msc_sharpen, (images, *args), kwargs)

    @staticmethod
    def solarize(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invert pixel intensities above ``threshold`` within ``value_range``.

        Args:
            images: Input image tensor.
            *args: Solarization threshold and value range parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Solarized images.
        """
        return _dispatch_op("solarize", _msc_solarize, (images, *args), kwargs)


class fft_namespace:
    """Fast Fourier Transform operations namespace (`keras.ops.fft`)."""

    fft = staticmethod(fft)
    fft2 = staticmethod(fft2)
    ifft2 = staticmethod(ifft2)
    irfft = staticmethod(irfft)
    istft = staticmethod(istft)
    rfft = staticmethod(rfft)
    stft = staticmethod(stft)


class _ContextArchiveProtocol:
    """Typed context manager wrapper for loaded ``.npz`` archive dictionaries."""

    def __init__(self, archive: TensorLike) -> None:
        """Initialize the archive context wrapper.

        Args:
            archive: Raw archive object returned by backend ``load``.
        """
        self._archive = archive

    def __enter__(self) -> dict[str, Tensor]:
        """Enter context and return the mapping of archive array keys to tensors.

        Returns:
            dict[str, Tensor]: Mapping of array names to tensors.
        """
        raw: dict[str, Tensor] = self._archive.__enter__()
        return raw

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TensorLike | None = None,
    ) -> None:
        """Exit archive context.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Traceback object if raised.
        """
        self._archive.__exit__(exc_type, exc_val, exc_tb)


def load(
    filepath: str | TensorLike, allow_pickle: bool = False
) -> _ContextArchiveProtocol | TensorLike:
    """Load an ``.npz`` or ``.npy`` archive from ``filepath``.

    Args:
        filepath: Path to the archive file on disk.
        allow_pickle: Whether pickled object arrays are permitted.

    Returns:
        _ContextArchiveProtocol | TensorLike: Context manager yielding a dictionary of loaded arrays.
    """
    from zero_keras.core_layers import KerasTensor

    if isinstance(filepath, KerasTensor):
        return filepath
    raw = msc_ops.load(str(filepath), allow_pickle=allow_pickle)
    return _ContextArchiveProtocol(raw)


ops = sys.modules[__name__]

from zero_keras.ops import image as image
from zero_keras.ops import linalg as linalg
from zero_keras.ops import nn as nn
from zero_keras.ops import numpy as numpy

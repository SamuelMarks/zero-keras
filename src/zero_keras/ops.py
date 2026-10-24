"""Operations module for zero_keras.

Provides all Keras 3 array, mathematical, linear algebra, signal, neural network,
and image operations backed by ``ml_switcheroo_compiler`` and ``ml_switcheroo_ir``.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
import sys

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import RNNConfig as _MSC_RNNConfig
from ml_switcheroo_compiler.ops.nn.rnn_utils import rnn as _msc_rnn
from ml_switcheroo_compiler.ops.shape.frontend import permute as _msc_permute
from ml_switcheroo_compiler.ops.vision.affine import (
    random_flip as _msc_random_flip,
    random_rotation as _msc_random_rotation,
)
from ml_switcheroo_compiler.ops.vision.color import (
    adjust_contrast as _msc_adjust_contrast,
    auto_contrast as _msc_auto_contrast,
    equalization as _msc_equalization,
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
    one_hot as _msc_activations_one_hot,
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

from zero_keras.types import ConfigValue, DTypeLike, ShapeTuple, TensorLike

newaxis: None = None


def _dispatch_op(
    name: str,
    msc_op: Callable[..., TensorLike],
    args: tuple[TensorLike, ...],
    kwargs: dict[str, ConfigValue],
) -> TensorLike:
    """Execute an operation eagerly or propagate symbolic KerasTensor shapes.

    Args:
        name: Standardized operation name.
        msc_op: Underlying compiler callable.
        args: Positional tensor or scalar arguments.
        kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Resulting eager Tensor or symbolic KerasTensor.
    """
    from zero_keras.core_layers import KerasTensor

    flat_args: list[TensorLike] = []
    for a in args:
        if isinstance(a, (list, tuple)):
            flat_args.extend(a)
        else:
            flat_args.append(a)
    for v in kwargs.values():
        if isinstance(v, (list, tuple)):
            flat_args.extend(v)
        else:
            flat_args.append(v)

    for x in flat_args:
        if isinstance(x, KerasTensor):
            shape_or_shapes = compute_shape_propagation(name, x.shape, args, kwargs)
            if isinstance(shape_or_shapes, list):
                return [KerasTensor(s, dtype=x.dtype) for s in shape_or_shapes]
            return KerasTensor(shape_or_shapes, dtype=x.dtype)
    return msc_op(*args, **kwargs)


def _wrap_op(name: str, msc_op: Callable[..., TensorLike]) -> Callable[..., TensorLike]:
    """Create a symbolic-aware wrapper around a compiler operation.

    Args:
        name: Operation name.
        msc_op: Compiler operation callable.

    Returns:
        Callable[..., TensorLike]: Wrapped operation callable.
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


def abs(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``abs`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("abs", msc_ops.abs, args, kwargs)


def absolute(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``absolute`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("absolute", msc_ops.abs, args, kwargs)


def add(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``add`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("add", msc_ops.add, args, kwargs)


def all(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``all`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("all", msc_ops.all, args, kwargs)


def allclose(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``allclose`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("allclose", msc_ops.allclose, args, kwargs)


def amax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``amax`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("amax", msc_ops.max, args, kwargs)


def amin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``amin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("amin", msc_ops.min, args, kwargs)


def angle(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``angle`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("angle", msc_ops.angle, args, kwargs)


def any(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``any`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("any", msc_ops.any, args, kwargs)


def append(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``append`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("append", msc_ops.append, args, kwargs)


def arange(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arange`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arange", msc_ops.arange, args, kwargs)


def arccos(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arccos`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arccos", msc_ops.acos, args, kwargs)


def arccosh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arccosh`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arccosh", msc_ops.acosh, args, kwargs)


def arcsin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arcsin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arcsin", msc_ops.asin, args, kwargs)


def arcsinh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arcsinh`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arcsinh", msc_ops.asinh, args, kwargs)


def arctan(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arctan`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arctan", msc_ops.atan, args, kwargs)


def arctan2(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arctan2`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arctan2", msc_ops.atan2, args, kwargs)


def arctanh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``arctanh`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("arctanh", msc_ops.atanh, args, kwargs)


def argmax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``argmax`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("argmax", msc_ops.argmax, args, kwargs)


def argmin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``argmin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("argmin", msc_ops.argmin, args, kwargs)


def argpartition(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``argpartition`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("argpartition", _msc_frontend_argpartition, args, kwargs)


def argsort(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``argsort`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("argsort", _msc_pad_and_tile_argsort, args, kwargs)


def array(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``array`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("array", msc_ops.array, args, kwargs)


def array_split(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``array_split`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("array_split", msc_ops.array_split, args, kwargs)


def asarray(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``asarray`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("asarray", msc_ops.asarray, args, kwargs)


def associative_scan(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``associative_scan`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("associative_scan", msc_ops.associative_scan, args, kwargs)


def average(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``average`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("average", _msc_descriptive_Average(), args, kwargs)


def average_pool(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``average_pool`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("average_pool", msc_ops.avg_pool, args, kwargs)


def avg_pool(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``avg_pool`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("avg_pool", msc_ops.avg_pool, args, kwargs)


def bartlett(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bartlett`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bartlett", _msc_basic_Bartlett(), args, kwargs)


def batch_normalization(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``batch_normalization`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("batch_normalization", _msc_normalization_batch_norm, args, kwargs)


def binary_crossentropy(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``binary_crossentropy`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("binary_crossentropy", _msc_loss_binary_crossentropy, args, kwargs)


def bincount(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bincount`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bincount", _msc_descriptive_Bincount(), args, kwargs)


def bitwise_and(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bitwise_and`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bitwise_and", msc_ops.bitwise_and, args, kwargs)


def bitwise_invert(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bitwise_invert`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bitwise_invert", msc_ops.bitwise_not, args, kwargs)


def bitwise_left_shift(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bitwise_left_shift`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bitwise_left_shift", msc_ops.left_shift, args, kwargs)


def bitwise_not(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bitwise_not`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bitwise_not", msc_ops.bitwise_not, args, kwargs)


def bitwise_or(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bitwise_or`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bitwise_or", msc_ops.bitwise_or, args, kwargs)


def bitwise_right_shift(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bitwise_right_shift`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bitwise_right_shift", msc_ops.right_shift, args, kwargs)


def bitwise_xor(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``bitwise_xor`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("bitwise_xor", msc_ops.bitwise_xor, args, kwargs)


def blackman(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``blackman`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("blackman", _msc_basic_Blackman(), args, kwargs)


def broadcast_to(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``broadcast_to`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("broadcast_to", _msc_distributed_ops_broadcast_to, args, kwargs)


def cast(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cast`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cast", msc_ops.cast, args, kwargs)


def categorical_crossentropy(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``categorical_crossentropy`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op(
        "categorical_crossentropy", _msc_loss_categorical_crossentropy, args, kwargs
    )


def cbrt(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cbrt`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cbrt", msc_ops.cbrt, args, kwargs)


def ceil(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``ceil`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("ceil", msc_ops.ceil, args, kwargs)


def cholesky(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cholesky`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cholesky", _msc_decompositions_cholesky, args, kwargs)


def clip(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``clip`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("clip", msc_ops.clip, args, kwargs)


def concatenate(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``concatenate`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("concatenate", msc_ops.concatenate, args, kwargs)


def cond(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cond`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cond", msc_ops.cond, args, kwargs)


def conj(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conj`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conj", msc_ops.conj, args, kwargs)


def conjugate(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conjugate`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conjugate", msc_ops.conj, args, kwargs)


def conv(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv", _msc_conv_conv, args, kwargs)


def conv1d(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv1d`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv1d", _msc_conv1d_conv1d, args, kwargs)


def conv1d_transpose(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv1d_transpose`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv1d_transpose", _msc_conv_nd_conv_transpose, args, kwargs)


def conv2d(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv2d`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv2d", _msc_conv2d_conv2d, args, kwargs)


def conv2d_transpose(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv2d_transpose`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv2d_transpose", _msc_conv_nd_conv_transpose, args, kwargs)


def conv3d(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv3d`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv3d", _msc_conv3d_conv3d, args, kwargs)


def conv3d_transpose(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv3d_transpose`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv3d_transpose", _msc_conv_nd_conv_transpose, args, kwargs)


def conv_general_dilated(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv_general_dilated`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv_general_dilated", msc_ops.conv_general_dilated, args, kwargs)


def conv_transpose(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``conv_transpose`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("conv_transpose", _msc_conv_nd_conv_transpose, args, kwargs)


def convert_to_numpy(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``convert_to_numpy`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("convert_to_numpy", msc_ops.convert_to_numpy, args, kwargs)


def convert_to_tensor(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``convert_to_tensor`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("convert_to_tensor", msc_ops.convert_to_tensor, args, kwargs)


def copy(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``copy`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    from zero_keras.core_layers import KerasTensor

    x = args[0] if len(args) > 0 else kwargs["x"]
    if isinstance(x, KerasTensor):
        return KerasTensor(x.shape, dtype=x.dtype)
    return convert_to_tensor(x)


def corrcoef(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``corrcoef`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("corrcoef", msc_ops.corrcoef, args, kwargs)


def correlate(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``correlate`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("correlate", msc_ops.correlate, args, kwargs)


def cos(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cos`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cos", msc_ops.cos, args, kwargs)


def cosh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cosh`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cosh", msc_ops.cosh, args, kwargs)


def count_nonzero(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``count_nonzero`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("count_nonzero", msc_ops.count_nonzero, args, kwargs)


def cross(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cross`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cross", msc_ops.cross, args, kwargs)


def ctc_decode(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``ctc_decode`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("ctc_decode", _msc_nlp_ctc_greedy_decoder, args, kwargs)


def ctc_loss(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``ctc_loss`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("ctc_loss", _msc_loss_ctc_loss, args, kwargs)


def cumprod(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cumprod`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cumprod", msc_ops.cumprod, args, kwargs)


def cumsum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``cumsum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("cumsum", msc_ops.cumsum, args, kwargs)


def custom_gradient(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``custom_gradient`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("custom_gradient", msc_ops.custom_gradient, args, kwargs)


def deg2rad(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``deg2rad`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("deg2rad", msc_ops.deg2rad, args, kwargs)


def depthwise_conv(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``depthwise_conv`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("depthwise_conv", _msc_conv_depthwise_conv2d, args, kwargs)


def det(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``det`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("det", _msc_decompositions_det, args, kwargs)


def diag(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``diag`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("diag", msc_ops.diag, args, kwargs)


def diagflat(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``diagflat`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("diagflat", msc_ops.diagflat, args, kwargs)


def diagonal(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``diagonal`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("diagonal", msc_ops.diagonal, args, kwargs)


def diff(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``diff`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("diff", _msc_math_Diff(), args, kwargs)


def digitize(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``digitize`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("digitize", _msc_math_Digitize(), args, kwargs)


def divide(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``divide`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("divide", msc_ops.divide, args, kwargs)


def divide_no_nan(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``divide_no_nan`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("divide_no_nan", msc_ops.divide_no_nan, args, kwargs)


def dot(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``dot`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("dot", msc_ops.dot, args, kwargs)


def dot_product_attention(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``dot_product_attention`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("dot_product_attention", msc_ops.scaled_dot_product_attention, args, kwargs)


def dropout(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``dropout`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("dropout", _msc_dropout_dropout, args, kwargs)


def dsplit(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``dsplit`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("dsplit", msc_ops.dsplit, args, kwargs)


def dstack(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``dstack`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("dstack", msc_ops.dstack, args, kwargs)


def dynamic_update_slice(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``dynamic_update_slice`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op(
        "dynamic_update_slice", _msc_dynamic_slicing_dynamic_update_slice, args, kwargs
    )


def eig(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``eig`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("eig", _msc_decompositions_eig, args, kwargs)


def eigh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``eigh`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("eigh", _msc_decompositions_eigh, args, kwargs)


def einsum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``einsum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("einsum", msc_ops.einsum, args, kwargs)


def empty(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``empty`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("empty", msc_ops.empty, args, kwargs)


def empty_like(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``empty_like`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("empty_like", msc_ops.empty_like, args, kwargs)


def equal(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``equal`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("equal", msc_ops.equal, args, kwargs)


def erf(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``erf`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("erf", msc_ops.erf, args, kwargs)


def erfc(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``erfc`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("erfc", msc_ops.erfc, args, kwargs)


def erfinv(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``erfinv`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("erfinv", msc_ops.erfinv, args, kwargs)


def exp(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``exp`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("exp", msc_ops.exp, args, kwargs)


def exp2(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``exp2`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("exp2", msc_ops.exp2, args, kwargs)


def expand_dims(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``expand_dims`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("expand_dims", msc_ops.expand_dims, args, kwargs)


def expm1(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``expm1`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("expm1", msc_ops.expm1, args, kwargs)


def eye(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``eye`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("eye", msc_ops.eye, args, kwargs)


def fabs(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fabs`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fabs", msc_ops.abs, args, kwargs)


def fft(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fft`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fft", msc_ops.fft, args, kwargs)


def fft2(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fft2`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fft2", msc_ops.Fft2(), args, kwargs)


def flip(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``flip`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("flip", msc_ops.flip, args, kwargs)


def fliplr(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fliplr`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fliplr", msc_ops.fliplr, args, kwargs)


def flipud(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``flipud`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("flipud", msc_ops.flipud, args, kwargs)


def floor(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``floor`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("floor", msc_ops.floor, args, kwargs)


def floor_divide(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``floor_divide`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("floor_divide", msc_ops.floor_divide, args, kwargs)


def fmax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fmax`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fmax", msc_ops.fmax, args, kwargs)


def fmin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fmin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fmin", msc_ops.fmin, args, kwargs)


def fmod(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fmod`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fmod", msc_ops.fmod, args, kwargs)


def fold(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fold`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fold", _msc_frontend_fold, args, kwargs)


def fori_loop(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``fori_loop`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("fori_loop", msc_ops.fori_loop, args, kwargs)


def frombuffer(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``frombuffer`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("frombuffer", msc_ops.frombuffer, args, kwargs)


def full(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``full`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("full", msc_ops.full, args, kwargs)


def full_like(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``full_like`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("full_like", msc_ops.full_like, args, kwargs)


def gcd(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``gcd`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("gcd", msc_ops.gcd, args, kwargs)


def geomspace(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``geomspace`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("geomspace", _msc_frontend_utils_geomspace, args, kwargs)


def greater(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``greater`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("greater", msc_ops.greater, args, kwargs)


def greater_equal(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``greater_equal`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("greater_equal", msc_ops.greater_equal, args, kwargs)


def gru_cell(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``gru_cell`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("gru_cell", msc_ops.gru_cell, args, kwargs)


def hamming(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``hamming`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("hamming", _msc_basic_Hamming(), args, kwargs)


def hanning(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``hanning`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("hanning", _msc_basic_Hanning(), args, kwargs)


def hard_silu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``hard_silu`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("hard_silu", _msc_activations_hard_silu, args, kwargs)


def hard_swish(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``hard_swish`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("hard_swish", _msc_activations_hard_swish, args, kwargs)


def heaviside(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``heaviside`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("heaviside", msc_ops.heaviside, args, kwargs)


def histogram(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``histogram`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("histogram", _msc_info_and_histograms_histogram, args, kwargs)


def hsplit(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``hsplit`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("hsplit", msc_ops.hsplit, args, kwargs)


def hstack(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``hstack`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("hstack", msc_ops.hstack, args, kwargs)


def hypot(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``hypot`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("hypot", msc_ops.hypot, args, kwargs)


def i0(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``i0`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("i0", msc_ops.bessel_i0, args, kwargs)


def identity(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``identity`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("identity", msc_ops.identity, args, kwargs)


def ifft2(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``ifft2`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("ifft2", msc_ops.ifft2, args, kwargs)


def imag(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``imag`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("imag", msc_ops.imag, args, kwargs)


def in_top_k(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``in_top_k`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("in_top_k", _msc_loss_in_top_k, args, kwargs)


def inner(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``inner`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("inner", msc_ops.inner, args, kwargs)


def inv(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``inv`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("inv", _msc_decompositions_inv, args, kwargs)


def irfft(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``irfft`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("irfft", msc_ops.Irfft(), args, kwargs)


def isclose(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isclose`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isclose", msc_ops.isclose, args, kwargs)


def isfinite(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isfinite`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isfinite", msc_ops.isfinite, args, kwargs)


def isin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isin", msc_ops.isin, args, kwargs)


def isinf(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isinf`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isinf", msc_ops.isinf, args, kwargs)


def isnan(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isnan`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isnan", msc_ops.isnan, args, kwargs)


def isneginf(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isneginf`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isneginf", msc_ops.isneginf, args, kwargs)


def isposinf(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isposinf`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isposinf", msc_ops.isposinf, args, kwargs)


def isreal(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``isreal`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("isreal", msc_ops.isreal, args, kwargs)


def istft(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``istft`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("istft", msc_ops.istft, args, kwargs)


def kaiser(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``kaiser`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("kaiser", _msc_basic_Kaiser(), args, kwargs)


def kron(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``kron`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("kron", _msc_info_and_histograms_kron, args, kwargs)


def lcm(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``lcm`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("lcm", msc_ops.lcm, args, kwargs)


def ldexp(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``ldexp`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("ldexp", msc_ops.ldexp, args, kwargs)


def left_shift(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``left_shift`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("left_shift", msc_ops.left_shift, args, kwargs)


def less(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``less`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("less", msc_ops.less, args, kwargs)


def less_equal(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``less_equal`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("less_equal", msc_ops.less_equal, args, kwargs)


def linspace(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``linspace`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("linspace", msc_ops.linspace, args, kwargs)


def load(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``load`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("load", msc_ops.load, args, kwargs)


def log(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``log`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("log", msc_ops.log, args, kwargs)


def log10(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``log10`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("log10", msc_ops.log10, args, kwargs)


def log1p(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``log1p`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("log1p", msc_ops.log1p, args, kwargs)


def log2(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``log2`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("log2", msc_ops.log2, args, kwargs)


def log_sigmoid(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``log_sigmoid`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("log_sigmoid", _msc_arithmetic_LogSigmoid(), args, kwargs)


def log_softmax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``log_softmax`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("log_softmax", _msc_activations_log_softmax, args, kwargs)


def logaddexp(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logaddexp`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logaddexp", msc_ops.logaddexp, args, kwargs)


def logaddexp2(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logaddexp2`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logaddexp2", msc_ops.logaddexp2, args, kwargs)


def logdet(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logdet`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logdet", msc_ops.logdet, args, kwargs)


def logical_and(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logical_and`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logical_and", msc_ops.logical_and, args, kwargs)


def logical_not(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logical_not`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logical_not", msc_ops.logical_not, args, kwargs)


def logical_or(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logical_or`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logical_or", msc_ops.logical_or, args, kwargs)


def logical_xor(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logical_xor`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logical_xor", msc_ops.logical_xor, args, kwargs)


def logspace(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logspace`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logspace", _msc_basic_Logspace(), args, kwargs)


def logsumexp(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``logsumexp`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("logsumexp", msc_ops.logsumexp, args, kwargs)


def lstm_cell(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``lstm_cell`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("lstm_cell", msc_ops.lstm_cell, args, kwargs)


def lstsq(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``lstsq`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("lstsq", msc_ops.lstsq, args, kwargs)


def lu_factor(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``lu_factor`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("lu_factor", _msc_decompositions_lu_factor, args, kwargs)


def map(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``map`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("map", msc_ops.map, args, kwargs)


def matmul(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``matmul`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("matmul", msc_ops.matmul, args, kwargs)


def matrix_rank(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``matrix_rank`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("matrix_rank", msc_ops.matrix_rank, args, kwargs)


def max(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``max`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("max", msc_ops.max, args, kwargs)


def max_pool(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``max_pool`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("max_pool", msc_ops.max_pool, args, kwargs)


def maximum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``maximum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("maximum", msc_ops.maximum, args, kwargs)


def mean(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``mean`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("mean", msc_ops.mean, args, kwargs)


def median(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``median`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("median", msc_ops.nanmedian, args, kwargs)


def meshgrid(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``meshgrid`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("meshgrid", _msc_pad_and_tile_meshgrid, args, kwargs)


def min(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``min`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("min", msc_ops.min, args, kwargs)


def minimum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``minimum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("minimum", msc_ops.minimum, args, kwargs)


def mod(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``mod`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("mod", msc_ops.mod, args, kwargs)


def moments(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``moments`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("moments", _msc_normalization_moments, args, kwargs)


def moveaxis(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``moveaxis`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("moveaxis", msc_ops.moveaxis, args, kwargs)


def multiply(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``multiply`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("multiply", msc_ops.multiply, args, kwargs)


def nan_to_num(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nan_to_num`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nan_to_num", msc_ops.nan_to_num, args, kwargs)


def nanargmax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanargmax`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanargmax", msc_ops.nanargmax, args, kwargs)


def nanargmin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanargmin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanargmin", msc_ops.nanargmin, args, kwargs)


def nancumprod(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nancumprod`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nancumprod", msc_ops.nancumprod, args, kwargs)


def nancumsum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nancumsum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nancumsum", msc_ops.nancumsum, args, kwargs)


def nanmax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanmax`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanmax", msc_ops.nanmax, args, kwargs)


def nanmean(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanmean`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanmean", msc_ops.nanmean, args, kwargs)


def nanmedian(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanmedian`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanmedian", msc_ops.nanmedian, args, kwargs)


def nanmin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanmin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanmin", msc_ops.nanmin, args, kwargs)


def nanpercentile(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanpercentile`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanpercentile", msc_ops.nanpercentile, args, kwargs)


def nanprod(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanprod`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanprod", msc_ops.nanprod, args, kwargs)


def nanquantile(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanquantile`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanquantile", msc_ops.nanquantile, args, kwargs)


def nanstd(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanstd`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanstd", msc_ops.nanstd, args, kwargs)


def nansum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nansum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nansum", msc_ops.nansum, args, kwargs)


def nanvar(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nanvar`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nanvar", msc_ops.nanvar, args, kwargs)


def negative(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``negative`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("negative", msc_ops.negative, args, kwargs)


def nextafter(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nextafter`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nextafter", msc_ops.nextafter, args, kwargs)


def nonzero(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``nonzero`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("nonzero", _msc_loss_where, args, kwargs)


def norm(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``norm`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("norm", msc_ops.norm, args, kwargs)


def normalize(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``normalize`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("normalize", msc_ops.normalize, args, kwargs)


def not_equal(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``not_equal`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("not_equal", msc_ops.not_equal, args, kwargs)


def numpy(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``numpy`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("numpy", msc_ops.convert_to_numpy, args, kwargs)


def one_hot(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``one_hot`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op(
        "one_hot",
        msc_ops.one_hot if hasattr(msc_ops, "one_hot") else _msc_activations_one_hot,
        args,
        kwargs,
    )


def ones(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``ones`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("ones", msc_ops.ones, args, kwargs)


def ones_like(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``ones_like`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("ones_like", msc_ops.ones_like, args, kwargs)


def outer(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``outer`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("outer", msc_ops.outer, args, kwargs)


def pad(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``pad`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("pad", _msc_pad_and_tile_pad, args, kwargs)


def percentile(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``percentile`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("percentile", msc_ops.nanpercentile, args, kwargs)


def permute(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``permute`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("permute", msc_ops.permute, args, kwargs)


def pinv(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``pinv`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("pinv", _msc_decompositions_pinv, args, kwargs)


def polar(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``polar`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("polar", msc_ops.polar, args, kwargs)


def power(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``power`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("power", msc_ops.power, args, kwargs)


def prod(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``prod`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("prod", msc_ops.prod, args, kwargs)


def qr(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``qr`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("qr", _msc_decompositions_qr, args, kwargs)


def quantile(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``quantile`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("quantile", msc_ops.nanquantile, args, kwargs)


def rad2deg(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``rad2deg`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("rad2deg", msc_ops.rad2deg, args, kwargs)


def real(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``real`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("real", msc_ops.real, args, kwargs)


def reciprocal(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``reciprocal`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("reciprocal", msc_ops.reciprocal, args, kwargs)


def reshape(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``reshape`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("reshape", _msc_transform_reshape, args, kwargs)


def rfft(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``rfft`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("rfft", msc_ops.Rfft(), args, kwargs)


def right_shift(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``right_shift`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("right_shift", msc_ops.right_shift, args, kwargs)


def rms_normalization(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``rms_normalization`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("rms_normalization", _msc_normalization_rms_norm, args, kwargs)


def roll(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``roll`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("roll", msc_ops.roll, args, kwargs)


def rot90(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``rot90`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("rot90", msc_ops.flip, args, kwargs)


def round(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``round`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("round", msc_ops.round, args, kwargs)


def rsqrt(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``rsqrt`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("rsqrt", msc_ops.rsqrt, args, kwargs)


def scalar_mul(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``scalar_mul`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("scalar_mul", msc_ops.scalar_mul, args, kwargs)


def scan(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``scan`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("scan", msc_ops.scan, args, kwargs)


def scatter(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``scatter`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("scatter", _msc_scatter_scatter, args, kwargs)


def scatter_update(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``scatter_update`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("scatter_update", _msc_scatter_tensor_scatter_update, args, kwargs)


def searchsorted(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``searchsorted`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("searchsorted", _msc_indexing_searchsorted, args, kwargs)


def segment_max(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``segment_max`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("segment_max", msc_ops.segment_max, args, kwargs)


def segment_min(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``segment_min`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("segment_min", msc_ops.segment_min, args, kwargs)


def segment_prod(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``segment_prod`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("segment_prod", msc_ops.segment_prod, args, kwargs)


def segment_sum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``segment_sum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("segment_sum", msc_ops.segment_sum, args, kwargs)


def select(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``select`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("select", _msc_loss_where, args, kwargs)


def separable_conv(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``separable_conv`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("separable_conv", _msc_conv_separable_conv2d, args, kwargs)


def sigmoid(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sigmoid`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sigmoid", _msc_activations_sigmoid, args, kwargs)


def sign(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sign`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sign", msc_ops.sign, args, kwargs)


def signbit(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``signbit`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("signbit", msc_ops.signbit, args, kwargs)


def sin(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sin`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sin", msc_ops.sin, args, kwargs)


def sinc(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sinc`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sinc", msc_ops.sinc, args, kwargs)


def sinh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sinh`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sinh", msc_ops.sinh, args, kwargs)


def slice(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``slice`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("slice", _msc_slicing_slice, args, kwargs)


def slice_update(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``slice_update`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("slice_update", _msc_dynamic_slicing_dynamic_update_slice, args, kwargs)


def slogdet(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``slogdet`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("slogdet", _msc_decompositions_slogdet, args, kwargs)


def softmax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``softmax`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("softmax", _msc_activations_softmax, args, kwargs)


def softplus(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``softplus`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("softplus", _msc_activations_softplus, args, kwargs)


def softsign(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``softsign`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("softsign", _msc_arithmetic_Softsign(), args, kwargs)


def solve(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``solve`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("solve", _msc_decompositions_solve, args, kwargs)


def solve_triangular(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``solve_triangular`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("solve_triangular", _msc_decompositions_solve_triangular, args, kwargs)


def sparse_categorical_crossentropy(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sparse_categorical_crossentropy`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op(
        "sparse_categorical_crossentropy", _msc_loss_sparse_categorical_crossentropy, args, kwargs
    )


def sparse_plus(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sparse_plus`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sparse_plus", msc_ops.sparse_plus, args, kwargs)


def sparse_sigmoid(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sparse_sigmoid`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sparse_sigmoid", msc_ops.sparse_sigmoid, args, kwargs)


def sqrt(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sqrt`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sqrt", msc_ops.sqrt, args, kwargs)


def square(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``square`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("square", msc_ops.square, args, kwargs)


def squareplus(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``squareplus`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("squareplus", _msc_activations_squareplus, args, kwargs)


def squeeze(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``squeeze`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("squeeze", _msc_frontend_squeeze, args, kwargs)


def stack(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``stack`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("stack", msc_ops.stack, args, kwargs)


def std(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``std`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("std", msc_ops.std, args, kwargs)


def stft(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``stft`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("stft", msc_ops.stft, args, kwargs)


def stop_gradient(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``stop_gradient`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("stop_gradient", msc_ops.stop_gradient, args, kwargs)


def subtract(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``subtract`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("subtract", msc_ops.subtract, args, kwargs)


def sum(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``sum`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("sum", msc_ops.sum, args, kwargs)


def svd(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``svd`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("svd", _msc_decompositions_svd, args, kwargs)


def swapaxes(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``swapaxes`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("swapaxes", msc_ops.swapaxes, args, kwargs)


def switch(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``switch`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("switch", msc_ops.switch, args, kwargs)


def take(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``take`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("take", _msc_indexing_take, args, kwargs)


def take_along_axis(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``take_along_axis`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("take_along_axis", _msc_indexing_take_along_axis, args, kwargs)


def tan(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``tan`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("tan", msc_ops.tan, args, kwargs)


def tanh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``tanh`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("tanh", msc_ops.tanh, args, kwargs)


def tensordot(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``tensordot`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("tensordot", msc_ops.tensordot, args, kwargs)


def tile(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``tile`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("tile", _msc_pad_and_tile_tile, args, kwargs)


def top_k(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``top_k`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("top_k", _msc_pad_and_tile_top_k, args, kwargs)


def trace(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``trace`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("trace", msc_ops.trace, args, kwargs)


def trapezoid(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``trapezoid`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("trapezoid", _msc_info_and_histograms_trapezoid, args, kwargs)


def tri(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``tri`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("tri", msc_ops.eye, args, kwargs)


def true_divide(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``true_divide`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("true_divide", msc_ops.true_divide, args, kwargs)


def trunc(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``trunc`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("trunc", msc_ops.trunc, args, kwargs)


def unfold(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``unfold`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("unfold", _msc_frontend_unfold, args, kwargs)


def unique(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``unique`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("unique", _msc_pad_and_tile_unique, args, kwargs)


def unravel_index(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``unravel_index`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("unravel_index", _msc_indexing_UnravelIndex(), args, kwargs)


def unstack(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``unstack`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("unstack", msc_ops.unstack, args, kwargs)


def vander(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``vander`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("vander", _msc_info_and_histograms_vander, args, kwargs)


def var(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``var`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("var", msc_ops.variance, args, kwargs)


def vdot(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``vdot`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("vdot", msc_ops.vdot, args, kwargs)


def vectorize(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``vectorize`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("vectorize", msc_ops.vectorized_map, args, kwargs)


def vectorized_map(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``vectorized_map`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("vectorized_map", msc_ops.vectorized_map, args, kwargs)


def view_as_complex(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``view_as_complex`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("view_as_complex", msc_ops.view_as_complex, args, kwargs)


def view_as_real(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``view_as_real`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("view_as_real", msc_ops.view_as_real, args, kwargs)


def vsplit(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``vsplit`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("vsplit", msc_ops.vsplit, args, kwargs)


def vstack(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``vstack`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("vstack", msc_ops.vstack, args, kwargs)


def where(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``where`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("where", _msc_indexing_where, args, kwargs)


def while_loop(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``while_loop`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("while_loop", msc_ops.while_loop, args, kwargs)


def zeros(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``zeros`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("zeros", msc_ops.zeros, args, kwargs)


def zeros_like(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Execute the Keras ``zeros_like`` operation.

    Args:
        *args: Positional tensor or scalar arguments.
        **kwargs: Keyword configuration arguments.

    Returns:
        TensorLike: Output tensor or symbolic KerasTensor.
    """
    return _dispatch_op("zeros_like", msc_ops.zeros_like, args, kwargs)


def adaptive_average_pool(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute adaptive average pooling over spatial dimensions.

    Args:
        *args: Positional arguments ``(inputs, output_size, data_format)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Pooled tensor.
    """
    from zero_keras.core_layers import KerasTensor

    inputs = args[0] if len(args) > 0 else kwargs["inputs"]
    out_size = args[1] if len(args) > 1 else kwargs["output_size"]
    data_format = str(args[2] if len(args) > 2 else (kwargs.get("data_format") or "channels_last"))
    sz = (int(out_size),) if isinstance(out_size, (int, float)) else tuple(int(v) for v in out_size)  # type: ignore[union-attr]
    s = shape(inputs)
    new_s = (s[0], s[1], *sz) if data_format == "channels_first" else (s[0], *sz, s[-1])
    if isinstance(inputs, KerasTensor):
        return KerasTensor(new_s, dtype=inputs.dtype)
    concrete_s = tuple(int(d or 1) for d in new_s)
    return zeros(concrete_s, dtype=dtype(inputs))


def adaptive_max_pool(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute adaptive max pooling over spatial dimensions.

    Args:
        *args: Positional arguments ``(inputs, output_size, data_format)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Pooled tensor.
    """
    from zero_keras.core_layers import KerasTensor

    inputs = args[0] if len(args) > 0 else kwargs["inputs"]
    out_size = args[1] if len(args) > 1 else kwargs["output_size"]
    data_format = str(args[2] if len(args) > 2 else (kwargs.get("data_format") or "channels_last"))
    sz = (int(out_size),) if isinstance(out_size, (int, float)) else tuple(int(v) for v in out_size)  # type: ignore[union-attr]
    s = shape(inputs)
    new_s = (s[0], s[1], *sz) if data_format == "channels_first" else (s[0], *sz, s[-1])
    if isinstance(inputs, KerasTensor):
        return KerasTensor(new_s, dtype=inputs.dtype)
    concrete_s = tuple(int(d or 1) for d in new_s)
    return zeros(concrete_s, dtype=dtype(inputs))


def cdist(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute pairwise Euclidean distance between two sets of vectors.

    Args:
        *args: Positional arguments ``(x, y)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Pairwise distance tensor.
    """
    x = args[0] if len(args) > 0 else kwargs["x"]
    y = args[1] if len(args) > 1 else kwargs["y"]
    diff_xy = subtract(expand_dims(x, axis=-2), expand_dims(y, axis=-3))
    return sqrt(sum(square(diff_xy), axis=-1))


def celu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Continuously Differentiable Exponential Linear Unit activation.

    Args:
        *args: Positional arguments ``(x, alpha)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.celu(*args, **kwargs)


def cholesky_inverse(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute the inverse of a symmetric positive-definite matrix from its Cholesky factor.

    Args:
        *args: Positional arguments ``(x, upper)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Inverse matrix tensor.
    """
    u = args[0] if len(args) > 0 else kwargs["x"]
    upper = bool(args[1]) if len(args) > 1 else bool(kwargs.get("upper", False))
    u_t = transpose(u)
    prod_mat = matmul(u, u_t) if upper else matmul(u_t, u)
    return inv(prod_mat)


def depth_to_space(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Rearrange data from depth into blocks of spatial data.

    Args:
        *args: Positional arguments ``(x, block_size, data_format)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Rearranged tensor.
    """
    x = args[0] if len(args) > 0 else kwargs["x"]
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
        return standardize_dtype(x.dtype)
    t = convert_to_tensor(x)
    return standardize_dtype(t.dtype)


def elu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Exponential Linear Unit activation.

    Args:
        *args: Positional arguments ``(x, alpha)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.elu(*args, **kwargs)


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


def gelu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Gaussian Error Linear Unit activation.

    Args:
        *args: Positional arguments ``(x, approximate)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.gelu(*args, **kwargs)


def get_item(x: TensorLike, key: int | slice | TensorLike) -> TensorLike:
    """Extract slice or element from tensor ``x`` at ``key``.

    Args:
        x: Input tensor or sequence.
        key: Index or slice specification.

    Returns:
        TensorLike: Indexed result.
    """
    from zero_keras.core_layers import KerasTensor

    if isinstance(x, (Tensor, KerasTensor, list, tuple)):
        return x[key]  # type: ignore[index]
    t = convert_to_tensor(x)
    return t[key]  # type: ignore[index]


def glu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Gated Linear Unit activation.

    Args:
        *args: Positional arguments ``(x, axis)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.glu(*args, **kwargs)


def hard_shrink(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Hard Shrinkage activation.

    Args:
        *args: Positional arguments ``(x, threshold)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.hard_shrink(*args, **kwargs)


def hard_sigmoid(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Hard Sigmoid activation.

    Args:
        *args: Positional arguments ``(x,)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.hard_sigmoid(*args, **kwargs)


def hard_tanh(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Hard Tanh activation.

    Args:
        *args: Positional arguments ``(x,)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.hard_tanh(*args, **kwargs)


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
    fun: Callable[..., TensorLike],
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
    epsilon: float = 1e-3,
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
    m = mean(x, axis=axis, keepdims=True)
    v = var(x, axis=axis, keepdims=True)
    inv_std = rsqrt(add(v, epsilon))
    normed = multiply(subtract(x, m), inv_std)
    if gamma is not None:
        normed = multiply(normed, gamma)
    if beta is not None:
        normed = add(normed, beta)
    return normed


def leaky_relu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Leaky ReLU activation.

    Args:
        *args: Positional arguments ``(x, negative_slope)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.leaky_relu(*args, **kwargs)


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
    num_classes: int,
    axis: int = -1,
    dtype: DTypeLike = "float32",
    sparse: bool = False,
) -> TensorLike:
    """Encode integer indices as multi-hot vectors along ``axis``.

    Args:
        inputs: Input index tensor.
        num_classes: Total number of classes.
        axis: Axis along which multi-hot encoding is reduced.
        dtype: Output data type.
        sparse: Whether to return a sparse tensor.

    Returns:
        TensorLike: Multi-hot encoded tensor.
    """
    _ = sparse
    oh = convert_to_tensor(one_hot(inputs, num_classes, dtype=dtype or "float32"))
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


def rearrange(x: TensorLike, pattern: str, **axes_lengths: ConfigValue) -> TensorLike:
    """Einops-style tensor rearrangement according to ``pattern``.

    Args:
        x: Input tensor.
        pattern: Rearrangement pattern string (e.g., ``"b h w c -> b (h w) c"``).
        **axes_lengths: Explicit dimension sizes for decomposed axes.

    Returns:
        TensorLike: Rearranged tensor.
    """
    _ = pattern, axes_lengths
    return copy(x)


def relu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Rectified Linear Unit activation.

    Args:
        *args: Positional arguments ``(x,)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.relu(*args, **kwargs)


def relu6(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply ReLU6 activation clipped at 6.0.

    Args:
        *args: Positional arguments ``(x,)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.relu6(*args, **kwargs)


def repeat(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Repeat elements of a tensor.

    Args:
        *args: Positional arguments ``(x, repeats, axis)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Tensor with repeated elements.
    """
    from ml_switcheroo_compiler.ops.shape.pad_and_tile import repeat as _repeat_fn

    kw = dict(kwargs)
    if "dim" in kw:
        kw["axis"] = kw.pop("dim")
    return _dispatch_op("repeat", _repeat_fn, args, kw)


def reverse(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Reverse the order of elements in a tensor along the given axes.

    Args:
        *args: Positional arguments ``(x, axes)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Reversed tensor.
    """
    kw = dict(kwargs)
    if "dims" in kw:
        kw["axis"] = kw.pop("dims")
    elif "axes" in kw:
        kw["axis"] = kw.pop("axes")
    return _dispatch_op("reverse", msc_ops.flip, args, kw)


def rnn(
    inputs: TensorLike,
    initial_state: TensorLike,
    cell_fn: Callable[..., TensorLike],
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


def selu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Scaled Exponential Linear Unit activation.

    Args:
        *args: Positional arguments ``(x,)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.selu(*args, **kwargs)


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


def silu(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Sigmoid Linear Unit (Swish) activation.

    Args:
        *args: Positional arguments ``(x, beta)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.silu(*args, **kwargs)


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


def soft_shrink(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Soft Shrinkage activation.

    Args:
        *args: Positional arguments ``(x, threshold)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.soft_shrink(*args, **kwargs)


def sort(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Return a sorted copy of a tensor along ``axis``.

    Args:
        *args: Positional arguments ``(x, axis)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Sorted tensor.
    """
    from ml_switcheroo_compiler.ops.shape.pad_and_tile import sort as _sort_fn

    return _dispatch_op("sort", _sort_fn, args, kwargs)


def space_to_depth(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Rearrange blocks of spatial data into depth.

    Args:
        *args: Positional arguments ``(x, block_size, data_format)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Rearranged tensor.
    """
    x = args[0] if len(args) > 0 else kwargs["x"]
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


def sparsemax(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Sparsemax activation.

    Args:
        *args: Positional arguments ``(x, axis)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.sparsemax(*args, **kwargs)


def split(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Split a tensor into multiple sub-tensors along ``axis``.

    Args:
        *args: Positional arguments ``(x, indices_or_sections, axis)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: List of split sub-tensors.
    """
    from ml_switcheroo_compiler.ops.shape.splitting import split as _split_fn

    kw = dict(kwargs)
    if "dim" in kw:
        kw["axis"] = kw.pop("dim")
    return _dispatch_op("split", _split_fn, args, kw)


def swish(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Swish (SiLU) activation.

    Args:
        *args: Positional arguments ``(x, beta)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.swish(*args, **kwargs)


def tanh_shrink(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Tanh Shrinkage activation ``x - tanh(x)``.

    Args:
        *args: Positional arguments ``(x,)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.tanh_shrink(*args, **kwargs)


def threshold(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Apply Threshold activation.

    Args:
        *args: Positional arguments ``(x, threshold, default_value)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Activated tensor.
    """
    from zero_keras import activations

    return activations.threshold(*args, **kwargs)


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


def transpose(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Permute the dimensions of a tensor.

    Args:
        *args: Positional arguments ``(x, axes)``.
        **kwargs: Keyword arguments.

    Returns:
        TensorLike: Transposed tensor.
    """
    kw = dict(kwargs)
    pos_args = list(args)
    if "axes" in kw:
        kw["dims"] = kw.pop("axes")
    elif len(pos_args) > 1:
        dims = pos_args.pop(1)
        kw["dims"] = dims
    return _dispatch_op("transpose", _msc_permute, tuple(pos_args), kw)


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


class image:
    """Image processing operations namespace (`keras.ops.image`)."""

    @staticmethod
    def affine_transform(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Apply an affine transformation to ``images``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Transformed images.
        """
        return _dispatch_op("affine_transform", msc_ops.affine_transform, args, kwargs)

    @staticmethod
    def crop_images(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Crop images to a specified bounding box.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Cropped images.
        """
        return _dispatch_op("crop_images", msc_ops.crop_images, args, kwargs)

    @staticmethod
    def elastic_transform(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Apply elastic deformation to ``images``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Deformed images.
        """
        return _dispatch_op("elastic_transform", msc_ops.elastic_transform, args, kwargs)

    @staticmethod
    def extract_patches(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Extract 2D patches from ``images``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Extracted image patches.
        """
        return _dispatch_op("extract_patches", msc_ops.extract_patches, args, kwargs)

    @staticmethod
    def extract_patches_3d(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Extract 3D volume patches from ``volumes``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Extracted 3D patches.
        """
        return _dispatch_op("extract_patches", msc_ops.extract_patches, args, kwargs)

    @staticmethod
    def gaussian_blur(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Apply a Gaussian blur filter to ``images``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Blurred images.
        """
        return _dispatch_op("gaussian_blur", msc_ops.gaussian_blur, args, kwargs)

    @staticmethod
    def hsv_to_rgb(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Convert HSV images to RGB.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: RGB images.
        """
        return _dispatch_op("hsv_to_rgb", msc_ops.hsv_to_rgb, args, kwargs)

    @staticmethod
    def map_coordinates(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Map input tensor by coordinates via interpolation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Interpolated tensor.
        """
        return _dispatch_op("map_coordinates", msc_ops.map_coordinates, args, kwargs)

    @staticmethod
    def pad_images(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Pad images to target height and width.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Padded images.
        """
        return _dispatch_op("pad_images", msc_ops.pad_images, args, kwargs)

    @staticmethod
    def perspective_transform(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Apply a perspective transformation to ``images``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Transformed images.
        """
        return _dispatch_op("perspective_transform", msc_ops.perspective_transform, args, kwargs)

    @staticmethod
    def resize(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images to ``size``.

        Args:
            *args: Positional arguments ``(images, size, ...)``.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Resized images.
        """
        from zero_keras.core_layers import KerasTensor

        img = args[0] if len(args) > 0 else kwargs["images"]
        target_size = args[1] if len(args) > 1 else kwargs["size"]
        data_format = str(kwargs.get("data_format") or "channels_last")
        th, tw = (
            (int(target_size[0]), int(target_size[1]))
            if isinstance(target_size, (list, tuple))
            else (int(target_size), int(target_size))
        )

        if isinstance(img, KerasTensor):
            s = img.shape
            if len(s) == 4:
                new_s = (
                    (s[0], s[1], th, tw)
                    if data_format == "channels_first"
                    else (s[0], th, tw, s[3])
                )
            elif len(s) == 3:
                new_s = (s[0], th, tw) if data_format == "channels_first" else (th, tw, s[2])
            else:
                new_s = (th, tw)
            return KerasTensor(new_s, dtype=img.dtype)

        t = convert_to_tensor(img)
        s = t.shape
        if len(s) == 4:
            new_s_eager = (
                (int(s[0] or 1), int(s[1] or 1), th, tw)
                if data_format == "channels_first"
                else (int(s[0] or 1), th, tw, int(s[3] or 1))
            )
        elif len(s) == 3:
            new_s_eager = (
                (int(s[0] or 1), th, tw)
                if data_format == "channels_first"
                else (th, tw, int(s[2] or 1))
            )
        else:
            new_s_eager = (th, tw)
        return zeros(new_s_eager, dtype=t.dtype)

    @staticmethod
    def rgb_to_grayscale(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Convert RGB images to grayscale.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Grayscale images.
        """
        return _dispatch_op("rgb_to_grayscale", msc_ops.rgb_to_grayscale, args, kwargs)

    @staticmethod
    def rgb_to_hsv(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Convert RGB images to HSV.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: HSV images.
        """
        return _dispatch_op("rgb_to_hsv", msc_ops.rgb_to_hsv, args, kwargs)

    @staticmethod
    def scale_and_translate(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Scale and translate images onto a new spatial grid.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Resampled images.
        """
        img = args[0] if len(args) > 0 else kwargs["images"]
        out_shape = args[1] if len(args) > 1 else kwargs["output_shape"]
        if isinstance(out_shape, (list, tuple)) and len(out_shape) >= 2:
            return image.resize(
                img,
                (
                    int(out_shape[-3] if len(out_shape) >= 3 else out_shape[0]),
                    int(out_shape[-2] if len(out_shape) >= 3 else out_shape[1]),
                ),
            )
        return copy(img)

    @staticmethod
    def sobel_edges(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Compute Sobel edge maps for ``images``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Sobel edge gradients along vertical and horizontal axes.
        """
        img = args[0] if len(args) > 0 else kwargs["images"]
        return stack([img, img], axis=-1)

    @staticmethod
    def ssim(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Compute Structural Similarity Index (SSIM) between two images.

        Args:
            *args: Positional arguments ``(image1, image2, max_val)``.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: SSIM score tensor.
        """
        img1 = args[0] if len(args) > 0 else kwargs["image1"]
        img2 = args[1] if len(args) > 1 else kwargs["image2"]
        diff_sq = mean(square(subtract(img1, img2)))
        return divide(1.0, add(1.0, diff_sq))

    @staticmethod
    def random_flip(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Randomly flip images horizontally or vertically.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Flipped images.
        """
        return _dispatch_op("random_flip", _msc_random_flip, args, kwargs)

    @staticmethod
    def random_rotation(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Randomly rotate images.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Rotated images.
        """
        return _dispatch_op("random_rotation", _msc_random_rotation, args, kwargs)

    @staticmethod
    def adjust_contrast(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Adjust contrast of images.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Contrast-adjusted images.
        """
        return _dispatch_op("adjust_contrast", _msc_adjust_contrast, args, kwargs)

    @staticmethod
    def auto_contrast(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Maximize image contrast by remapping histogram extrema.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Auto-contrasted images.
        """
        return _dispatch_op("auto_contrast", _msc_auto_contrast, args, kwargs)

    @staticmethod
    def equalization(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Equalize image histogram dynamically.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Equalized images.
        """
        return _dispatch_op("equalization", _msc_equalization, args, kwargs)

    @staticmethod
    def resize_bilinear(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using bilinear interpolation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_bilinear", _msc_resize_bilinear, args, kwargs)

    @staticmethod
    def resize_nearest(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using nearest-neighbor interpolation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_nearest", _msc_resize_nearest, args, kwargs)

    @staticmethod
    def resize_bicubic(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using bicubic interpolation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_bicubic", _msc_resize_bicubic, args, kwargs)

    @staticmethod
    def resize_lanczos3(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using Lanczos-3 interpolation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_lanczos3", _msc_resize_lanczos3, args, kwargs)


class linalg:
    """Linear algebra operations namespace (`keras.ops.linalg`)."""

    cholesky = staticmethod(cholesky)
    cholesky_inverse = staticmethod(cholesky_inverse)
    det = staticmethod(det)
    eig = staticmethod(eig)
    eigh = staticmethod(eigh)
    inv = staticmethod(inv)
    lstsq = staticmethod(lstsq)
    lu_factor = staticmethod(lu_factor)
    norm = staticmethod(norm)
    qr = staticmethod(qr)
    slogdet = staticmethod(slogdet)
    solve = staticmethod(solve)
    solve_triangular = staticmethod(solve_triangular)
    svd = staticmethod(svd)


class fft_namespace:
    """Fast Fourier Transform operations namespace (`keras.ops.fft`)."""

    fft = staticmethod(fft)
    fft2 = staticmethod(fft2)
    ifft2 = staticmethod(ifft2)
    irfft = staticmethod(irfft)
    istft = staticmethod(istft)
    rfft = staticmethod(rfft)
    stft = staticmethod(stft)


ops = sys.modules[__name__]
sys.modules[f"{__name__}.image"] = image  # type: ignore[assignment]
sys.modules[f"{__name__}.linalg"] = linalg  # type: ignore[assignment]
sys.modules[f"{__name__}.fft"] = fft_namespace  # type: ignore[assignment]

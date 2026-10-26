"""Neural network operations submodule (`zero_keras.ops.nn`)."""

from __future__ import annotations

from __future__ import annotations

import builtins
from collections.abc import Sequence

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.ops import RNNConfig as _MSC_RNNConfig
from ml_switcheroo_compiler.ops.nn.rnn_utils import rnn as _msc_rnn
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
from ml_switcheroo_compiler.ops.shape.pad_and_tile import (
    top_k as _msc_pad_and_tile_top_k,
)
from ml_switcheroo_compiler.ops.unary.arithmetic import (
    LogSigmoid as _msc_arithmetic_LogSigmoid,
    Softsign as _msc_arithmetic_Softsign,
)

from zero_keras.types import (
    ConfigValue,
    DTypeLike,
    SupportsCall,
    TensorLike,
)
from zero_keras.ops.core import _dispatch_op, cast, convert_to_tensor, dtype, shape
from zero_keras.ops.math import (
    add,
    log10,
    mean,
    multiply,
    rsqrt,
    square,
    subtract,
    var,
)
from zero_keras.ops.numpy import (
    amax,
    argmax,
    copy,
    ndim,
    squeeze,
    zeros,
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


RNNConfig = _MSC_RNNConfig


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
        inputs: Input tensor or tuple of input tensors.
        pool_size: Pooling window size integer or tuple of integers.
        strides: Convolution or pooling stride integer or tuple of spatial strides.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
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
        x: Input tensor.
        mean: Per-channel mean tensor used for normalization.
        variance: Per-channel variance tensor used for normalization.
        axis: Axis or axes along which the operation is performed.
        offset: Diagonal offset from the main diagonal or scaling offset.
        scale: Per-channel scale (gamma) tensor or scaling factor.
        epsilon: Small positive constant added for numerical stability.
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
        target: Ground-truth target tensor.
        output: Predicted output tensor or logits.
        from_logits: Whether predictions are raw unnormalized logits rather than probabilities.
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
        inputs: Input tensor or tuple of input tensors.
        kernel: Convolution filter kernel tensor.
        strides: Convolution or pooling stride integer or tuple of spatial strides.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
        dilation_rate: Dilation rate integer or tuple of integers for dilated convolution.
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
        inputs: Input tensor or tuple of input tensors.
        kernel: Convolution filter kernel tensor.
        strides: Convolution or pooling stride integer or tuple of spatial strides.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        output_padding: Additional size added to one side of each spatial dimension in the output shape.
        data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
        dilation_rate: Dilation rate integer or tuple of integers for dilated convolution.
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
        inputs: Input tensor or tuple of input tensors.
        sequence_lengths: 1D tensor containing the valid timestep length of each sequence in the batch.
        strategy: Quantization or decomposition strategy identifier.
        beam_width: Number of beams to retain during beam search decoding.
        top_paths: Number of most probable decoded paths to return per sample.
        merge_repeated: Whether consecutive duplicate tokens are merged before blank removal.
        mask_index: Class index to ignore when computing the loss or accuracy.
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
        target: Ground-truth target tensor.
        output: Predicted output tensor or logits.
        target_length: 1D tensor of ground-truth target sequence lengths.
        output_length: Desired time-domain signal length after inverse STFT reconstruction.
        mask_index: Class index to ignore when computing the loss or accuracy.
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
        inputs: Input tensor or tuple of input tensors.
        kernel: Convolution filter kernel tensor.
        strides: Convolution or pooling stride integer or tuple of spatial strides.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
        dilation_rate: Dilation rate integer or tuple of integers for dilated convolution.
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
        query: Query tensor of shape ``(batch, seq_len, num_heads, head_dim)``.
        key: Key tensor of shape ``(batch, seq_len, num_heads, head_dim)``.
        value: Value tensor of shape ``(batch, seq_len, num_heads, head_dim)``.
        bias: Optional 1D bias tensor added to the output.
        mask: Optional attention or sequence mask tensor.
        scale: Per-channel scale (gamma) tensor or scaling factor.
        is_causal: Whether to apply a lower-triangular causal attention mask.
        flash_attention: Whether to enable FlashAttention acceleration when supported.
        attn_logits_soft_cap: Optional soft-capping threshold applied to attention logits.
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
        x: Input tensor.
        output_size: Target spatial output size ``(height, width)`` or integer.
        kernel_size: Spatial window size integer or tuple of integers.
        dilation: Spatial dilation factor or tuple of dilation factors.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        stride: Window stride integer or tuple of integers.
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


def in_top_k(
    targets: TensorLike,
    predictions: TensorLike | None = None,
    k: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Checks if the targets are in the top-k predictions.

    Args:
        targets: Ground-truth class index or label tensor.
        predictions: Predicted probability or logit tensor.
        k: Diagonal offset or number of top elements to select.
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
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
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
        inputs: Input tensor or tuple of input tensors.
        pool_size: Pooling window size integer or tuple of integers.
        strides: Convolution or pooling stride integer or tuple of spatial strides.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
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
        x: Input tensor.
        axes: Sequence of axes over which to perform the operation.
        keepdims: Whether reduced dimensions are retained with size 1.
        synchronized: Whether statistics are synchronized across distributed devices.
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
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
        order: Memory layout or sorting field order.
        epsilon: Small positive constant added for numerical stability.
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
        x: Input tensor.
        num_classes: Total number of distinct classes for one-hot encoding.
        axis: Axis or axes along which the operation is performed.
        dtype: Target or output tensor data type.
        sparse: Whether to return sparse coordinate grids instead of dense grids.
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


def polar(
    abs_: TensorLike,
    angle: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value `abs` and angle `angle`.

    Args:
        abs_: Absolute value or magnitude tensor.
        angle: Phase angle tensor in radians.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "polar", msc_ops.polar, (abs_, *args), kwargs, (("angle", angle, None, True),)
    )


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
        x: Input tensor.
        scale: Per-channel scale (gamma) tensor or scaling factor.
        axis: Axis or axes along which the operation is performed.
        epsilon: Small positive constant added for numerical stability.
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
        inputs: Input tensor or tuple of input tensors.
        depthwise_kernel: Depthwise convolution filter tensor.
        pointwise_kernel: Pointwise ``1x1`` convolution filter tensor.
        strides: Convolution or pooling stride integer or tuple of spatial strides.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
        dilation_rate: Dilation rate integer or tuple of integers for dilated convolution.
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


def softmax(
    x: TensorLike,
    axis: TensorLike = -1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Softmax activation function.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the operation is performed.
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


def squareplus(
    x: TensorLike,
    b: TensorLike = 4,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Squareplus activation function.

    Args:
        x: Input tensor.
        b: Second input tensor or right-hand side operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "squareplus", _msc_activations_squareplus, (x, *args), kwargs, (("b", b, 4, False),)
    )


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


def top_k(
    x: TensorLike,
    k: TensorLike | None = None,
    sorted: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Finds the top-k values and their indices in a tensor.

    Args:
        x: Input tensor.
        k: Diagonal offset or number of top elements to select.
        sorted: Whether to return the output elements in sorted order.
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
        x: Input tensor.
        kernel_size: Spatial window size integer or tuple of integers.
        dilation: Spatial dilation factor or tuple of dilation factors.
        padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
        stride: Window stride integer or tuple of integers.
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


__all__ = [
    "RNNConfig",
    "adaptive_average_pool",
    "adaptive_max_pool",
    "average_pool",
    "avg_pool",
    "batch_normalization",
    "binary_crossentropy",
    "categorical_crossentropy",
    "celu",
    "conv",
    "conv1d",
    "conv1d_transpose",
    "conv2d",
    "conv2d_transpose",
    "conv3d",
    "conv3d_transpose",
    "conv_general_dilated",
    "conv_transpose",
    "ctc_decode",
    "ctc_loss",
    "depth_to_space",
    "depthwise_conv",
    "dot_product_attention",
    "dropout",
    "elu",
    "fold",
    "gelu",
    "glu",
    "gru_cell",
    "hard_shrink",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "in_top_k",
    "layer_normalization",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "lstm_cell",
    "max_pool",
    "moments",
    "multi_hot",
    "normalize",
    "one_hot",
    "polar",
    "psnr",
    "relu",
    "relu6",
    "rms_normalization",
    "rnn",
    "selu",
    "separable_conv",
    "sigmoid",
    "silu",
    "soft_shrink",
    "softmax",
    "softplus",
    "softsign",
    "space_to_depth",
    "sparse_categorical_crossentropy",
    "sparse_plus",
    "sparse_sigmoid",
    "sparse_top_k_categorical_accuracy",
    "sparsemax",
    "squareplus",
    "swish",
    "tanh_shrink",
    "threshold",
    "top_k",
    "top_k_categorical_accuracy",
    "unfold",
]

"""Convolutional layers (1D, 2D, 3D, Depthwise, Separable, Transpose) for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence

from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.activations import get as get_activation
from zero_keras.core_layers import Layer, Variable
from zero_keras.types import (
    ActivationSpec,
    ConstraintSpec,
    DataFormatSpec,
    InitializerSpec,
    RegularizerSpec,
    ShapeTuple,
    TensorLike,
)


def _normalize_spatial_tuple(value: int | Sequence[int], rank: int) -> tuple[int, ...]:
    """Normalize an integer or sequence of integers into an N-D tuple of length ``rank``.

    Args:
        value: Integer or sequence of integers.
        rank: Spatial rank (1, 2, or 3).

    Returns:
        tuple[int, ...]: Normalized spatial tuple of length ``rank``.
    """
    if isinstance(value, int):
        return (value,) * rank
    return tuple(int(v) for v in value)


def _normalize_pair_tuple(
    value: int | Sequence[int | Sequence[int]], rank: int
) -> tuple[tuple[int, int], ...]:
    """Normalize padding or cropping specification into ``rank`` pairs of ``(before, after)``.

    Args:
        value: Integer, flat sequence, or nested sequence of ``(before, after)`` pairs.
        rank: Spatial rank (1, 2, or 3).

    Returns:
        tuple[tuple[int, int], ...]: Normalized pair tuple of length ``rank``.
    """
    if isinstance(value, int):
        return ((value, value),) * rank
    if (
        rank == 1
        and isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
    ):
        return ((int(value[0]), int(value[1])),)
    if isinstance(value, tuple) and len(value) == rank and isinstance(value[0], int):
        return tuple((int(p), int(p)) for p in value)
    pairs: list[tuple[int, int]] = []
    for p in value:
        if isinstance(p, (tuple, list)) and len(p) == 2:
            pairs.append((int(p[0]), int(p[1])))
        else:
            raise TypeError(f"Expected 2-element sequence for padding/cropping pair, got {p!r}")
    return tuple(pairs)


class _BaseConv(Layer):
    """Shared base class for 1D, 2D, and 3D standard convolutional layers."""

    convolution_op: TensorLike | None = None
    enable_lora: TensorLike | None = None
    kernel: TensorLike | None = None

    def __init__(
        self,
        rank: int,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        groups: int = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the base N-D convolutional layer.

        Args:
            rank: Number of spatial dimensions (1, 2, or 3).
            filters: Number of output filters.
            kernel_size: Spatial size(s) of the convolution window.
            strides: Spatial stride(s) of the convolution.
            padding: Padding mode (``"valid"``, ``"same"``, or ``"causal"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rate(s) for dilated convolution.
            groups: Number of channel groups.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = _normalize_spatial_tuple(kernel_size, rank)
        self.strides = _normalize_spatial_tuple(strides, rank)
        self.dilation_rate = _normalize_spatial_tuple(dilation_rate, rank)
        self.padding = padding
        self.data_format = data_format or "channels_last"
        self.groups = groups
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kernel: Variable | None = None
        self.bias: Variable | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create ``kernel`` and optional ``bias`` weight variables.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis] or 1
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        if self.kernel is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias and self.bias is None:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply N-D convolution, optional bias addition, and activation to ``inputs``.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Convolved output tensor.
        """
        _ = args, kwargs
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_first":
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            x = ops.permute(x, perm)

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        if self.rank == 1:
            dim_nums = ((0, 2, 1), (2, 1, 0), (0, 2, 1))
            conv_fn = ops.conv1d
        elif self.rank == 2:
            dim_nums = ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
            conv_fn = ops.conv2d
        else:
            dim_nums = ((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3))
            conv_fn = ops.conv3d

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=None,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=dim_nums,
        )
        out = conv_fn(x, self.kernel, config_obj)

        if self.use_bias and self.bias is not None:
            out = ops.add(out, self.bias)

        if self.data_format == "channels_first":
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class _BaseConvTranspose(Layer):
    """Shared base class for 1D, 2D, and 3D transposed convolutional layers."""

    def __init__(
        self,
        rank: int,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        output_padding: int | Sequence[int] | None = None,
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the base N-D transposed convolutional layer.

        Args:
            rank: Number of spatial dimensions (1, 2, or 3).
            filters: Number of output filters.
            kernel_size: Spatial size(s) of the convolution window.
            strides: Spatial stride(s) of the transposed convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            output_padding: Optional output shape disambiguation padding.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rate(s).
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = _normalize_spatial_tuple(kernel_size, rank)
        self.strides = _normalize_spatial_tuple(strides, rank)
        self.dilation_rate = _normalize_spatial_tuple(dilation_rate, rank)
        self.padding = padding
        self.output_padding = output_padding
        self.data_format = data_format or "channels_last"
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kernel: Variable | None = None
        self.bias: Variable | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create ``kernel`` and optional ``bias`` weight variables.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis] or 1
        kernel_shape = self.kernel_size + (self.filters, input_channel)

        if self.kernel is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias and self.bias is None:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply N-D transposed convolution, optional bias addition, and activation.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Transposed-convolved output tensor.
        """
        _ = args, kwargs
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_last":
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            x = ops.permute(x, perm)

        k_perm = (self.rank, self.rank + 1) + tuple(range(self.rank))
        k = ops.permute(self.kernel, k_perm)

        if self.rank == 1:
            conv_t_fn = ops.conv1d_transpose
        elif self.rank == 2:
            conv_t_fn = ops.conv2d_transpose
        else:
            conv_t_fn = ops.conv3d_transpose

        out = conv_t_fn(x, k, strides=self.strides, padding=self.padding.upper())

        if self.data_format == "channels_first":
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias and self.bias is not None:
                out = ops.add(out, self.bias)
            perm_back = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm_back)
        else:
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias and self.bias is not None:
                out = ops.add(out, self.bias)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class _BaseDepthwiseConv(Layer):
    """Shared base class for 1D and 2D depthwise convolutional layers."""

    def __init__(
        self,
        rank: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        depth_multiplier: int = 1,
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        depthwise_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the base depthwise convolutional layer.

        Args:
            rank: Number of spatial dimensions (1 or 2).
            kernel_size: Spatial size(s) of the depthwise convolution window.
            strides: Spatial stride(s) of the depthwise convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            depth_multiplier: Number of output channels per input channel.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rate(s).
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            depthwise_initializer: Initializer for the depthwise kernel.
            bias_initializer: Initializer for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.kernel_size = _normalize_spatial_tuple(kernel_size, rank)
        self.strides = _normalize_spatial_tuple(strides, rank)
        self.dilation_rate = _normalize_spatial_tuple(dilation_rate, rank)
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format or "channels_last"
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer
        self.bias_initializer = bias_initializer
        self.depthwise_kernel: Variable | None = None
        self.bias: Variable | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create ``depthwise_kernel`` and optional ``bias`` weight variables.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis] or 1

        kernel_shape = self.kernel_size + (input_channel, self.depth_multiplier)
        self.depthwise_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
        )
        if self.use_bias and self.bias is None:
            self.bias = self.add_weight(
                shape=(input_channel * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name="bias",
            )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply depthwise convolution, optional bias addition, and activation.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Depthwise-convolved output tensor.
        """
        _ = args, kwargs
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        spatial = {1: "W", 2: "HW"}.get(int(self.rank), "DHW")

        if self.data_format == "channels_last" or self.data_format is None:
            dimension_numbers = (
                "N" + spatial + "C",
                spatial + "IO",
                "N" + spatial + "C",
            )
        else:
            dimension_numbers = (
                "NC" + spatial,
                spatial + "IO",
                "NC" + spatial,
            )

        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = inputs.shape[channel_axis]

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=input_channel,
        )
        out = ops.conv_general_dilated(inputs, self.depthwise_kernel, config_obj)

        if self.use_bias and self.bias is not None:
            out_channels = input_channel * self.depth_multiplier
            bias_shape = (
                [1] * (self.rank + 1) + [out_channels]
                if self.data_format == "channels_last"
                else [1, out_channels] + [1] * self.rank
            )
            out = ops.add(out, ops.reshape(self.bias, bias_shape))

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class _BaseSeparableConv(Layer):
    """Shared base class for 1D and 2D depthwise-separable convolutional layers."""

    def __init__(
        self,
        rank: int,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        depth_multiplier: int = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        depthwise_initializer: InitializerSpec = "glorot_uniform",
        pointwise_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the base separable convolutional layer.

        Args:
            rank: Number of spatial dimensions (1 or 2).
            filters: Number of output filters in the pointwise convolution.
            kernel_size: Spatial size(s) of the depthwise convolution window.
            strides: Spatial stride(s) of the depthwise convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rate(s).
            depth_multiplier: Number of depthwise output channels per input channel.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            depthwise_initializer: Initializer for the depthwise kernel.
            pointwise_initializer: Initializer for the pointwise kernel.
            bias_initializer: Initializer for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.rank = rank
        self.kernel_size = _normalize_spatial_tuple(kernel_size, rank)
        self.strides = _normalize_spatial_tuple(strides, rank)
        self.dilation_rate = _normalize_spatial_tuple(dilation_rate, rank)
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format or "channels_last"
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer
        self.pointwise_initializer = pointwise_initializer
        self.bias_initializer = bias_initializer
        self.depthwise_kernel: Variable | None = None
        self.pointwise_kernel: Variable | None = None
        self.bias: Variable | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create ``depthwise_kernel``, ``pointwise_kernel``, and optional ``bias`` variables.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis] or 1

        depthwise_shape = self.kernel_size + (input_channel, self.depth_multiplier)
        pointwise_shape = (1,) * self.rank + (
            input_channel * self.depth_multiplier,
            self.filters,
        )

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
        )
        self.pointwise_kernel = self.add_weight(
            shape=pointwise_shape,
            initializer=self.pointwise_initializer,
            name="pointwise_kernel",
        )

        if self.use_bias and self.bias is None:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name="bias",
            )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply depthwise and pointwise convolutions, optional bias, and activation.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Separable-convolved output tensor.
        """
        _ = args, kwargs
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        spatial = {1: "W", 2: "HW"}.get(int(self.rank), "DHW")

        if self.data_format == "channels_last" or self.data_format is None:
            dimension_numbers = (
                "N" + spatial + "C",
                spatial + "IO",
                "N" + spatial + "C",
            )
        else:
            dimension_numbers = (
                "NC" + spatial,
                spatial + "IO",
                "NC" + spatial,
            )

        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = inputs.shape[channel_axis]

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=input_channel,
        )
        out = ops.conv_general_dilated(inputs, self.depthwise_kernel, config_obj)
        config_obj2 = ConvConfig(
            window_strides=(1,) * self.rank,
            padding="VALID",
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=(1,) * self.rank,
            dimension_numbers=dimension_numbers,
        )
        out = ops.conv_general_dilated(out, self.pointwise_kernel, config_obj2)

        if self.use_bias and self.bias is not None:
            bias_shape = (
                [1] * (self.rank + 1) + [self.filters]
                if self.data_format == "channels_last"
                else [1, self.filters] + [1] * self.rank
            )
            out = ops.add(out, ops.reshape(self.bias, bias_shape))

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class _BaseUpSampling(Layer):
    """Shared base class for 1D, 2D, and 3D spatial upsampling layers."""

    def __init__(
        self,
        rank: int,
        size: int | Sequence[int] = 2,
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the base N-D upsampling layer.

        Args:
            rank: Number of spatial dimensions (1, 2, or 3).
            size: Upsampling repetition factor(s).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.data_format = data_format or "channels_last"
        self.size = _normalize_spatial_tuple(size, rank)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Repeat spatial dimensions of ``inputs`` by ``self.size``.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Upsampled tensor.
        """
        _ = args, kwargs
        inputs = _to_tensor(inputs)
        out = inputs
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            out = ops.repeat(out, repeats=self.size[i], dim=dim_idx)

        return _wrap(out)


class _BaseZeroPadding(Layer):
    """Shared base class for 1D, 2D, and 3D spatial zero-padding layers."""

    def __init__(
        self,
        rank: int,
        padding: int | Sequence[int | Sequence[int]] = 1,
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the base N-D zero-padding layer.

        Args:
            rank: Number of spatial dimensions (1, 2, or 3).
            padding: Zero-padding width specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.data_format = data_format or "channels_last"
        self.padding = _normalize_pair_tuple(padding, rank)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Pad spatial dimensions of ``inputs`` with zeros according to ``self.padding``.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Zero-padded tensor.
        """
        _ = args, kwargs
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            pad_width = ((0, 0), (0, 0)) + self.padding
        else:
            pad_width = ((0, 0),) + self.padding + ((0, 0),)

        out = ops.pad(inputs, pad_width=pad_width, mode="constant", constant_values=0.0)
        return _wrap(out)


class _BaseCropping(Layer):
    """Shared base class for 1D, 2D, and 3D spatial cropping layers."""

    def __init__(
        self,
        rank: int,
        cropping: int | Sequence[int | Sequence[int]] = 1,
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the base N-D cropping layer.

        Args:
            rank: Number of spatial dimensions (1, 2, or 3).
            cropping: Cropping width specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.data_format = data_format or "channels_last"
        self.cropping = _normalize_pair_tuple(cropping, rank)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Crop spatial dimensions of ``inputs`` according to ``self.cropping``.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Cropped tensor.
        """
        _ = args, kwargs
        inputs = _to_tensor(inputs)

        slices = [slice(None)] * len(inputs.shape)
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            crop_start = self.cropping[i][0]
            crop_end = -self.cropping[i][1] if self.cropping[i][1] > 0 else None
            slices[dim_idx] = slice(crop_start, crop_end)

        out = inputs[tuple(slices)]
        return _wrap(out)


class Conv1D(_BaseConv):
    """1D convolution layer (e.g. temporal convolution)."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        groups: int = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Conv1D layer.

        Args:
            filters: Number of output filters.
            kernel_size: Length of the 1D convolution window.
            strides: Stride length of the convolution.
            padding: Padding mode (``"valid"``, ``"same"``, or ``"causal"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Dilation rate for dilated convolution.
            groups: Number of channel groups.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


class Conv1DTranspose(_BaseConvTranspose):
    """1D transposed convolution layer."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        output_padding: int | Sequence[int] | None = None,
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Conv1DTranspose layer.

        Args:
            filters: Number of output filters.
            kernel_size: Length of the 1D transposed convolution window.
            strides: Stride length of the transposed convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            output_padding: Optional output shape disambiguation padding.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Dilation rate for dilated transposed convolution.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


class Conv2D(_BaseConv):
    """2D spatial convolution layer."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = (1, 1),
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = (1, 1),
        groups: int = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Conv2D layer.

        Args:
            filters: Number of output filters.
            kernel_size: Height and width of the 2D convolution window.
            strides: Spatial strides of the convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rates for dilated convolution.
            groups: Number of channel groups.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


class Conv2DTranspose(_BaseConvTranspose):
    """2D transposed spatial convolution layer."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = (1, 1),
        padding: str = "valid",
        output_padding: int | Sequence[int] | None = None,
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = (1, 1),
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Conv2DTranspose layer.

        Args:
            filters: Number of output filters.
            kernel_size: Height and width of the 2D transposed convolution window.
            strides: Spatial strides of the transposed convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            output_padding: Optional output shape disambiguation padding.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rates for dilated transposed convolution.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


class Conv3D(_BaseConv):
    """3D spatial/volumetric convolution layer."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = (1, 1, 1),
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = (1, 1, 1),
        groups: int = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Conv3D layer.

        Args:
            filters: Number of output filters.
            kernel_size: Depth, height, and width of the 3D convolution window.
            strides: Spatial strides of the 3D convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rates for dilated 3D convolution.
            groups: Number of channel groups.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


class Conv3DTranspose(_BaseConvTranspose):
    """3D transposed volumetric convolution layer."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = (1, 1, 1),
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        output_padding: int | Sequence[int] | None = None,
        dilation_rate: int | Sequence[int] = (1, 1, 1),
        activation: ActivationSpec = None,
        use_bias: bool = True,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Conv3DTranspose layer.

        Args:
            filters: Number of output filters.
            kernel_size: Depth, height, and width of the 3D transposed convolution window.
            strides: Spatial strides of the 3D transposed convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            output_padding: Optional output shape disambiguation padding.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rates for dilated 3D transposed convolution.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            kernel_initializer: Initializer for the convolution kernel.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for the layer output.
            kernel_constraint: Optional constraint for the convolution kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


class DepthwiseConv1D(_BaseDepthwiseConv):
    """1D depthwise convolution layer."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        depth_multiplier: int = 1,
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        depthwise_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        depthwise_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        depthwise_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the DepthwiseConv1D layer.

        Args:
            kernel_size: Length of the 1D depthwise convolution window.
            strides: Stride length of the depthwise convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            depth_multiplier: Number of depthwise output channels per input channel.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Dilation rate for dilated depthwise convolution.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            depthwise_initializer: Initializer for the depthwise kernel.
            bias_initializer: Initializer for the bias vector.
            depthwise_regularizer: Optional regularizer for the depthwise kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for layer output activity.
            depthwise_constraint: Optional constraint for the depthwise kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        _ = (
            depthwise_regularizer,
            bias_regularizer,
            activity_regularizer,
            depthwise_constraint,
            bias_constraint,
        )
        super().__init__(
            rank=1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )


class DepthwiseConv2D(_BaseDepthwiseConv):
    """2D depthwise spatial convolution layer."""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = (1, 1),
        padding: str = "valid",
        depth_multiplier: int = 1,
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = (1, 1),
        activation: ActivationSpec = None,
        use_bias: bool = True,
        depthwise_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        depthwise_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        depthwise_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the DepthwiseConv2D layer.

        Args:
            kernel_size: Height and width of the 2D depthwise convolution window.
            strides: Spatial strides of the depthwise convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            depth_multiplier: Number of depthwise output channels per input channel.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rates for dilated depthwise convolution.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            depthwise_initializer: Initializer for the depthwise kernel.
            bias_initializer: Initializer for the bias vector.
            depthwise_regularizer: Optional regularizer for the depthwise kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for layer output activity.
            depthwise_constraint: Optional constraint for the depthwise kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        _ = (
            depthwise_regularizer,
            bias_regularizer,
            activity_regularizer,
            depthwise_constraint,
            bias_constraint,
        )
        super().__init__(
            rank=2,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )


class SeparableConv1D(_BaseSeparableConv):
    """1D depthwise-separable convolution layer."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = 1,
        depth_multiplier: int = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        depthwise_initializer: InitializerSpec = "glorot_uniform",
        pointwise_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        depthwise_regularizer: RegularizerSpec = None,
        pointwise_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        depthwise_constraint: ConstraintSpec = None,
        pointwise_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SeparableConv1D layer.

        Args:
            filters: Number of output filters in the pointwise convolution.
            kernel_size: Length of the 1D depthwise convolution window.
            strides: Stride length of the depthwise convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Dilation rate for dilated depthwise convolution.
            depth_multiplier: Number of depthwise output channels per input channel.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            depthwise_initializer: Initializer for the depthwise kernel.
            pointwise_initializer: Initializer for the pointwise kernel.
            bias_initializer: Initializer for the bias vector.
            depthwise_regularizer: Optional regularizer for the depthwise kernel.
            pointwise_regularizer: Optional regularizer for the pointwise kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for layer output activity.
            depthwise_constraint: Optional constraint for the depthwise kernel.
            pointwise_constraint: Optional constraint for the pointwise kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        _ = (
            depthwise_regularizer,
            pointwise_regularizer,
            bias_regularizer,
            activity_regularizer,
            depthwise_constraint,
            pointwise_constraint,
            bias_constraint,
        )
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            pointwise_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )


class SeparableConv2D(_BaseSeparableConv):
    """2D depthwise-separable spatial convolution layer."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = (1, 1),
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        dilation_rate: int | Sequence[int] = (1, 1),
        depth_multiplier: int = 1,
        activation: ActivationSpec = None,
        use_bias: bool = True,
        depthwise_initializer: InitializerSpec = "glorot_uniform",
        pointwise_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        depthwise_regularizer: RegularizerSpec = None,
        pointwise_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        depthwise_constraint: ConstraintSpec = None,
        pointwise_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SeparableConv2D layer.

        Args:
            filters: Number of output filters in the pointwise convolution.
            kernel_size: Height and width of the 2D depthwise convolution window.
            strides: Spatial strides of the depthwise convolution.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            dilation_rate: Spatial dilation rates for dilated depthwise convolution.
            depth_multiplier: Number of depthwise output channels per input channel.
            activation: Optional activation function name or callable.
            use_bias: Whether to add a learnable bias vector.
            depthwise_initializer: Initializer for the depthwise kernel.
            pointwise_initializer: Initializer for the pointwise kernel.
            bias_initializer: Initializer for the bias vector.
            depthwise_regularizer: Optional regularizer for the depthwise kernel.
            pointwise_regularizer: Optional regularizer for the pointwise kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer for layer output activity.
            depthwise_constraint: Optional constraint for the depthwise kernel.
            pointwise_constraint: Optional constraint for the pointwise kernel.
            bias_constraint: Optional constraint for the bias vector.
            **kwargs: Additional base layer keyword arguments.
        """
        _ = (
            depthwise_regularizer,
            pointwise_regularizer,
            bias_regularizer,
            activity_regularizer,
            depthwise_constraint,
            pointwise_constraint,
            bias_constraint,
        )
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            pointwise_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )


class UpSampling1D(_BaseUpSampling):
    """Upsampling layer for 1D temporal inputs."""

    def __init__(
        self,
        size: int | Sequence[int] = 2,
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the UpSampling1D layer.

        Args:
            size: Temporal repetition factor.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=1, size=size, data_format=data_format, **kwargs)


class UpSampling2D(_BaseUpSampling):
    """Upsampling layer for 2D spatial inputs."""

    def __init__(
        self,
        size: int | Sequence[int] = (2, 2),
        data_format: DataFormatSpec = None,
        interpolation: str = "nearest",
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the UpSampling2D layer.

        Args:
            size: Row and column repetition factor(s).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            interpolation: Interpolation method string.
            **kwargs: Additional base layer keyword arguments.
        """
        _ = interpolation
        super().__init__(rank=2, size=size, data_format=data_format, **kwargs)


class UpSampling3D(_BaseUpSampling):
    """Upsampling layer for 3D volumetric inputs."""

    def __init__(
        self,
        size: int | Sequence[int] = (2, 2, 2),
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the UpSampling3D layer.

        Args:
            size: Depth, row, and column repetition factor(s).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=3, size=size, data_format=data_format, **kwargs)


class ZeroPadding1D(_BaseZeroPadding):
    """Zero-padding layer for 1D temporal inputs."""

    def __init__(
        self,
        padding: int | Sequence[int | Sequence[int]] = 1,
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the ZeroPadding1D layer.

        Args:
            padding: Temporal zero-padding width specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=1, padding=padding, data_format=data_format, **kwargs)


class ZeroPadding2D(_BaseZeroPadding):
    """Zero-padding layer for 2D spatial inputs."""

    def __init__(
        self,
        padding: int | Sequence[int | Sequence[int]] = (1, 1),
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the ZeroPadding2D layer.

        Args:
            padding: Height and width zero-padding specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=2, padding=padding, data_format=data_format, **kwargs)


class ZeroPadding3D(_BaseZeroPadding):
    """Zero-padding layer for 3D volumetric inputs."""

    def __init__(
        self,
        padding: int | Sequence[int | Sequence[int]] = ((1, 1), (1, 1), (1, 1)),
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the ZeroPadding3D layer.

        Args:
            padding: Depth, height, and width zero-padding specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=3, padding=padding, data_format=data_format, **kwargs)


class Cropping1D(_BaseCropping):
    """Cropping layer for 1D temporal inputs."""

    def __init__(
        self,
        cropping: int | Sequence[int | Sequence[int]] = (1, 1),
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Cropping1D layer.

        Args:
            cropping: Temporal cropping width specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=1, cropping=cropping, data_format=data_format, **kwargs)


class Cropping2D(_BaseCropping):
    """Cropping layer for 2D spatial inputs."""

    def __init__(
        self,
        cropping: int | Sequence[int | Sequence[int]] = ((0, 0), (0, 0)),
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Cropping2D layer.

        Args:
            cropping: Height and width cropping specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=2, cropping=cropping, data_format=data_format, **kwargs)


class Cropping3D(_BaseCropping):
    """Cropping layer for 3D volumetric inputs."""

    def __init__(
        self,
        cropping: int | Sequence[int | Sequence[int]] = ((1, 1), (1, 1), (1, 1)),
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Cropping3D layer.

        Args:
            cropping: Depth, height, and width cropping specification.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Additional base layer keyword arguments.
        """
        super().__init__(rank=3, cropping=cropping, data_format=data_format, **kwargs)

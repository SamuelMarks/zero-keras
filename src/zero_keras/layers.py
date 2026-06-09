"""Keras layers."""

from typing import Any, Optional
from zero_keras import Layer


class Activation(Layer):
    """Applies an activation function to an output."""

    def __init__(
        self,
        activation,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ActivityRegularization(Layer):
    """Layer that applies an update to the cost function based input activity."""

    def __init__(
        self,
        l1: float = 0.0,
        l2: float = 0.0,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Add(Layer):
    """Performs elementwise addition operation."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class AdditiveAttention(Layer):
    """Additive attention layer, a.k.a. Bahdanau-style attention."""

    def __init__(
        self,
        use_scale: bool = True,
        dropout: float = 0.0,
        score_mode="dot",
        seed=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AlphaDropout(Layer):
    """Applies Alpha Dropout to the input."""

    def __init__(
        self,
        rate,
        noise_shape="None",
        seed="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Attention(Layer):
    """Dot-product attention layer, a.k.a. Luong-style attention."""

    def __init__(
        self,
        use_scale: bool = False,
        dropout: float = 0.0,
        seed="None",
        score_mode: str = "dot",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AugMix(Layer):
    """Performs the AugMix data augmentation technique."""

    def __init__(
        self,
        value_range: tuple = (0, 255),
        num_chains: int = 3,
        chain_depth: int = 3,
        factor: float = 0.3,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: str = "bilinear",
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AutoContrast(Layer):
    """Performs the auto-contrast operation on an image."""

    def __init__(
        self,
        value_range: tuple = None,
        _USE_BASE_FACTOR: bool = False,
        _VALUE_RANGE_VALIDATION_ERROR: str = "The value_range argument should be a list of two numbers. ",
        factor=None,
        bounding_box_format=None,
        data_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Average(Layer):
    """Averages a list of inputs element-wise.."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class AveragePooling1D(Layer):
    """Average pooling for temporal data."""

    def __init__(
        self,
        pool_size,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AveragePooling2D(Layer):
    """Average pooling operation for 2D spatial data."""

    def __init__(
        self,
        pool_size,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AveragePooling3D(Layer):
    """Average pooling operation for 3D data (spatial or spatio-temporal)."""

    def __init__(
        self,
        pool_size,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AvgPool1D(Layer):
    """Average pooling for temporal data."""

    def __init__(
        self,
        pool_size,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AvgPool2D(Layer):
    """Average pooling operation for 2D spatial data."""

    def __init__(
        self,
        pool_size,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class AvgPool3D(Layer):
    """Average pooling operation for 3D data (spatial or spatio-temporal)."""

    def __init__(
        self,
        pool_size,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class BatchNormalization(Layer):
    """Layer that normalizes its inputs."""

    def __init__(
        self,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
        moving_mean_initializer: str = "zeros",
        moving_variance_initializer: str = "ones",
        beta_regularizer="None",
        gamma_regularizer="None",
        beta_constraint="None",
        gamma_constraint="None",
        synchronized: bool = False,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Bidirectional(Layer):
    """Bidirectional wrapper for RNNs."""

    def __init__(
        self,
        layer,
        merge_mode: str = "concat",
        backward_layer="None",
        weights="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class CategoryEncoding(Layer):
    """A preprocessing layer which encodes integer features."""

    def __init__(
        self,
        num_tokens="None",
        output_mode: str = "multi_hot",
        sparse: bool = False,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class CenterCrop(Layer):
    """A preprocessing layer which crops images."""

    def __init__(
        self,
        height,
        width,
        data_format: str = None,
        _USE_BASE_FACTOR: bool = False,
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Concatenate(Layer):
    """Concatenates a list of inputs."""

    def __init__(
        self,
        axis: int = -1,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Conv1D(Layer):
    """1D convolution layer (e.g. temporal convolution)."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        groups: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        lora_rank=None,
        lora_alpha=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Conv1DTranspose(Layer):
    """1D transposed convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        output_padding=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Conv2D(Layer):
    """2D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1),
        groups: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        lora_rank=None,
        lora_alpha=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Conv2DTranspose(Layer):
    """2D transposed convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1),
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        output_padding=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Conv3D(Layer):
    """3D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1, 1),
        groups: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        lora_rank=None,
        lora_alpha=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Conv3DTranspose(Layer):
    """3D transposed convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1, 1),
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        output_padding=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ConvLSTM1D(Layer):
    """1D Convolutional LSTM."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed="None",
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: str = "",
        stateful: bool = False,
        unroll: str = "",
        rank=None,
        cell=None,
        zero_output_for_mask=False,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ConvLSTM2D(Layer):
    """2D Convolutional LSTM."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed="None",
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: str = "",
        stateful: bool = False,
        unroll: str = "",
        rank=None,
        cell=None,
        zero_output_for_mask=False,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ConvLSTM3D(Layer):
    """3D Convolutional LSTM."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed="None",
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: str = "",
        stateful: bool = False,
        unroll: str = "",
        rank=None,
        cell=None,
        zero_output_for_mask=False,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Convolution1D(Layer):
    """1D convolution layer (e.g. temporal convolution)."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        groups: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        lora_rank=None,
        lora_alpha=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Convolution1DTranspose(Layer):
    """1D transposed convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        output_padding=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Convolution2D(Layer):
    """2D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1),
        groups: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        lora_rank=None,
        lora_alpha=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Convolution2DTranspose(Layer):
    """2D transposed convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1),
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        output_padding=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Convolution3D(Layer):
    """3D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1, 1),
        groups: int = 1,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        lora_rank=None,
        lora_alpha=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Convolution3DTranspose(Layer):
    """3D transposed convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1, 1),
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        rank=None,
        output_padding=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Cropping1D(Layer):
    """Cropping layer for 1D input (e.g. temporal sequence)."""

    def __init__(
        self,
        cropping: tuple = (1, 1),
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Cropping2D(Layer):
    """Cropping layer for 2D input (e.g. picture)."""

    def __init__(
        self,
        cropping: tuple = ((0, 0), (0, 0)),
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Cropping3D(Layer):
    """Cropping layer for 3D data (e.g. spatial or spatio-temporal)."""

    def __init__(
        self,
        cropping: tuple = ((1, 1), (1, 1), (1, 1)),
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class CutMix(Layer):
    """CutMix data augmentation technique."""

    def __init__(
        self,
        factor: int = 1,
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Dense(Layer):
    """Just your regular densely-connected NN layer."""

    def __init__(
        self,
        units,
        activation="None",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        lora_rank="None",
        lora_alpha="None",
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class DepthwiseConv1D(Layer):
    """1D depthwise convolution layer."""

    def __init__(
        self,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        depth_multiplier: int = 1,
        data_format: str = None,
        dilation_rate: int = 1,
        activation="None",
        use_bias: bool = True,
        depthwise_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        depthwise_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        depthwise_constraint="None",
        bias_constraint="None",
        rank=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class DepthwiseConv2D(Layer):
    """2D depthwise convolution layer."""

    def __init__(
        self,
        kernel_size,
        strides: tuple = (1, 1),
        padding: str = "valid",
        depth_multiplier: int = 1,
        data_format: str = None,
        dilation_rate: tuple = (1, 1),
        activation="None",
        use_bias: bool = True,
        depthwise_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        depthwise_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        depthwise_constraint="None",
        bias_constraint="None",
        rank=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Discretization(Layer):
    """A preprocessing layer which buckets continuous features by ranges."""

    def __init__(
        self,
        bin_boundaries="None",
        num_bins="None",
        epsilon: float = 0.01,
        output_mode: str = "int",
        sparse: bool = False,
        dtype="None",
        name="None",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Dot(Layer):
    """Computes element-wise dot product of two tensors."""

    def __init__(
        self,
        axes,
        normalize: bool = False,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Dropout(Layer):
    """Applies dropout to the input."""

    def __init__(
        self,
        rate,
        noise_shape="None",
        seed="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ELU(Layer):
    """Applies an Exponential Linear Unit function to an output."""

    def __init__(
        self,
        alpha: float = 1.0,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class EinsumDense(Layer):
    """A layer that uses `einsum` as the backing computation."""

    def __init__(
        self,
        equation,
        output_shape,
        activation="None",
        bias_axes="None",
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        lora_rank="None",
        lora_alpha="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Embedding(Layer):
    """Turns nonnegative integers (indexes) into dense vectors of fixed size."""

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer: str = "uniform",
        embeddings_regularizer="None",
        embeddings_constraint="None",
        mask_zero: bool = False,
        weights="None",
        lora_rank="None",
        lora_alpha="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Equalization(Layer):
    """Preprocessing layer for histogram equalization on image channels."""

    def __init__(
        self,
        value_range: Optional[str] = None,
        bins: str = None,
        data_format="None",
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size."""

    def __init__(
        self,
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class FlaxLayer(Layer):
    """Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module."""

    def __init__(
        self,
        module,
        method="None",
        variables="None",
        call_fn=None,
        init_fn=None,
        params=None,
        state=None,
        seed=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GRU(Layer):
    """Gated Recurrent Unit - Cho et al. 2014."""

    def __init__(
        self,
        units,
        activation: str = None,
        recurrent_activation: str = 'sigmoid (sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).',
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: int = 0,
        recurrent_dropout: int = 0,
        seed="None",
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: str = "",
        unroll: str = "",
        reset_after: bool = True,
        use_cudnn: str = "auto",
        cell=None,
        zero_output_for_mask=False,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GRUCell(Layer):
    """Cell class for the GRU layer."""

    def __init__(
        self,
        units,
        activation: str = None,
        recurrent_activation: str = 'sigmoid (sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).',
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: int = 0,
        recurrent_dropout: int = 0,
        reset_after: bool = True,
        seed="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GaussianDropout(Layer):
    """Apply multiplicative 1-centered Gaussian noise."""

    def __init__(
        self,
        rate,
        seed="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GaussianNoise(Layer):
    """Apply additive zero-centered Gaussian noise."""

    def __init__(
        self,
        stddev,
        seed="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalAveragePooling1D(Layer):
    """Global average pooling operation for temporal data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalAveragePooling2D(Layer):
    """Global average pooling operation for 2D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalAveragePooling3D(Layer):
    """Global average pooling operation for 3D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalAvgPool1D(Layer):
    """Global average pooling operation for temporal data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalAvgPool2D(Layer):
    """Global average pooling operation for 2D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalAvgPool3D(Layer):
    """Global average pooling operation for 3D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalMaxPool1D(Layer):
    """Global max pooling operation for temporal data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalMaxPool2D(Layer):
    """Global max pooling operation for 2D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalMaxPool3D(Layer):
    """Global max pooling operation for 3D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalMaxPooling1D(Layer):
    """Global max pooling operation for temporal data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalMaxPooling2D(Layer):
    """Global max pooling operation for 2D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GlobalMaxPooling3D(Layer):
    """Global max pooling operation for 3D data."""

    def __init__(
        self,
        data_format: str = None,
        keepdims: bool = False,
        pool_dimensions=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GroupNormalization(Layer):
    """Group normalization layer."""

    def __init__(
        self,
        groups: int = 32,
        axis: float = -1.0,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
        beta_regularizer="None",
        gamma_regularizer="None",
        beta_constraint="None",
        gamma_constraint="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class GroupQueryAttention(Layer):
    """Grouped Query Attention layer."""

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        dropout: float = 0.0,
        use_bias: bool = True,
        flash_attention="None",
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        seed="None",
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class HashedCrossing(Layer):
    """A preprocessing layer which crosses features using the "hashing trick"."""

    def __init__(
        self,
        num_bins,
        output_mode: str = "int",
        sparse: bool = False,
        dtype="None",
        name="None",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Hashing(Layer):
    """A preprocessing layer which hashes and bins categorical features."""

    def __init__(
        self,
        num_bins,
        mask_value="None",
        salt="None",
        output_mode: str = "int",
        sparse: bool = False,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Identity(Layer):
    """Identity layer."""

    def __init__(
        self,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class InputLayer(Layer):
    """This is the class from which all layers inherit."""

    def __init__(
        self,
        shape="None",
        batch_size="None",
        dtype="None",
        sparse="None",
        ragged="None",
        batch_shape="None",
        input_tensor="None",
        optional: bool = False,
        name="None",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class InputSpec(Layer):
    """Specifies the rank, dtype and shape of every input to a layer."""

    def __init__(
        self,
        dtype="None",
        shape="None",
        ndim="None",
        max_ndim="None",
        min_ndim="None",
        axes="None",
        allow_last_axis_squeeze: bool = False,
        name="None",
        optional: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class IntegerLookup(Layer):
    """A preprocessing layer that maps integers to (possibly encoded) indices."""

    def __init__(
        self,
        max_tokens="None",
        num_oov_indices: int = 1,
        mask_token="None",
        oov_token: float = -1.0,
        vocabulary="None",
        vocabulary_dtype: str = "int64",
        idf_weights="None",
        invert: bool = False,
        output_mode: str = "int",
        pad_to_max_tokens: bool = False,
        sparse: bool = False,
        name="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class JaxLayer(Layer):
    """Keras Layer that wraps a JAX model."""

    def __init__(
        self,
        call_fn,
        init_fn="None",
        seed="None",
        params="None",
        state="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class LSTM(Layer):
    """Long Short-Term Memory layer - Hochreiter 1997."""

    def __init__(
        self,
        units,
        activation: str = None,
        recurrent_activation: str = 'sigmoid (sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).',
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: int = 0,
        recurrent_dropout: int = 0,
        seed="None",
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: str = "",
        stateful: str = "",
        unroll: bool = False,
        use_cudnn: str = "auto",
        cell=None,
        zero_output_for_mask=False,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class LSTMCell(Layer):
    """Cell class for the LSTM layer."""

    def __init__(
        self,
        units,
        activation: str = None,
        recurrent_activation: str = 'sigmoid (sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).',
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: int = 0,
        recurrent_dropout: int = 0,
        seed="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Lambda(Layer):
    """Wraps arbitrary expressions as a `Layer` object."""

    def __init__(
        self,
        function,
        output_shape="None",
        mask="None",
        arguments="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class LayerNormalization(Layer):
    """Layer normalization layer (Ba et al., 2016)."""

    def __init__(
        self,
        axis: float = -1.0,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        rms_scaling: bool = False,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
        beta_regularizer="None",
        gamma_regularizer="None",
        beta_constraint="None",
        gamma_constraint="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit activation layer."""

    def __init__(
        self,
        negative_slope: float = 0.3,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Masking(Layer):
    """Masks a sequence by using a mask value to skip timesteps."""

    def __init__(
        self,
        mask_value: float = 0.0,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MaxNumBoundingBoxes(Layer):
    """Ensure the maximum number of bounding boxes."""

    def __init__(
        self,
        max_number,
        padding_value: float = -1.0,
        fill_value: int = -1,
        factor=None,
        bounding_box_format=None,
        data_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MaxPool1D(Layer):
    """Max pooling operation for 1D temporal data."""

    def __init__(
        self,
        pool_size: int = 2,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MaxPool2D(Layer):
    """Max pooling operation for 2D spatial data."""

    def __init__(
        self,
        pool_size: tuple = (2, 2),
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MaxPool3D(Layer):
    """Max pooling operation for 3D data (spatial or spatio-temporal)."""

    def __init__(
        self,
        pool_size: tuple = (2, 2, 2),
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MaxPooling1D(Layer):
    """Max pooling operation for 1D temporal data."""

    def __init__(
        self,
        pool_size: int = 2,
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MaxPooling2D(Layer):
    """Max pooling operation for 2D spatial data."""

    def __init__(
        self,
        pool_size: tuple = (2, 2),
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MaxPooling3D(Layer):
    """Max pooling operation for 3D data (spatial or spatio-temporal)."""

    def __init__(
        self,
        pool_size: tuple = (2, 2, 2),
        strides="None",
        padding: str = "valid",
        data_format: str = None,
        name="None",
        pool_dimensions=None,
        pool_mode="max",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Maximum(Layer):
    """Computes element-wise maximum on a list of inputs."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class MelSpectrogram(Layer):
    """A preprocessing layer to convert raw audio signals to Mel spectrograms."""

    def __init__(
        self,
        fft_length: int = 2048,
        sequence_stride: int = 512,
        sequence_length: str = "fft_length",
        window: str = "hann",
        sampling_rate: int = 16000,
        num_mel_bins: int = 128,
        min_freq: float = 20.0,
        max_freq: str = "sampling_rate / 2",
        power_to_db: bool = True,
        top_db: float = 80.0,
        mag_exp: float = 2.0,
        ref_power: float = 1.0,
        min_power: float = 1e-10,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Minimum(Layer):
    """Computes elementwise minimum on a list of inputs."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class MixUp(Layer):
    """MixUp implements the MixUp data augmentation technique."""

    def __init__(
        self,
        alpha: str = 0.2,
        seed="None",
        data_format="None",
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class MultiHeadAttention(Layer):
    """MultiHeadAttention layer."""

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim="None",
        dropout: float = 0.0,
        use_bias: bool = True,
        output_shape="None",
        attention_axes="None",
        flash_attention="None",
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        bias_constraint="None",
        seed="None",
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Multiply(Layer):
    """Performs elementwise multiplication."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class Normalization(Layer):
    """A preprocessing layer that normalizes continuous features."""

    def __init__(
        self,
        axis: float = -1.0,
        mean="None",
        variance="None",
        invert: bool = False,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class PReLU(Layer):
    """Parametric Rectified Linear Unit activation layer."""

    def __init__(
        self,
        alpha_initializer: str = "Zeros",
        alpha_regularizer="None",
        alpha_constraint="None",
        shared_axes="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern."""

    def __init__(
        self,
        dims,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Pipeline(Layer):
    """Applies a series of layers to an input."""

    def __init__(self, layers, name="None", **kwargs: Any):
        super().__init__(**kwargs)


class RMSNormalization(Layer):
    """Root Mean Square (RMS) Normalization layer."""

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-06,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RNN(Layer):
    """Base class for recurrent layers."""

    def __init__(
        self,
        cell,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        zero_output_for_mask: bool = False,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandAugment(Layer):
    """RandAugment performs the Rand Augment operation on input images."""

    def __init__(
        self,
        value_range: tuple = (0, 255),
        num_ops: int = 2,
        factor: float = 0.5,
        interpolation: str = "bilinear",
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        _AUGMENT_LAYERS: list = [
            "random_shear",
            "random_translation",
            "random_rotation",
            "random_brightness",
            "random_color_degeneration",
            "random_contrast",
            "random_sharpness",
            "random_posterization",
            "solarization",
            "auto_contrast",
            "equalization",
        ],
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomBrightness(Layer):
    """A preprocessing layer which randomly adjusts brightness during training."""

    def __init__(
        self,
        factor,
        value_range: Optional[str] = None,
        seed="None",
        _VALUE_RANGE_VALIDATION_ERROR: str = "The value_range argument should be a list of two numbers. ",
        bounding_box_format=None,
        data_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomColorDegeneration(Layer):
    """Randomly performs the color degeneration operation on given images."""

    def __init__(
        self,
        factor,
        seed="None",
        _VALUE_RANGE_VALIDATION_ERROR: str = "The value_range argument should be a list of two numbers. ",
        data_format="None",
        value_range: tuple = (0, 255),
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomColorJitter(Layer):
    """RandomColorJitter class randomly apply brightness, contrast, saturation and hue image processing ..."""

    def __init__(
        self,
        value_range: tuple = (0, 255),
        brightness_factor="None",
        contrast_factor="None",
        saturation_factor="None",
        hue_factor="None",
        seed="None",
        data_format="None",
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomContrast(Layer):
    """A preprocessing layer which randomly adjusts contrast during training."""

    def __init__(
        self,
        factor,
        value_range: tuple = (0, 255),
        seed="None",
        _FACTOR_BOUNDS: tuple = (0, 1),
        bounding_box_format=None,
        data_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomCrop(Layer):
    """A preprocessing layer which randomly crops images during training."""

    def __init__(
        self,
        height,
        width,
        seed="None",
        data_format="None",
        name="None",
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomElasticTransform(Layer):
    """A preprocessing layer that applies random elastic transformations."""

    def __init__(
        self,
        factor: float = 1.0,
        scale: float = 1.0,
        interpolation: str = "bilinear",
        fill_mode: str = "constant",
        fill_value: float = 0.0,
        value_range: tuple = (0, 255),
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        _SUPPORTED_INTERPOLATION: tuple = ("nearest", "bilinear"),
        _SUPPORTED_FILL_MODES: set = {
            "nearest",
            "mirror",
            "reflect",
            "wrap",
            "constant",
        },
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomErasing(Layer):
    """Random Erasing data augmentation technique."""

    def __init__(
        self,
        factor: float = 1.0,
        scale: tuple = (0.02, 0.33),
        fill_value="None",
        value_range: tuple = (0, 255),
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomFlip(Layer):
    """A preprocessing layer which randomly flips images during training."""

    def __init__(
        self,
        mode: str = "horizontal_and_vertical",
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        data_format="None",
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomGaussianBlur(Layer):
    """Applies random Gaussian blur to images for data augmentation."""

    def __init__(
        self,
        factor: float = 1.0,
        kernel_size: int = 3,
        sigma: float = 1.0,
        value_range: tuple = (0, 255),
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomGrayscale(Layer):
    """Preprocessing layer for random conversion of RGB images to grayscale."""

    def __init__(
        self,
        factor: float = 0.5,
        data_format="None",
        seed="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomHue(Layer):
    """Randomly adjusts the hue on given images."""

    def __init__(
        self,
        factor,
        value_range: tuple = (0, 255),
        seed="None",
        _USE_BASE_FACTOR: bool = True,
        _FACTOR_BOUNDS: tuple = (0, 1),
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomInvert(Layer):
    """Preprocessing layer for random inversion of image colors."""

    def __init__(
        self,
        factor: str = None,
        value_range: tuple = None,
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomPerspective(Layer):
    """A preprocessing layer that applies random perspective transformations."""

    def __init__(
        self,
        factor: float = 1.0,
        scale: float = 1.0,
        interpolation: str = "bilinear",
        fill_value: float = 0.0,
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        _SUPPORTED_INTERPOLATION: tuple = ("nearest", "bilinear"),
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomPosterization(Layer):
    """Reduces the number of bits for each color channel."""

    def __init__(
        self,
        value_range: tuple = None,
        factor="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (1, 8),
        _MAX_FACTOR: int = 8,
        _VALUE_RANGE_VALIDATION_ERROR: str = "The value_range argument should be a list of two numbers. ",
        data_format="None",
        seed="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomRotation(Layer):
    """A preprocessing layer which randomly rotates images during training."""

    def __init__(
        self,
        factor,
        fill_mode: str = "reflect",
        interpolation: str = "bilinear",
        seed="None",
        fill_value: float = 0.0,
        data_format: str = None,
        _SUPPORTED_FILL_MODE: tuple = ("reflect", "wrap", "constant", "nearest"),
        _SUPPORTED_INTERPOLATION: tuple = ("nearest", "bilinear"),
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomSaturation(Layer):
    """Randomly adjusts the saturation on given images."""

    def __init__(
        self,
        factor,
        value_range: tuple = (0, 255),
        seed="None",
        _VALUE_RANGE_VALIDATION_ERROR: str = "The value_range argument should be a list of two numbers. ",
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomSharpness(Layer):
    """Randomly performs the sharpness operation on given images."""

    def __init__(
        self,
        factor,
        value_range: tuple = (0, 255),
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        _VALUE_RANGE_VALIDATION_ERROR: str = "The value_range argument should be a list of two numbers. ",
        data_format="None",
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomShear(Layer):
    """A preprocessing layer that randomly applies shear transformations to images."""

    def __init__(
        self,
        x_factor: float = 0.0,
        y_factor: float = 0.0,
        interpolation: str = "bilinear",
        fill_mode: str = "constant",
        fill_value: float = 0.0,
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_BOUNDS: tuple = (0, 1),
        _FACTOR_VALIDATION_ERROR: str = "The factor argument should be a number (or a list of two numbers) in the range [0, 1.0]. ",
        _SUPPORTED_FILL_MODE: tuple = ("reflect", "wrap", "constant", "nearest"),
        _SUPPORTED_INTERPOLATION: tuple = ("nearest", "bilinear"),
        data_format="None",
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomTranslation(Layer):
    """A preprocessing layer which randomly translates images during training."""

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode: str = "constant",
        interpolation: str = "bilinear",
        seed="None",
        fill_value: float = 0.0,
        data_format: str = None,
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_VALIDATION_ERROR: str = "The factor argument should be a number (or a list of two numbers) in the range [-1.0, 1.0]. ",
        _SUPPORTED_FILL_MODE: tuple = ("reflect", "wrap", "constant", "nearest"),
        _SUPPORTED_INTERPOLATION: tuple = ("nearest", "bilinear"),
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RandomZoom(Layer):
    """A preprocessing layer which randomly zooms images during training."""

    def __init__(
        self,
        height_factor,
        width_factor="None",
        fill_mode: str = "reflect",
        interpolation: str = "bilinear",
        seed="None",
        fill_value: float = 0.0,
        data_format: str = None,
        _USE_BASE_FACTOR: bool = False,
        _FACTOR_VALIDATION_ERROR: str = "The height_factor and width_factor arguments should be a number (or a list of two numbers) in the range [-1.0, 1.0]. ",
        _SUPPORTED_FILL_MODE: tuple = ("reflect", "wrap", "constant", "nearest"),
        _SUPPORTED_INTERPOLATION: tuple = ("nearest", "bilinear"),
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ReLU(Layer):
    """Rectified Linear Unit activation function layer."""

    def __init__(
        self,
        max_value="None",
        negative_slope: float = 0.0,
        threshold: float = 0.0,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class RepeatVector(Layer):
    """Repeats the input n times."""

    def __init__(
        self,
        n,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Rescaling(Layer):
    """A preprocessing layer which rescales input values to a new range."""

    def __init__(
        self,
        scale,
        offset: float = 0.0,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Reshape(Layer):
    """Layer that reshapes inputs into the given shape."""

    def __init__(
        self,
        target_shape,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Resizing(Layer):
    """A preprocessing layer which resizes images."""

    def __init__(
        self,
        height,
        width,
        interpolation: str = "bilinear",
        crop_to_aspect_ratio: bool = False,
        pad_to_aspect_ratio: bool = False,
        fill_mode: str = "constant",
        fill_value: float = 0.0,
        data_format: str = None,
        _USE_BASE_FACTOR: bool = False,
        antialias: bool = False,
        factor=None,
        bounding_box_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class STFTSpectrogram(Layer):
    """Layer to compute the Short-Time Fourier Transform (STFT) on a 1D signal."""

    def __init__(
        self,
        mode: str = "log",
        frame_length: int = 256,
        frame_step: str = None,
        fft_length="None",
        window: str = "hann",
        periodic: bool = False,
        scaling: str = "density",
        padding: str = "valid",
        expand_dims: bool = False,
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SeparableConv1D(Layer):
    """1D separable convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        depth_multiplier: int = 1,
        activation="None",
        use_bias: bool = True,
        depthwise_initializer: str = "glorot_uniform",
        pointwise_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        depthwise_regularizer="None",
        pointwise_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        depthwise_constraint="None",
        pointwise_constraint="None",
        bias_constraint="None",
        rank=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SeparableConv2D(Layer):
    """2D separable convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1),
        depth_multiplier: int = 1,
        activation="None",
        use_bias: bool = True,
        depthwise_initializer: str = "glorot_uniform",
        pointwise_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        depthwise_regularizer="None",
        pointwise_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        depthwise_constraint="None",
        pointwise_constraint="None",
        bias_constraint="None",
        rank=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SeparableConvolution1D(Layer):
    """1D separable convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: int = 1,
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: int = 1,
        depth_multiplier: int = 1,
        activation="None",
        use_bias: bool = True,
        depthwise_initializer: str = "glorot_uniform",
        pointwise_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        depthwise_regularizer="None",
        pointwise_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        depthwise_constraint="None",
        pointwise_constraint="None",
        bias_constraint="None",
        rank=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SeparableConvolution2D(Layer):
    """2D separable convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides: tuple = (1, 1),
        padding: str = "valid",
        data_format: str = None,
        dilation_rate: tuple = (1, 1),
        depth_multiplier: int = 1,
        activation="None",
        use_bias: bool = True,
        depthwise_initializer: str = "glorot_uniform",
        pointwise_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        depthwise_regularizer="None",
        pointwise_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        depthwise_constraint="None",
        pointwise_constraint="None",
        bias_constraint="None",
        rank=None,
        trainable=True,
        name=None,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SimpleRNN(Layer):
    """Fully-connected RNN where the output is to be fed back as the new input."""

    def __init__(
        self,
        units,
        activation: str = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        activity_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: int = 0,
        recurrent_dropout: int = 0,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: str = "",
        stateful: str = "",
        unroll: str = "",
        seed="None",
        cell=None,
        zero_output_for_mask=False,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SimpleRNNCell(Layer):
    """Cell class for SimpleRNN."""

    def __init__(
        self,
        units,
        activation: str = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        kernel_regularizer="None",
        recurrent_regularizer="None",
        bias_regularizer="None",
        kernel_constraint="None",
        recurrent_constraint="None",
        bias_constraint="None",
        dropout: int = 0,
        recurrent_dropout: int = 0,
        seed="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Softmax(Layer):
    """Softmax activation layer."""

    def __init__(
        self,
        axis: int = -1,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Solarization(Layer):
    """Applies `(max_value - pixel + min_value)` for each pixel in the image."""

    def __init__(
        self,
        addition_factor: Optional[float] = 0.0,
        threshold_factor: float = 0.0,
        value_range: tuple = (0, 255),
        seed="None",
        _USE_BASE_FACTOR: bool = False,
        _VALUE_RANGE_VALIDATION_ERROR: str = "The value_range argument should be a list of two numbers. ",
        _FACTOR_VALIDATION_ERROR: str = "The addition_factor and threshold_factor arguments should be a number (or a list of two numbers) in the range [0, 1]. ",
        factor=None,
        bounding_box_format=None,
        data_format=None,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SpatialDropout1D(Layer):
    """Spatial 1D version of Dropout."""

    def __init__(self, rate, seed="None", name="None", dtype="None", **kwargs: Any):
        super().__init__(**kwargs)


class SpatialDropout2D(Layer):
    """Spatial 2D version of Dropout."""

    def __init__(
        self,
        rate,
        data_format: str = None,
        seed="None",
        dtype="None",
        name="None",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SpatialDropout3D(Layer):
    """Spatial 3D version of Dropout."""

    def __init__(
        self,
        rate,
        data_format: str = None,
        seed="None",
        dtype="None",
        name="None",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class SpectralNormalization(Layer):
    """Performs spectral normalization on the weights of a target layer."""

    def __init__(
        self,
        layer,
        power_iterations: int = 1,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class StackedRNNCells(Layer):
    """Wrapper allowing a stack of RNN cells to behave as a single cell."""

    def __init__(
        self,
        cells,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class StringLookup(Layer):
    """A preprocessing layer that maps strings to (possibly encoded) indices."""

    def __init__(
        self,
        max_tokens="None",
        num_oov_indices: int = 1,
        mask_token="None",
        oov_token: str = '"[UNK]"',
        vocabulary="None",
        vocabulary_dtype: str = "int64",
        idf_weights="None",
        invert: bool = False,
        output_mode: str = "int",
        pad_to_max_tokens: bool = False,
        sparse: bool = False,
        encoding: Optional[str] = "utf-8",
        name="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Subtract(Layer):
    """Performs elementwise subtraction."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class TFSMLayer(Layer):
    """Reload a Keras model/layer that was saved via SavedModel / ExportArchive."""

    def __init__(
        self,
        filepath,
        call_endpoint: str = "serve",
        call_training_endpoint="None",
        trainable: bool = True,
        name="None",
        dtype="None",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class TextVectorization(Layer):
    """A preprocessing layer which maps text features to integer sequences."""

    def __init__(
        self,
        max_tokens="None",
        standardize: str = "lower_and_strip_punctuation",
        split: str = "whitespace",
        ngrams="None",
        output_mode: str = "int",
        output_sequence_length="None",
        pad_to_max_tokens: bool = False,
        vocabulary="None",
        idf_weights="None",
        ragged: bool = False,
        sparse: bool = False,
        encoding: Optional[str] = "utf-8",
        name="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class TimeDistributed(Layer):
    """This wrapper allows to apply a layer to every temporal slice of an input."""

    def __init__(
        self,
        layer,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class TorchModuleWrapper(Layer):
    """Torch module wrapper layer."""

    def __init__(
        self,
        module,
        output_shape="None",
        name="None",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class UnitNormalization(Layer):
    """Unit normalization layer."""

    def __init__(
        self,
        axis: float = -1.0,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class UpSampling1D(Layer):
    """Upsampling layer for 1D inputs."""

    def __init__(
        self,
        size: int = 2,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs."""

    def __init__(
        self,
        size: tuple = (2, 2),
        data_format: str = "channels_last",
        interpolation: str = "nearest",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class UpSampling3D(Layer):
    """Upsampling layer for 3D inputs."""

    def __init__(
        self,
        size: tuple = (2, 2, 2),
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class Wrapper(Layer):
    """Abstract wrapper base class."""

    def __init__(
        self,
        layer,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence)."""

    def __init__(
        self,
        padding: int = 1,
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ZeroPadding2D(Layer):
    """Zero-padding layer for 2D input (e.g. picture)."""

    def __init__(
        self,
        padding: tuple = (1, 1),
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)


class ZeroPadding3D(Layer):
    """Zero-padding layer for 3D data (spatial or spatio-temporal)."""

    def __init__(
        self,
        padding: tuple = ((1, 1), (1, 1), (1, 1)),
        data_format: str = "channels_last",
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

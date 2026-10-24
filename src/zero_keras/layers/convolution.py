"""Convolutional layers (1D, 2D, 3D, Depthwise, Separable, Transpose) for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.activations import get as get_activation
from zero_keras.core_layers import Layer


class Conv1D(Layer):
    """1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved with the layer
    input over a single spatial (or temporal) dimension to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the
    outputs. Finally, if `activation` is not `None`, it is applied to the
    outputs as well.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 1 integer, specifying the size of the
            convolution window.
        strides: int or tuple/list of 1 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, `"valid"`, `"same"` or `"causal"`(case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
            `"causal"` results in causal(dilated) convolutions, e.g. `output[t]`
            does not depend on`input[t+1:]`. Useful when modeling temporal data
            where the model should not violate the temporal order.
            See [WaveNet: A Generative Model for Raw Audio, section2.1](
            https://arxiv.org/abs/1609.03499).
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated convolution.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters // groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape: `(batch_shape, steps, channels)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape: `(batch_shape, channels, steps)`

    Output shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape: `(batch_shape, new_steps, filters)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape: `(batch_shape, filters, new_steps)`

    Returns:
        A 3D tensor representing `activation(conv1d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:
    >>> # The inputs are 128-length vectors with 10 timesteps, and the
    >>> # batch size is 4.
    >>> x = np.random.rand(4, 10, 128)
    >>> y = keras.layers.Conv1D(32, 3, activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 32)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        groups: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        kernel_regularizer: TensorLike = None,
        bias_regularizer: TensorLike = None,
        activity_regularizer: TensorLike = None,
        kernel_constraint: TensorLike = None,
        bias_constraint: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            groups: Description.
            activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            bias_initializer: Description.
            kernel_regularizer: Description.
            bias_regularizer: Description.
            activity_regularizer: Description.
            kernel_constraint: Description.
            bias_constraint: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.filters = filters
        self.kernel_size = (
            (kernel_size,) * 1 if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides,) * 1 if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) * 1 if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
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

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_first":
            # N C W/HW/DHW -> N W/HW/DHW C
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            x = ops.permute(x, perm)

        # Apply conv
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=None,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=((0, 2, 1), (2, 1, 0), (0, 2, 1)),
        )
        out = ops.conv1d(x, self.kernel, config_obj)

        if self.use_bias:
            out = ops.add(out, self.bias)

        if self.data_format == "channels_first":
            # N W/HW/DHW C -> N C W/HW/DHW
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class Conv1DTranspose(Layer):
    """1D transposed convolution layer.

    The need for transposed convolutions generally arise from the desire to use
    a transformation going in the opposite direction of a normal convolution,
    i.e., from something that has the shape of the output of some convolution
    to something that has the shape of its input while maintaining a
    connectivity pattern that is compatible with said convolution.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the transpose convolution).
        kernel_size: int or tuple/list of 1 integer, specifying the size of the
            transposed convolution window.
        strides: int or tuple/list of 1 integer, specifying the stride length
            of the transposed convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated transposed convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape: `(batch_shape, steps, channels)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape: `(batch_shape, channels, steps)`

    Output shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape: `(batch_shape, new_steps, filters)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape: `(batch_shape, filters, new_steps)`

    Returns:
        A 3D tensor representing
        `activation(conv1d_transpose(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    References:
    - [A guide to convolution arithmetic for deep learning](
        https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional Networks](
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)

    Example:
    >>> x = np.random.rand(4, 10, 128)
    >>> y = keras.layers.Conv1DTranspose(32, 3, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 21, 32)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        output_padding: TensorLike = None,
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        kernel_regularizer: TensorLike = None,
        bias_regularizer: TensorLike = None,
        activity_regularizer: TensorLike = None,
        kernel_constraint: TensorLike = None,
        bias_constraint: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            output_padding: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            bias_initializer: Description.
            kernel_regularizer: Description.
            bias_regularizer: Description.
            activity_regularizer: Description.
            kernel_constraint: Description.
            bias_constraint: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.filters = filters
        self.kernel_size = (
            (kernel_size,) * 1 if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides,) * 1 if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) * 1 if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
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

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_channel)

        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
            inputs: Input parameter ``inputs``.
            args: Input parameter ``args``.
            kwargs: Input parameter ``kwargs``.
        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_last":
            # N W/HW/DHW C -> N C W/HW/DHW
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            x = ops.permute(x, perm)

        # Kernel is (spatial..., O, I). compiler expects (I, O, spatial...)
        k_perm = (self.rank + 1, self.rank) + tuple(range(self.rank))
        k = ops.permute(self.kernel, k_perm)

        out = ops.conv1d_transpose(x, k, strides=self.strides, padding=self.padding.upper())

        # Add bias / broadcast
        if self.data_format == "channels_first":
            # transpose to channels last for easy bias broadcast
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias:
                out = ops.add(out, self.bias)
            # transpose back
            perm_back = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm_back)
        else:
            # already channels_last from the perspective of what we need to output
            # wait, `conv_transpose` outputs `channels_first` (N C W H)
            # so we must transpose it to channels_last for the final output
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias:
                out = ops.add(out, self.bias)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class Conv2D(Layer):
    """2D convolution layer.

    This layer creates a convolution kernel that is convolved with the layer
    input over a 2D spatial (or temporal) dimension (height and width) to
    produce a tensor of outputs. If `use_bias` is True, a bias vector is created
    and added to the outputs. Finally, if `activation` is not `None`, it is
    applied to the outputs as well.

    Note on numerical precision: While in general Keras operation execution
    results are identical across backends up to 1e-7 precision in float32,
    `Conv2D` operations may show larger variations. Due to the large
    number of element-wise multiplications and additions in convolution
    operations, especially with large inputs or kernel sizes, accumulated
    floating-point differences can exceed this 1e-7 threshold. These variations
    are particularly noticeable when using different backends (e.g., TensorFlow
    vs JAX) or different hardware.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 2 integer, specifying the size of the
            convolution window.
        strides: int or tuple/list of 2 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters // groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`

    Returns:
        A 4D tensor representing `activation(conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:
    >>> x = np.random.rand(4, 10, 10, 128)
    >>> y = keras.layers.Conv2D(32, 3, activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 8, 32)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        groups: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        kernel_regularizer: TensorLike = None,
        bias_regularizer: TensorLike = None,
        activity_regularizer: TensorLike = None,
        kernel_constraint: TensorLike = None,
        bias_constraint: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            groups: Description.
            activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            bias_initializer: Description.
            kernel_regularizer: Description.
            bias_regularizer: Description.
            activity_regularizer: Description.
            kernel_constraint: Description.
            bias_constraint: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = (
            (kernel_size,) * 2 if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides,) * 2 if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) * 2 if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
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

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_first":
            # N C W/HW/DHW -> N W/HW/DHW C
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            x = ops.permute(x, perm)

        # Apply conv
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=None,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)),
        )
        out = ops.conv2d(x, self.kernel, config_obj)

        if self.use_bias:
            out = ops.add(out, self.bias)

        if self.data_format == "channels_first":
            # N W/HW/DHW C -> N C W/HW/DHW
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class Conv2DTranspose(Layer):
    """2D transposed convolution layer.

    The need for transposed convolutions generally arise from the desire to use
    a transformation going in the opposite direction of a normal convolution,
    i.e., from something that has the shape of the output of some convolution
    to something that has the shape of its input while maintaining a
    connectivity pattern that is compatible with said convolution.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the transposed convolution).
        kernel_size: int or tuple/list of 1 integer, specifying the size of the
            transposed convolution window.
        strides: int or tuple/list of 1 integer, specifying the stride length
            of the transposed convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated transposed convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`

    Returns:
        A 4D tensor representing
        `activation(conv2d_transpose(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    References:
    - [A guide to convolution arithmetic for deep learning](
        https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional Networks](
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)

    Example:
    >>> x = np.random.rand(4, 10, 8, 128)
    >>> y = keras.layers.Conv2DTranspose(32, 2, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 20, 16, 32)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        output_padding: TensorLike = None,
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        kernel_regularizer: TensorLike = None,
        bias_regularizer: TensorLike = None,
        activity_regularizer: TensorLike = None,
        kernel_constraint: TensorLike = None,
        bias_constraint: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            output_padding: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            bias_initializer: Description.
            kernel_regularizer: Description.
            bias_regularizer: Description.
            activity_regularizer: Description.
            kernel_constraint: Description.
            bias_constraint: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = (
            (kernel_size,) * 2 if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides,) * 2 if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) * 2 if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
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

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_channel)

        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
            inputs: Input parameter ``inputs``.
            args: Input parameter ``args``.
            kwargs: Input parameter ``kwargs``.
        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_last":
            # N W/HW/DHW C -> N C W/HW/DHW
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            x = ops.permute(x, perm)

        # Kernel is (spatial..., O, I). compiler expects (I, O, spatial...)
        k_perm = (self.rank + 1, self.rank) + tuple(range(self.rank))
        k = ops.permute(self.kernel, k_perm)

        out = ops.conv2d_transpose(x, k, strides=self.strides, padding=self.padding.upper())

        # Add bias / broadcast
        if self.data_format == "channels_first":
            # transpose to channels last for easy bias broadcast
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias:
                out = ops.add(out, self.bias)
            # transpose back
            perm_back = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm_back)
        else:
            # already channels_last from the perspective of what we need to output
            # wait, `conv_transpose` outputs `channels_first` (N C W H)
            # so we must transpose it to channels_last for the final output
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias:
                out = ops.add(out, self.bias)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class Conv3D(Layer):
    """3D convolution layer.

    This layer creates a convolution kernel that is convolved with the layer
    input over a 3D spatial (or temporal) dimension (width,height and depth) to
    produce a tensor of outputs. If `use_bias` is True, a bias vector is created
    and added to the outputs. Finally, if `activation` is not `None`, it is
    applied to the outputs as well.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 3 integer, specifying the size of the
            convolution window.
        strides: int or tuple/list of 3 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. If you never set it, then it
            will be `"channels_last"`.
        dilation_rate: int or tuple/list of 3 integers, specifying the dilation
            rate to use for dilated convolution.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters // groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:

    - If `data_format="channels_last"`:
        5D tensor with shape:
        `(batch_size, new_spatial_dim1, new_spatial_dim2, new_spatial_dim3,
        filters)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, filters, new_spatial_dim1, new_spatial_dim2,
        new_spatial_dim3)`

    Returns:
        A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:
    >>> x = np.random.rand(4, 10, 10, 10, 128)
    >>> y = keras.layers.Conv3D(32, 3, activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 8, 8, 32)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        groups: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        kernel_regularizer: TensorLike = None,
        bias_regularizer: TensorLike = None,
        activity_regularizer: TensorLike = None,
        kernel_constraint: TensorLike = None,
        bias_constraint: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            groups: Description.
            activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            bias_initializer: Description.
            kernel_regularizer: Description.
            bias_regularizer: Description.
            activity_regularizer: Description.
            kernel_constraint: Description.
            bias_constraint: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 3
        self.filters = filters
        self.kernel_size = (
            (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides,) * 3 if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) * 3 if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
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

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_first":
            # N C W/HW/DHW -> N W/HW/DHW C
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            x = ops.permute(x, perm)

        # Apply conv
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=None,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3)),
        )
        out = ops.conv3d(x, self.kernel, config_obj)

        if self.use_bias:
            out = ops.add(out, self.bias)

        if self.data_format == "channels_first":
            # N W/HW/DHW C -> N C W/HW/DHW
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class Conv3DTranspose(Layer):
    """3D transposed convolution layer.

    The need for transposed convolutions generally arise from the desire to use
    a transformation going in the opposite direction of a normal convolution,
    i.e., from something that has the shape of the output of some convolution
    to something that has the shape of its input while maintaining a
    connectivity pattern that is compatible with said convolution.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the transposed convolution).
        kernel_size: int or tuple/list of 1 integer, specifying the size of the
            transposed convolution window.
        strides: int or tuple/list of 1 integer, specifying the stride length
            of the transposed convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. If you never set it, then it
            will be `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated transposed convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:

    - If `data_format="channels_last"`:
        5D tensor with shape:
        `(batch_size, new_spatial_dim1, new_spatial_dim2, new_spatial_dim3,
        filters)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, filters, new_spatial_dim1, new_spatial_dim2,
        new_spatial_dim3)`

    Returns:
        A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    References:
    - [A guide to convolution arithmetic for deep learning](
        https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional Networks](
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)

    Example:
    >>> x = np.random.rand(4, 10, 8, 12, 128)
    >>> y = keras.layers.Conv3DTranspose(32, 2, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 20, 16, 24, 32)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        output_padding: TensorLike = None,
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        kernel_regularizer: TensorLike = None,
        bias_regularizer: TensorLike = None,
        activity_regularizer: TensorLike = None,
        kernel_constraint: TensorLike = None,
        bias_constraint: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            output_padding: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            bias_initializer: Description.
            kernel_regularizer: Description.
            bias_regularizer: Description.
            activity_regularizer: Description.
            kernel_constraint: Description.
            bias_constraint: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 3
        self.filters = filters
        self.kernel_size = (
            (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides,) * 3 if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) * 3 if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
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

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_channel)

        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name="kernel",
            )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
            inputs: Input parameter ``inputs``.
            args: Input parameter ``args``.
            kwargs: Input parameter ``kwargs``.
        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        x = inputs
        if self.data_format == "channels_last":
            # N W/HW/DHW C -> N C W/HW/DHW
            perm = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            x = ops.permute(x, perm)

        # Kernel is (spatial..., O, I). compiler expects (I, O, spatial...)
        k_perm = (self.rank + 1, self.rank) + tuple(range(self.rank))
        k = ops.permute(self.kernel, k_perm)

        out = ops.conv3d_transpose(x, k, strides=self.strides, padding=self.padding.upper())

        # Add bias / broadcast
        if self.data_format == "channels_first":
            # transpose to channels last for easy bias broadcast
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias:
                out = ops.add(out, self.bias)
            # transpose back
            perm_back = (0, self.rank + 1) + tuple(range(1, self.rank + 1))
            out = ops.permute(out, perm_back)
        else:
            # already channels_last from the perspective of what we need to output
            # wait, `conv_transpose` outputs `channels_first` (N C W H)
            # so we must transpose it to channels_last for the final output
            perm = (0,) + tuple(range(2, self.rank + 2)) + (1,)
            out = ops.permute(out, perm)
            if self.use_bias:
                out = ops.add(out, self.bias)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class DepthwiseConv1D(Layer):
    """1D depthwise convolution layer.

    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular 1D convolution, depthwise convolution does not mix
    information across different input channels.

    The `depth_multiplier` argument determines how many filters are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.

    Args:
        kernel_size: int or tuple/list of 1 integer, specifying the size of the
            depthwise convolution window.
        strides: int or tuple/list of 1 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        depthwise_initializer: Initializer for the convolution kernel.
            If `None`, the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        depthwise_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape: `(batch_shape, steps, channels)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape: `(batch_shape, channels, steps)`

    Output shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape:
        `(batch_shape, new_steps, channels * depth_multiplier)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape:
        `(batch_shape, channels * depth_multiplier, new_steps)`

    Returns:
        A 3D tensor representing
        `activation(depthwise_conv1d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:
    >>> x = np.random.rand(4, 10, 12)
    >>> y = keras.layers.DepthwiseConv1D(3, 3, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 4, 36)

    """

    def __init__(
        self,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        depth_multiplier: TensorLike = 1,
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        depthwise_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            kernel_size: Description.
            strides: Description.
            padding: Description.
            depth_multiplier: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            use_bias: Description.
            depthwise_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.strides = (strides,) if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format or "channels_last"
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]

        kernel_shape = self.kernel_size + (input_channel, self.depth_multiplier)
        self.depthwise_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
        )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(input_channel * self.depth_multiplier,),
                    initializer=self.bias_initializer,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
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

        conv_general_dilated = ops.conv_general_dilated

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=input_channel,
        )
        out = conv_general_dilated(inputs, self.depthwise_kernel, config_obj)

        if self.use_bias:
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


class DepthwiseConv2D(Layer):
    """2D depthwise convolution layer.

    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular 2D convolution, depthwise convolution does not mix
    information across different input channels.

    The `depth_multiplier` argument determines how many filters are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.

    Args:
        kernel_size: int or tuple/list of 2 integer, specifying the size of the
            depthwise convolution window.
        strides: int or tuple/list of 2 integer, specifying the stride length
            of the depthwise convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file
            at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        depthwise_initializer: Initializer for the convolution kernel.
            If `None`, the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        depthwise_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape:
        `(batch_size, new_height, new_width, channels * depth_multiplier)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape:
        `(batch_size, channels * depth_multiplier, new_height, new_width)`

    Returns:
        A 4D tensor representing
        `activation(depthwise_conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:
    >>> x = np.random.rand(4, 10, 10, 12)
    >>> y = keras.layers.DepthwiseConv2D(kernel_size=3, activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 8, 12)

    """

    def __init__(
        self,
        kernel_size: TensorLike,
        strides: TensorLike = (1, 1),
        padding: TensorLike = "valid",
        depth_multiplier: TensorLike = 1,
        data_format: TensorLike = None,
        dilation_rate: TensorLike = (1, 1),
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        depthwise_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            kernel_size: Description.
            strides: Description.
            padding: Description.
            depth_multiplier: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            use_bias: Description.
            depthwise_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides, strides) if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate, dilation_rate)
            if isinstance(dilation_rate, int)
            else tuple(dilation_rate)
        )
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format or "channels_last"
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]

        kernel_shape = self.kernel_size + (input_channel, self.depth_multiplier)
        self.depthwise_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
        )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(input_channel * self.depth_multiplier,),
                    initializer=self.bias_initializer,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
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

        conv_general_dilated = ops.conv_general_dilated

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=input_channel,
        )
        out = conv_general_dilated(inputs, self.depthwise_kernel, config_obj)

        if self.use_bias:
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


class SeparableConv1D(Layer):
    """1D separable convolution layer.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels.
    If `use_bias` is True and a bias initializer is provided,
    it adds a bias vector to the output. It then optionally applies an
    activation function to produce the final output.

    Args:
        filters: int, the dimensionality of the output space (i.e. the number
            of filters in the pointwise convolution).
        kernel_size: int or tuple/list of 1 integers, specifying the size of the
            depthwise convolution window.
        strides: int or tuple/list of 1 integers, specifying the stride length
            of the depthwise convolution. If only one int is specified, the same
            stride size will be used for all dimensions. `strides > 1` is
            incompatible with `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated convolution. If only one int is specified,
            the same dilation rate will be used for all dimensions.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        depthwise_initializer: An initializer for the depthwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        pointwise_initializer: An initializer for the pointwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: An initializer for the bias vector. If None, the
            default initializer ('"zeros"') will be used.
        depthwise_regularizer: Optional regularizer for the depthwise
            convolution kernel.
        pointwise_regularizer: Optional regularizer for the pointwise
            convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            depthwise kernel after being updated by an `Optimizer` (e.g. used
            for norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape).
        pointwise_constraint: Optional projection function to be applied to the
            pointwise kernel after being updated by an `Optimizer`.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape: `(batch_shape, steps, channels)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape: `(batch_shape, channels, steps)`

    Output shape:

    - If `data_format="channels_last"`:
        A 3D tensor with shape: `(batch_shape, new_steps, filters)`
    - If `data_format="channels_first"`:
        A 3D tensor with shape: `(batch_shape, filters, new_steps)`

    Returns:
        A 3D tensor representing
        `activation(separable_conv1d(inputs, kernel) + bias)`.

    Example:
    >>> x = np.random.rand(4, 10, 12)
    >>> y = keras.layers.SeparableConv1D(3, 4, 3, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 4, 4)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        depth_multiplier: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        depthwise_initializer: TensorLike = "glorot_uniform",
        pointwise_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            depth_multiplier: Description.
            activation: Description.
            use_bias: Description.
            depthwise_initializer: Description.
            pointwise_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.rank = 1
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.strides = (strides,) if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate,) if isinstance(dilation_rate, int) else tuple(dilation_rate)
        )
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format or "channels_last"
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer
        self.pointwise_initializer = pointwise_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]

        depthwise_shape = self.kernel_size + (input_channel, self.depth_multiplier)
        pointwise_shape = (1,) + (input_channel * self.depth_multiplier, self.filters)

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

        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
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

        conv_general_dilated = ops.conv_general_dilated

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=input_channel,
        )
        out = conv_general_dilated(inputs, self.depthwise_kernel, config_obj)
        config_obj2 = ConvConfig(
            window_strides=(1,) * self.rank,
            padding="VALID",
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=(1,) * self.rank,
            dimension_numbers=dimension_numbers,
        )
        out = conv_general_dilated(out, self.pointwise_kernel, config_obj2)

        if self.use_bias:
            bias_shape = (
                [1] * (self.rank + 1) + [self.filters]
                if self.data_format == "channels_last"
                else [1, self.filters] + [1] * self.rank
            )
            out = ops.add(out, ops.reshape(self.bias, bias_shape))

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class SeparableConv2D(Layer):
    """2D separable convolution layer.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels.
    If `use_bias` is True and a bias initializer is provided,
    it adds a bias vector to the output. It then optionally applies an
    activation function to produce the final output.

    Args:
        filters: int, the dimensionality of the output space (i.e. the number
            of filters in the pointwise convolution).
        kernel_size: int or tuple/list of 2 integers, specifying the size of the
            depthwise convolution window.
        strides: int or tuple/list of 2 integers, specifying the stride length
            of the depthwise convolution. If only one int is specified, the same
            stride size will be used for all dimensions. `strides > 1` is
            incompatible with `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file
            at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. If only one int is specified,
            the same dilation rate will be used for all dimensions.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        depthwise_initializer: An initializer for the depthwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        pointwise_initializer: An initializer for the pointwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: An initializer for the bias vector. If None, the
            default initializer ('"zeros"') will be used.
        depthwise_regularizer: Optional regularizer for the depthwise
            convolution kernel.
        pointwise_regularizer: Optional regularizer for the pointwise
            convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            depthwise kernel after being updated by an `Optimizer` (e.g. used
            for norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape).
        pointwise_constraint: Optional projection function to be applied to the
            pointwise kernel after being updated by an `Optimizer`.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`

    Returns:
        A 4D tensor representing
        `activation(separable_conv2d(inputs, kernel) + bias)`.

    Example:
    >>> x = np.random.rand(4, 10, 10, 12)
    >>> y = keras.layers.SeparableConv2D(3, 4, 3, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 4, 4, 4)

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = (1, 1),
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = (1, 1),
        depth_multiplier: TensorLike = 1,
        activation: TensorLike = None,
        use_bias: TensorLike = True,
        depthwise_initializer: TensorLike = "glorot_uniform",
        pointwise_initializer: TensorLike = "glorot_uniform",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            depth_multiplier: Description.
            activation: Description.
            use_bias: Description.
            depthwise_initializer: Description.
            pointwise_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.rank = 2
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.strides = (strides, strides) if isinstance(strides, int) else tuple(strides)
        self.dilation_rate = (
            (dilation_rate, dilation_rate)
            if isinstance(dilation_rate, int)
            else tuple(dilation_rate)
        )
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format or "channels_last"
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer
        self.pointwise_initializer = pointwise_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        channel_axis = -1 if self.data_format == "channels_last" else 1
        input_channel = input_shape[channel_axis]

        depthwise_shape = self.kernel_size + (input_channel, self.depth_multiplier)
        pointwise_shape = (1, 1) + (input_channel * self.depth_multiplier, self.filters)

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

        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    name="bias",
                )
        self.built = True

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
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

        conv_general_dilated = ops.conv_general_dilated

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding.upper(),
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=dimension_numbers,
            feature_group_count=input_channel,
        )
        out = conv_general_dilated(inputs, self.depthwise_kernel, config_obj)
        config_obj2 = ConvConfig(
            window_strides=(1,) * self.rank,
            padding="VALID",
            lhs_dilation=(1,) * self.rank,
            rhs_dilation=(1,) * self.rank,
            dimension_numbers=dimension_numbers,
        )
        out = conv_general_dilated(out, self.pointwise_kernel, config_obj2)

        if self.use_bias:
            bias_shape = (
                [1] * (self.rank + 1) + [self.filters]
                if self.data_format == "channels_last"
                else [1, self.filters] + [1] * self.rank
            )
            out = ops.add(out, ops.reshape(self.bias, bias_shape))

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class UpSampling1D(Layer):
    """Upsampling layer for 1D inputs.

    Repeats each temporal step `size` times along the time axis.

    Example:
    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = keras.layers.UpSampling1D(size=2)(x)
    >>> y
    [[[ 0.  1.  2.]
      [ 0.  1.  2.]
      [ 3.  4.  5.]
      [ 3.  4.  5.]]
     [[ 6.  7.  8.]
      [ 6.  7.  8.]
      [ 9. 10. 11.]
      [ 9. 10. 11.]]]

    Args:
        size: Integer. Upsampling factor.

    Input shape:
        3D tensor with shape: `(batch_size, steps, features)`.

    Output shape:
        3D tensor with shape: `(batch_size, upsampled_steps, features)`.

    """

    def __init__(
        self, size: TensorLike = 2, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            size: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.data_format = data_format or "channels_last"
        self.size = (size,) * 1 if isinstance(size, int) else tuple(size)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)
        out = inputs
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            out = ops.repeat(out, repeats=self.size[i], dim=dim_idx)

        return _wrap(out)


class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs.

    The implementation uses interpolative resizing, given the resize method
    (specified by the `interpolation` argument). Use `interpolation=nearest`
    to repeat the rows and columns of the data.

    Example:
    >>> input_shape = (2, 2, 1, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[ 0  1  2]]
      [[ 3  4  5]]]
     [[[ 6  7  8]]
      [[ 9 10 11]]]]
    >>> y = keras.layers.UpSampling2D(size=(1, 2))(x)
    >>> print(y)
    [[[[ 0  1  2]
       [ 0  1  2]]
      [[ 3  4  5]
       [ 3  4  5]]]
     [[[ 6  7  8]
       [ 6  7  8]]
      [[ 9 10 11]
       [ 9 10 11]]]]

    Args:
        size: Int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            When unspecified, uses
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json` (if exists) else `"channels_last"`.
            Defaults to `"channels_last"`.
        interpolation: A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`,
            `"lanczos5"`, `"nearest"`.

    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, rows, cols)`

    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, upsampled_rows, upsampled_cols)`

    """

    def __init__(
        self, size: TensorLike = 2, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            size: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.data_format = data_format or "channels_last"
        self.size = (size,) * 2 if isinstance(size, int) else tuple(size)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)
        out = inputs
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            out = ops.repeat(out, repeats=self.size[i], dim=dim_idx)

        return _wrap(out)


class UpSampling3D(Layer):
    """Upsampling layer for 3D inputs.

    Repeats the 1st, 2nd and 3rd dimensions
    of the data by `size[0]`, `size[1]` and `size[2]` respectively.

    Example:
    >>> input_shape = (2, 1, 2, 1, 3)
    >>> x = np.ones(input_shape)
    >>> y = keras.layers.UpSampling3D(size=(2, 2, 2))(x)
    >>> y.shape
    (2, 2, 4, 2, 3)

    Args:
        size: Int, or tuple of 3 integers.
            The upsampling factors for dim1, dim2 and dim3.
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            When unspecified, uses
            `image_data_format` value found in your Keras config file at
             `~/.keras/keras.json` (if exists) else `"channels_last"`.
            Defaults to `"channels_last"`.

    Input shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, dim1, dim2, dim3, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, dim1, dim2, dim3)`

    Output shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3,
            channels)`
        - If `data_format` is `"channels_first"`:
            `(batch_size, channels, upsampled_dim1, upsampled_dim2,
            upsampled_dim3)`

    """

    def __init__(
        self, size: TensorLike = 2, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            size: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 3
        self.data_format = data_format or "channels_last"
        self.size = (size,) * 3 if isinstance(size, int) else tuple(size)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)
        out = inputs
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            out = ops.repeat(out, repeats=self.size[i], dim=dim_idx)

        return _wrap(out)


class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

    Example:
    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = keras.layers.ZeroPadding1D(padding=2)(x)
    >>> y
    [[[ 0  0  0]
      [ 0  0  0]
      [ 0  1  2]
      [ 3  4  5]
      [ 0  0  0]
      [ 0  0  0]]
     [[ 0  0  0]
      [ 0  0  0]
      [ 6  7  8]
      [ 9 10 11]
      [ 0  0  0]
      [ 0  0  0]]]

    Args:
        padding: Int, or tuple of int (length 2), or dictionary.
            - If int: how many zeros to add at the beginning and end of
              the padding dimension (axis 1).
            - If tuple of 2 ints: how many zeros to add at the beginning and the
              end of the padding dimension (`(left_pad, right_pad)`).
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, axis_to_pad, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, axis_to_pad)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        3D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, axis_to_pad, features)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, features, axis_to_pad)`

    Output shape:
        3D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, padded_axis, features)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, features, padded_axis)`

    """

    def __init__(
        self, padding: TensorLike = 1, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            padding: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.data_format = data_format or "channels_last"

        if isinstance(padding, int):
            self.padding = ((padding, padding),) * 1
        elif isinstance(padding, tuple) and len(padding) == 1 and isinstance(padding[0], int):
            self.padding = tuple((p, p) for p in padding)
        else:
            self.padding = tuple(tuple(p) for p in padding)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            pad_width = ((0, 0), (0, 0)) + self.padding
        else:
            pad_width = ((0, 0),) + self.padding + ((0, 0),)

        out = ops.pad(inputs, pad_width=pad_width, mode="constant", constant_values=0.0)
        return _wrap(out)


class ZeroPadding2D(Layer):
    """Zero-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns of zeros at the top, bottom, left and
    right side of an image tensor.

    Example:
    >>> input_shape = (1, 1, 2, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[[0 1]
       [2 3]]]]
    >>> y = keras.layers.ZeroPadding2D(padding=1)(x)
    >>> y
    [[[[0 0]
       [0 0]
       [0 0]
       [0 0]]
      [[0 0]
       [0 1]
       [2 3]
       [0 0]]
      [[0 0]
       [0 0]
       [0 0]
       [0 0]]]]

    Args:
        padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding is applied to height and width.
            - If tuple of 2 ints: interpreted as two different symmetric padding
              values for height and width:
              `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints: interpreted as
             `((top_pad, bottom_pad), (left_pad, right_pad))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, height, width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, padded_height, padded_width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, padded_height, padded_width)`

    """

    def __init__(
        self, padding: TensorLike = 1, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            padding: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.data_format = data_format or "channels_last"

        if isinstance(padding, int):
            self.padding = ((padding, padding),) * 2
        elif isinstance(padding, tuple) and len(padding) == 2 and isinstance(padding[0], int):
            self.padding = tuple((p, p) for p in padding)
        else:
            self.padding = tuple(tuple(p) for p in padding)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            pad_width = ((0, 0), (0, 0)) + self.padding
        else:
            pad_width = ((0, 0),) + self.padding + ((0, 0),)

        out = ops.pad(inputs, pad_width=pad_width, mode="constant", constant_values=0.0)
        return _wrap(out)


class ZeroPadding3D(Layer):
    """Zero-padding layer for 3D data (spatial or spatio-temporal).

    Example:
    >>> input_shape = (1, 1, 2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = keras.layers.ZeroPadding3D(padding=2)(x)
    >>> y.shape
    (1, 5, 6, 6, 3)

    Args:
        padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
            - If int: the same symmetric padding is applied to depth, height,
              and width.
            - If tuple of 3 ints: interpreted as three different symmetric
              padding values for depth, height, and width:
              `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
            - If tuple of 3 tuples of 2 ints: interpreted as
              `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
              right_dim2_pad), (left_dim3_pad, right_dim3_pad))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad, depth)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad)`

    Output shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_padded_axis, second_padded_axis,
          third_axis_to_pad, depth)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, depth, first_padded_axis, second_padded_axis,
          third_axis_to_pad)`

    """

    def __init__(
        self, padding: TensorLike = 1, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            padding: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 3
        self.data_format = data_format or "channels_last"

        if isinstance(padding, int):
            self.padding = ((padding, padding),) * 3
        elif isinstance(padding, tuple) and len(padding) == 3 and isinstance(padding[0], int):
            self.padding = tuple((p, p) for p in padding)
        else:
            self.padding = tuple(tuple(p) for p in padding)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            pad_width = ((0, 0), (0, 0)) + self.padding
        else:
            pad_width = ((0, 0),) + self.padding + ((0, 0),)

        out = ops.pad(inputs, pad_width=pad_width, mode="constant", constant_values=0.0)
        return _wrap(out)


class Cropping1D(Layer):
    """Cropping layer for 1D input (e.g. temporal sequence).

    It crops along the time dimension (axis 1).

    Example:
    >>> input_shape = (2, 3, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1]
      [ 2  3]
      [ 4  5]]
     [[ 6  7]
      [ 8  9]
      [10 11]]]
    >>> y = keras.layers.Cropping1D(cropping=1)(x)
    >>> y
    [[[2 3]]
     [[8 9]]]

    Args:
        cropping: Int, or tuple of int (length 2), or dictionary.
            - If int: how many units should be trimmed off at the beginning and
              end of the cropping dimension (axis 1).
            - If tuple of 2 ints: how many units should be trimmed off at the
              beginning and end of the cropping dimension
              (`(left_crop, right_crop)`).

    Input shape:
        3D tensor with shape `(batch_size, axis_to_crop, features)`

    Output shape:
        3D tensor with shape `(batch_size, cropped_axis, features)`

    """

    def __init__(
        self, cropping: TensorLike = 1, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            cropping: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.data_format = data_format or "channels_last"

        if isinstance(cropping, int):
            self.cropping = ((cropping, cropping),) * 1
        elif isinstance(cropping, tuple) and len(cropping) == 1 and isinstance(cropping[0], int):
            self.cropping = tuple((p, p) for p in cropping)
        else:
            self.cropping = tuple(tuple(p) for p in cropping)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)

        # Build slicing bounds
        slices = [slice(None)] * len(inputs.shape)
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            crop_start = self.cropping[i][0]
            crop_end = -self.cropping[i][1] if self.cropping[i][1] > 0 else None
            slices[dim_idx] = slice(crop_start, crop_end)

        out = inputs[tuple(slices)]
        return _wrap(out)


class Cropping2D(Layer):
    """Cropping layer for 2D input (e.g. picture).

    It crops along spatial dimensions, i.e. height and width.

    Example:
    >>> input_shape = (2, 28, 28, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
    >>> y.shape
    (2, 24, 20, 3)

    Args:
        cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric cropping is applied to height and
              width.
            - If tuple of 2 ints: interpreted as two different symmetric
              cropping values for height and width:
              `(symmetric_height_crop, symmetric_width_crop)`.
            - If tuple of 2 tuples of 2 ints: interpreted as
              `((top_crop, bottom_crop), (left_crop, right_crop))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, height, width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, cropped_height, cropped_width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, cropped_height, cropped_width)`

    """

    def __init__(
        self, cropping: TensorLike = 1, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            cropping: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.data_format = data_format or "channels_last"

        if isinstance(cropping, int):
            self.cropping = ((cropping, cropping),) * 2
        elif isinstance(cropping, tuple) and len(cropping) == 2 and isinstance(cropping[0], int):
            self.cropping = tuple((p, p) for p in cropping)
        else:
            self.cropping = tuple(tuple(p) for p in cropping)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)

        # Build slicing bounds
        slices = [slice(None)] * len(inputs.shape)
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            crop_start = self.cropping[i][0]
            crop_end = -self.cropping[i][1] if self.cropping[i][1] > 0 else None
            slices[dim_idx] = slice(crop_start, crop_end)

        out = inputs[tuple(slices)]
        return _wrap(out)


class Cropping3D(Layer):
    """Cropping layer for 3D data (e.g. spatial or spatio-temporal).

    Example:
    >>> input_shape = (2, 28, 28, 10, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = keras.layers.Cropping3D(cropping=(2, 4, 2))(x)
    >>> y.shape
    (2, 24, 20, 6, 3)

    Args:
        cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
            - If int: the same symmetric cropping is applied to depth, height,
              and width.
            - If tuple of 3 ints: interpreted as three different symmetric
              cropping values for depth, height, and width:
              `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
            - If tuple of 3 tuples of 2 ints: interpreted as
              `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
              right_dim2_crop), (left_dim3_crop, right_dim3_crop))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_axis_to_crop, second_axis_to_crop,
          third_axis_to_crop, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, first_axis_to_crop, second_axis_to_crop,
          third_axis_to_crop)`

    Output shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_cropped_axis, second_cropped_axis,
          third_cropped_axis, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, first_cropped_axis, second_cropped_axis,
          third_cropped_axis)`

    """

    def __init__(
        self, cropping: TensorLike = 1, data_format: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            cropping: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rank = 3
        self.data_format = data_format or "channels_last"

        if isinstance(cropping, int):
            self.cropping = ((cropping, cropping),) * 3
        elif isinstance(cropping, tuple) and len(cropping) == 3 and isinstance(cropping[0], int):
            self.cropping = tuple((p, p) for p in cropping)
        else:
            self.cropping = tuple(tuple(p) for p in cropping)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)

        # Build slicing bounds
        slices = [slice(None)] * len(inputs.shape)
        spatial_start = 2 if self.data_format == "channels_first" else 1

        for i in range(self.rank):
            dim_idx = spatial_start + i
            crop_start = self.cropping[i][0]
            crop_end = -self.cropping[i][1] if self.cropping[i][1] > 0 else None
            slices[dim_idx] = slice(crop_start, crop_end)

        out = inputs[tuple(slices)]
        return _wrap(out)

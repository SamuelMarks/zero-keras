from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.activations import get as get_activation
from zero_keras.core_layers import Layer as BaseLayer


class Layer(BaseLayer):
    """This is the class from which all layers inherit.

    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables). State can be
    created:

    * in `__init__()`, for instance via `self.add_weight()`;
    * in the optional `build()` method, which is invoked by the first
      `__call__()` to the layer, and supplies the shape(s) of the input(s),
      which may not have been known at initialization time.

    Layers are recursively composable: If you assign a Layer instance as an
    attribute of another Layer, the outer layer will start tracking the weights
    created by the inner layer. Nested layers should be instantiated in the
    `__init__()` method or `build()` method.

    Users will just instantiate a layer and then treat it as a callable.

    Args:
        trainable: Boolean, whether the layer's variables should be trainable.
        name: String name of the layer.
        dtype: The dtype of the layer's computations and weights. Can also be a
            `keras.DTypePolicy`, which allows the computation and weight dtype
            to differ. Defaults to `None`. `None` means to use
            `keras.config.dtype_policy()`, which is a `float32` policy unless
            set to different value (via `keras.config.set_dtype_policy()`).

    Attributes:
        name: The name of the layer (string).
        dtype: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
        variable_dtype: Dtype of the layer's weights.
        compute_dtype: The dtype of the layer's computations.
            Layers automatically cast inputs to this dtype, which causes
            the computations and output to also be in this dtype.
            When mixed precision is used with a
            `keras.DTypePolicy`, this will be different
            than `variable_dtype`.
        trainable_weights: List of variables to be included in backprop.
        non_trainable_weights: List of variables that should not be
            included in backprop.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        trainable: Whether the layer should be trained (boolean), i.e.
            whether its potentially-trainable weights should be returned
            as part of `layer.trainable_weights`.
        input_spec: Optional (list of) `InputSpec` object(s) specifying the
            constraints on inputs that can be accepted by the layer.

    We recommend that descendants of `Layer` implement the following methods:

    * `__init__()`: Defines custom layer attributes, and creates layer weights
        that do not depend on input shapes, using `add_weight()`,
        or other state.
    * `build(self, input_shape)`: This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`, or other
        state. `__call__()` will automatically build the layer
        (if it has not been built yet) by calling `build()`.
    * `call(self, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying
        the layer to the input arguments.
        Two reserved keyword arguments you can optionally use in `call()` are:
            1. `training` (boolean, whether the call is in inference mode or
                training mode).
            2. `mask` (boolean tensor encoding masked timesteps in the input,
                used e.g. in RNN layers).
        A typical signature for this method is `call(self, inputs)`, and user
        could optionally add `training` and `mask` if the layer need them.
    * `get_config(self)`: Returns a dictionary containing the configuration
        used to initialize this layer. If the keys differ from the arguments
        in `__init__()`, then override `from_config(self)` as well.
        This method is used when saving
        the layer or a model that contains this layer.

    Examples:
    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).

    ```python
    class SimpleDense(Layer):
        def __init__(self, units=32):
            super().__init__()
            self.units = units

        # Create the state of the layer (weights)
            # pragma: no cover
    def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
                name="kernel",
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

        # Defines the computation
            # pragma: no cover
    def call(self, inputs):
            return ops.matmul(inputs, self.kernel) + self.bias

    # Instantiates the layer.
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(ops.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```

    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:

    ```python
    class ComputeSum(Layer):

      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            name="total",
          )

          # pragma: no cover
    def call(self, inputs):
          self.total.assign(self.total + ops.sum(inputs))
          return self.total

    my_sum = ComputeSum(2)
    x = ops.ones((2, 2))
    y = my_sum(x)

    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```

    """

    def __init__(self, **kwargs):
        """Function docstring.

        Args:
            kwargs: Description.
        """
        super().__init__(**kwargs)

        # pragma: no cover

    def call(self, *args, **kwargs):
        """Call function.

        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        return args[0] if args else None


class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors). The output in this case will have
    shape `(batch_size, d0, units)`.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `Dense` layer by calling `layer.enable_lora(rank)`.
        lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.

    """

    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        """Function docstring.

        Args:
            units: Description.
            activation: Description.
            use_bias: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None

    def compute_output_shape(self, input_shape):
        """compute_output_shape docstring.

        Args:
            input_shape: Input shape.
        """

        input_shape = tuple(input_shape)

        return input_shape[:-1] + (self.units,)

        # pragma: no cover

    def get_config(self):
        """get_config docstring."""
        from zero_keras.activations import serialize as serialize_activation

        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": serialize_activation(self.activation),
                "use_bias": self.use_bias,
            }
        )
        return config

        # pragma: no cover

    def build(self, input_shape):
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        Any: Return value.

        """
        if self.built:
            return
        in_dim = input_shape[-1]
        if getattr(self, "kernel", None) is None:  # pragma: no branch
            self.kernel = self.add_weight(
                name="kernel",
                shape=(in_dim, self.units),
                initializer=self._kwargs.get("kernel_initializer", "glorot_uniform"),
                trainable=True,
            )
        if self.use_bias:  # pragma: no branch  # pragma: no branch
            if getattr(self, "bias", None) is None:  # pragma: no branch
                self.bias = self.add_weight(
                    name="bias",
                    shape=(self.units,),
                    initializer=self._kwargs.get("bias_initializer", "zeros"),
                    trainable=True,
                )
        self.built = True

        # pragma: no cover

    def call(self, inputs, *args, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)  # pragma: no cover

        out = ops.matmul(inputs, _to_tensor(self.kernel))
        if self.use_bias:  # pragma: no branch  # pragma: no branch
            out = ops.add(out, self.bias)

        if self.activation is not None:  # pragma: no branch
            out = self.activation(out)
        return _wrap(out)


class Dropout(Layer):
    """Applies dropout to the input.

    The `Dropout` layer randomly sets input units to 0 with a frequency of
    `rate` at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by `1 / (1 - rate)` such that the sum over
    all inputs is unchanged.

    Note that the `Dropout` layer only applies when `training` is set to `True`
    in `call()`, such that no values are dropped during inference.
    When using `model.fit`, `training` will be appropriately set to `True`
    automatically. In other contexts, you can set the argument explicitly
    to `True` when calling the layer.

    (This is in contrast to setting `trainable=False` for a `Dropout` layer.
    `trainable` does not affect the layer's behavior, as `Dropout` does
    not have any variables/weights that can be frozen during training.)

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).

    """

    def __init__(self, rate, **kwargs):
        """Function docstring.

        Args:
            rate: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rate = rate

        # pragma: no cover

    def call(self, inputs, training=None, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        training: Parameter training.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        if training or training is None:
            from ml_switcheroo_compiler import random

            key = random.PRNGKey(42)
            from zero_keras.core_layers import KerasTensor

            if isinstance(inputs, KerasTensor):  # pragma: no branch
                return _wrap(inputs)
            mask_t = ops.cast(
                random.bernoulli(key, 1.0 - self.rate, inputs.shape), dtype=inputs.dtype
            ) / (1.0 - self.rate)
            return _wrap(ops.multiply(inputs, mask_t))
        return _wrap(inputs)  # pragma: no cover


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    Note: If inputs are shaped `(batch,)` without a feature axis, then
    flattening adds an extra channel dimension and output shape is `(batch, 1)`.

    Args:
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Example:
    >>> x = keras.Input(shape=(10, 64))
    >>> y = keras.layers.Flatten()(x)
    >>> y.shape
    (None, 640)

    """

    def __init__(self, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format

        # pragma: no cover

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        shape = inputs.shape
        if len(shape) <= 1:
            return _wrap(inputs)  # pragma: no cover

        new_dim = 1
        for d in shape[1:]:
            new_dim *= d
        return _wrap(ops.reshape(inputs, (shape[0], int(new_dim))))


class Reshape(Layer):
    """Layer that reshapes inputs into the given shape.

    Args:
        target_shape: Target shape. Tuple of integers, does not include the
            samples dimension (batch size).

    Input shape:
        Arbitrary, although all dimensions in the input shape must be
        known/fixed. Use the keyword argument `input_shape` (tuple of integers,
        does not include the samples/batch size axis) when using this layer as
        the first layer in a model.

    Output shape:
        `(batch_size, *target_shape)`

    Example:
    >>> x = keras.Input(shape=(12,))
    >>> y = keras.layers.Reshape((3, 4))(x)
    >>> y.shape
    (None, 3, 4)

    >>> # also supports shape inference using `-1` as dimension
    >>> y = keras.layers.Reshape((-1, 2, 2))(x)
    >>> y.shape
    (None, 3, 2, 2)

    """

    def __init__(self, target_shape, **kwargs):
        """Function docstring.

        Args:
            target_shape: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

        # pragma: no cover

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        shape = (inputs.shape[0],) + self.target_shape
        return _wrap(ops.reshape(inputs, shape))


class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern.

    Useful e.g. connecting RNNs and convnets.

    Args:
        dims: Tuple of integers. Permutation pattern does not include the
            batch dimension. Indexing starts at 1.
            For instance, `(1, 3, 2)` permutes the second and third dimensions
            of the input.

    Input shape:
        Arbitrary.

    Output shape:
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.

    Example:
    >>> x = keras.Input(shape=(10, 64))
    >>> y = keras.layers.Permute((2, 1))(x)
    >>> y.shape
    (None, 64, 10)

    """

    def __init__(self, dims, **kwargs):
        """Function docstring.

        Args:
            dims: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.dims = tuple(dims)

        # pragma: no cover

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        perm = [0] + [d for d in self.dims]
        return _wrap(ops.permute(inputs, perm))


class RepeatVector(Layer):
    """Repeats the input n times.

    Example:
    >>> x = keras.Input(shape=(32,))
    >>> y = keras.layers.RepeatVector(3)(x)
    >>> y.shape
    (None, 3, 32)

    Args:
        n: Integer, repetition factor.

    Input shape:
        2D tensor with shape `(batch_size, features)`.

    Output shape:
        3D tensor with shape `(batch_size, n, features)`.

    """

    def __init__(self, n, **kwargs):
        """Function docstring.

        Args:
            n: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.n = n

        # pragma: no cover

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        expanded = ops.expand_dims(inputs, 1)
        repeats = [1, self.n, 1]
        return _wrap(ops.tile(expanded, repeats))


class Masking(Layer):
    """Masks a sequence by using a mask value to skip timesteps.

    For each timestep in the input tensor (dimension #1 in the tensor),
    if all values in the input tensor at that timestep
    are equal to `mask_value`, then the timestep will be masked (skipped)
    in all downstream layers (as long as they support masking).

    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.

    Example:
    Consider a NumPy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
    lack data for these timesteps. You can:

    - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
    - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
    samples, timesteps, features = 32, 10, 8
    inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
    inputs[:, 3, :] = 0.
    inputs[:, 5, :] = 0.

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0.0))
    model.add(keras.layers.LSTM(32))
    output = model(inputs)
    # The time step 3 and 5 will be skipped from LSTM calculation.
    ```

    Note: in the Keras masking convention, a masked timestep is denoted by
    a mask value of `False`, while a non-masked (i.e. usable) timestep
    is denoted by a mask value of `True`.

    """

    def __init__(self, mask_value=0.0, **kwargs):
        """Function docstring.

        Args:
            mask_value: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.mask_value = mask_value

        # pragma: no cover

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        return _wrap(_to_tensor(inputs))


class Lambda(Layer):
    """Wraps arbitrary expressions as a `Layer` object.

    The `Lambda` layer exists so that arbitrary expressions can be used
    as a `Layer` when constructing Sequential
    and Functional API models. `Lambda` layers are best suited for simple
    operations or quick experimentation. For more advanced use cases,
    prefer writing new subclasses of `Layer`.

    WARNING: `Lambda` layers have (de)serialization limitations!

    The main reason to subclass `Layer` instead of using a
    `Lambda` layer is saving and inspecting a model. `Lambda` layers
    are saved by serializing the Python bytecode, which is fundamentally
    non-portable and potentially unsafe.
    They should only be loaded in the same environment where
    they were saved. Subclassed layers can be saved in a more portable way
    by overriding their `get_config()` method. Models that rely on
    subclassed Layers are also often easier to visualize and reason about.

    Example:
    ```python
    # add a x -> x^2 layer
    model.add(Lambda(lambda x: x ** 2))
    ```

    Args:
        function: The function to be evaluated. Takes input tensor as first
            argument.
        output_shape: Expected output shape from function. This argument
            can usually be inferred if not explicitly provided.
            Can be a tuple or function. If a tuple, it only specifies
            the first dimension onward; sample dimension is assumed
            either the same as the input:
            `output_shape = (input_shape[0], ) + output_shape` or,
            the input is `None` and the sample dimension is also `None`:
            `output_shape = (None, ) + output_shape`.
            If a function, it specifies the
            entire shape as a function of the input shape:
            `output_shape = f(input_shape)`.
        mask: Either None (indicating no masking) or a callable with the same
            signature as the `compute_mask` layer method, or a tensor
            that will be returned as output mask regardless
            of what the input is.
        arguments: Optional dictionary of keyword arguments to be passed to the
            function.

    """

    def __init__(self, function, **kwargs):
        """Function docstring.

        Args:
            function: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.function = function

        # pragma: no cover

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        return self.function(inputs, **kwargs)


class Identity(Layer):
    """Identity layer.

    This layer should be used as a placeholder when no operation is to be
    performed. The layer just returns its `inputs` argument as output.
    """

    # pragma: no cover
    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        return _wrap(inputs)


class InputLayer(Layer):
    """This is the class from which all layers inherit.

    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables). State can be
    created:

    * in `__init__()`, for instance via `self.add_weight()`;
    * in the optional `build()` method, which is invoked by the first
      `__call__()` to the layer, and supplies the shape(s) of the input(s),
      which may not have been known at initialization time.

    Layers are recursively composable: If you assign a Layer instance as an
    attribute of another Layer, the outer layer will start tracking the weights
    created by the inner layer. Nested layers should be instantiated in the
    `__init__()` method or `build()` method.

    Users will just instantiate a layer and then treat it as a callable.

    Args:
        trainable: Boolean, whether the layer's variables should be trainable.
        name: String name of the layer.
        dtype: The dtype of the layer's computations and weights. Can also be a
            `keras.DTypePolicy`, which allows the computation and weight dtype
            to differ. Defaults to `None`. `None` means to use
            `keras.config.dtype_policy()`, which is a `float32` policy unless
            set to different value (via `keras.config.set_dtype_policy()`).

    Attributes:
        name: The name of the layer (string).
        dtype: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
        variable_dtype: Dtype of the layer's weights.
        compute_dtype: The dtype of the layer's computations.
            Layers automatically cast inputs to this dtype, which causes
            the computations and output to also be in this dtype.
            When mixed precision is used with a
            `keras.DTypePolicy`, this will be different
            than `variable_dtype`.
        trainable_weights: List of variables to be included in backprop.
        non_trainable_weights: List of variables that should not be
            included in backprop.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        trainable: Whether the layer should be trained (boolean), i.e.
            whether its potentially-trainable weights should be returned
            as part of `layer.trainable_weights`.
        input_spec: Optional (list of) `InputSpec` object(s) specifying the
            constraints on inputs that can be accepted by the layer.

    We recommend that descendants of `Layer` implement the following methods:

    * `__init__()`: Defines custom layer attributes, and creates layer weights
        that do not depend on input shapes, using `add_weight()`,
        or other state.
    * `build(self, input_shape)`: This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`, or other
        state. `__call__()` will automatically build the layer
        (if it has not been built yet) by calling `build()`.
    * `call(self, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying
        the layer to the input arguments.
        Two reserved keyword arguments you can optionally use in `call()` are:
            1. `training` (boolean, whether the call is in inference mode or
                training mode).
            2. `mask` (boolean tensor encoding masked timesteps in the input,
                used e.g. in RNN layers).
        A typical signature for this method is `call(self, inputs)`, and user
        could optionally add `training` and `mask` if the layer need them.
    * `get_config(self)`: Returns a dictionary containing the configuration
        used to initialize this layer. If the keys differ from the arguments
        in `__init__()`, then override `from_config(self)` as well.
        This method is used when saving
        the layer or a model that contains this layer.

    Examples:
    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).

    ```python
    class SimpleDense(Layer):
        def __init__(self, units=32):
            super().__init__()
            self.units = units

        # Create the state of the layer (weights)
            # pragma: no cover
    def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
                name="kernel",
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

        # Defines the computation
            # pragma: no cover
    def call(self, inputs):
            return ops.matmul(inputs, self.kernel) + self.bias

    # Instantiates the layer.
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(ops.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```

    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:

    ```python
    class ComputeSum(Layer):

      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            name="total",
          )

          # pragma: no cover
    def call(self, inputs):
          self.total.assign(self.total + ops.sum(inputs))
          return self.total

    my_sum = ComputeSum(2)
    x = ops.ones((2, 2))
    y = my_sum(x)

    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```

    """

    def __init__(
        self,
        input_shape=None,
        batch_size=None,
        dtype=None,
        input_tensor=None,
        sparse=None,
        name=None,
        batch_input_shape=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            input_shape: Description.
            batch_size: Description.
            dtype: Description.
            input_tensor: Description.
            sparse: Description.
            name: Description.
            batch_input_shape: Description.
            kwargs: Description.
        """

        # pragma: no cover
        super().__init__(name=name, **kwargs)
        # pragma: no cover

        # pragma: no cover
        if input_shape is not None and batch_size is not None:  # pragma: no branch
            # pragma: no cover

            # pragma: no cover
            self.batch_input_shape = (batch_size,) + tuple(input_shape)
        # pragma: no cover

        # pragma: no cover
        elif input_shape is not None:  # pragma: no branch
            # pragma: no cover

            # pragma: no cover
            self.batch_input_shape = (None,) + tuple(input_shape)
        # pragma: no cover

        # pragma: no cover
        else:
            # pragma: no cover

            # pragma: no cover
            self.batch_input_shape = batch_input_shape
        # pragma: no cover

        # pragma: no cover

        # pragma: no cover

        # pragma: no cover
        self._dtype = dtype or "float32"
        # pragma: no cover

        # pragma: no cover
        self.sparse = sparse or False
        # pragma: no cover

        # pragma: no cover
        if input_tensor is None and self.batch_input_shape is not None:
            # pragma: no cover

            # pragma: no cover
            from zero_keras.core_layers import KerasTensor
            # pragma: no cover

            # pragma: no cover

            # pragma: no cover

            # pragma: no cover
            self.input_tensor = KerasTensor(
                # pragma: no cover
                # pragma: no cover
                self.batch_input_shape,
                self.dtype,
                name=name,
                # pragma: no cover
            )
        # pragma: no cover
        else:
            # pragma: no cover
            self.input_tensor = input_tensor

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        return inputs


class InputSpec:
    """Specifies the rank, dtype and shape of every input to a layer.

    Layers can expose (if appropriate) an `input_spec` attribute:
    an instance of `InputSpec`, or a nested structure of `InputSpec` instances
    (one per input tensor). These objects enable the layer to run input
    compatibility checks for input structure, input rank, input shape, and
    input dtype for the first argument of `Layer.__call__`.

    A `None` entry in a shape is compatible with any dimension.

    Args:
        dtype: Expected dtype of the input.
        shape: Shape tuple, expected shape of the input
            (may include `None` for dynamic axes).
            Includes the batch size.
        ndim: Integer, expected rank of the input.
        max_ndim: Integer, maximum rank of the input.
        min_ndim: Integer, minimum rank of the input.
        axes: Dictionary mapping integer axes to
            a specific dimension value.
        allow_last_axis_squeeze: If `True`, allow inputs of rank N+1 as long
            as the last axis of the input is 1, as well as inputs of rank N-1
            as long as the last axis of the spec is 1.
        name: Expected key corresponding to this input when passing data as
            a dictionary.
        optional: Boolean, whether the input is optional or not.
            An optional input can accept `None` values.

    Example:
    ```python
    class MyLayer(Layer):
        def __init__(self):
            super().__init__()
            # The layer will accept inputs with
            # shape (*, 28, 28) & (*, 28, 28, 1)
            # and raise an appropriate error message otherwise.
            self.input_spec = InputSpec(
                shape=(None, 28, 28, 1),
                allow_last_axis_squeeze=True)
    ```

    """

    def __init__(
        self,
        dtype=None,
        shape=None,
        ndim=None,
        max_ndim=None,
        min_ndim=None,
        axes=None,
        allow_last_axis_squeeze=False,
        name=None,
    ):
        """Function docstring.

        Args:
            dtype: Description.
            shape: Description.
            ndim: Description.
            max_ndim: Description.
            min_ndim: Description.
            axes: Description.
            allow_last_axis_squeeze: Description.
            name: Description.
        """
        # pragma: no cover
        self.dtype = dtype
        # pragma: no cover
        self.shape = shape
        # pragma: no cover

        # pragma: no cover
        self.ndim = ndim
        # pragma: no cover

        # pragma: no cover
        self.max_ndim = max_ndim
        # pragma: no cover

        # pragma: no cover
        self.min_ndim = min_ndim
        # pragma: no cover

        # pragma: no cover
        self.axes = axes or {}
        # pragma: no cover

        # pragma: no cover
        self.allow_last_axis_squeeze = allow_last_axis_squeeze
        # pragma: no cover

        # pragma: no cover
        self.name = name


class Wrapper(Layer):
    """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` layers.

    Args:
        layer: The layer to be wrapped.

    """

    def __init__(self, layer, **kwargs):
        """Function docstring.



        Args:

            layer: Description.

            kwargs: Description.

        """

        super().__init__(layer=layer, **kwargs)

        self.layer = layer

        # pragma: no cover

    def build(self, input_shape=None):
        """Build function.



            Args:

            input_shape: Parameter input_shape.



            Returns:

            Any: Return value.



        # pragma: no cover
        """
        # pragma: no cover

        # pragma: no cover
        if self.built:
            # pragma: no cover

            # pragma: no cover
            return
        # pragma: no cover

        # pragma: no cover
        if not self.layer.built:
            self.layer.build(input_shape)

        self.built = True

        # pragma: no cover

    def call(self, inputs, *args, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        return self.layer(inputs, *args, **kwargs)


class _LegacyInput(Layer):
    """Input layer is not strictly needed in eager API compatible mode."""

    def __init__(
        self,  # pragma: no cover
        shape=None,  # pragma: no cover
        batch_size=None,  # pragma: no cover
        dtype=None,  # pragma: no cover
        input_tensor=None,  # pragma: no cover
        sparse=None,  # pragma: no cover
        name=None,  # pragma: no cover
        batch_shape=None,  # pragma: no cover
        **kwargs,  # pragma: no cover
    ):  # pragma: no cover
        """Function docstring.  # pragma: no cover
        # pragma: no cover
              Args:  # pragma: no cover
                  shape: Description.  # pragma: no cover
                  batch_size: Description.  # pragma: no cover
                  dtype: Description.  # pragma: no cover
                  input_tensor: Description.  # pragma: no cover
                  sparse: Description.  # pragma: no cover
                  name: Description.  # pragma: no cover
                  batch_shape: Description.  # pragma: no cover
                  kwargs: Description.  # pragma: no cover
        """  # pragma: no cover
        # pragma: no cover  # pragma: no cover
        super().__init__(name=name, dtype=dtype, **kwargs)  # pragma: no cover
        # pragma: no cover  # pragma: no cover
        if shape is not None and batch_shape is None:  # pragma: no cover
            # pragma: no cover  # pragma: no cover  # pragma: no cover
            batch_shape = (batch_size,) + tuple(
                shape
            )  # pragma: no cover  # pragma: no cover
        # pragma: no cover  # pragma: no cover  # pragma: no cover
        self.batch_shape = batch_shape  # pragma: no cover  # pragma: no cover

    # pragma: no cover
    # pragma: no cover  # pragma: no cover
    # pragma: no cover
    def call(self, inputs, *args, **kwargs):  # pragma: no cover
        """Function docstring.  # pragma: no cover
        # pragma: no cover
              Args:  # pragma: no cover
                  inputs: Description.  # pragma: no cover
                  args: Description.  # pragma: no cover
                  kwargs: Description.  # pragma: no cover
        """  # pragma: no cover
        return inputs  # pragma: no cover

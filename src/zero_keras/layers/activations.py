from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer


class Activation(Layer):
    """Applies an activation function to an output.

    Args:
        activation: Activation function. It could be a callable, or the name of
            an activation from the `keras.activations` namespace.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:
    >>> layer = keras.layers.Activation('relu')
    >>> layer(np.array([-3.0, -1.0, 0.0, 2.0]))
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = keras.layers.Activation(keras.activations.relu)
    >>> layer(np.array([-3.0, -1.0, 0.0, 2.0]))
    [0.0, 0.0, 0.0, 2.0]

    """

    def __init__(self, activation, **kwargs):
        """Function docstring.

        Args:
            activation: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        from zero_keras.activations import get

        self.activation = get(activation)

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        return self.activation(inputs)


class ELU(Layer):
    """Applies an Exponential Linear Unit function to an output.

    Formula:

    ```
    f(x) = alpha * (exp(x) - 1.) for x < 0
    f(x) = x for x >= 0
    ```

    Args:
        alpha: float, slope of negative section. Defaults to `1.0`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    """

    def __init__(self, alpha=1.0, **kwargs):
        """Function docstring.

        Args:
            alpha: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        from zero_keras import activations

        return activations.elu(inputs, alpha=self.alpha)


class PReLU(Layer):
    """Parametric Rectified Linear Unit activation layer.

    Formula:
    ``` python
    f(x) = alpha * x for x < 0
    f(x) = x for x >= 0
    ```
    where `alpha` is a learned array with the same shape as x.

    Args:
        alpha_initializer: Initializer function for the weights.
        alpha_regularizer: Regularizer for the weights.
        alpha_constraint: Constraint for the weights.
        shared_axes: The axes along which to share learnable parameters for the
            activation function. For example, if the incoming feature maps are
            from a 2D convolution with output shape
            `(batch, height, width, channels)`, and you wish to share parameters
            across space so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    """

    def __init__(
        self,
        alpha_initializer="zeros",
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            alpha_initializer: Description.
            alpha_regularizer: Description.
            alpha_constraint: Description.
            shared_axes: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.alpha_initializer = alpha_initializer
        self.shared_axes = shared_axes
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        from zero_keras.initializers import get

        self._alpha_init_fn = get(alpha_initializer)

    def build(self, input_shape):
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        Any: Return value.

        """
        if input_shape is not None:
            shape = input_shape[1:]
        else:  # pragma: no cover
            shape = ()  # pragma: no cover
        # Actually in zero_keras we might not have variables, just return a tensor
        self.alpha = self._alpha_init_fn(shape=shape)
        self.shared_axes = getattr(self, "shared_axes", None)
        super().build(input_shape)
        self.shared_axes = self.shared_axes if hasattr(self, "shared_axes") else None

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        from zero_keras import ops as backend_ops
        from zero_keras.activations import _to_tensor, _wrap

        inputs_t = _to_tensor(inputs)
        alpha_t = getattr(self.alpha, "data", None) if hasattr(self, "alpha") else None
        if alpha_t is None:
            alpha_t = 0.0  # pragma: no cover
        alpha_t = _to_tensor(alpha_t)
        zero_t = backend_ops.asarray(0.0, dtype=getattr(inputs_t, "dtype", "float32"))
        pos = backend_ops.maximum(inputs_t, zero_t)
        min_t = backend_ops.minimum(inputs_t, zero_t)
        neg = backend_ops.multiply(alpha_t, min_t)
        return _wrap(backend_ops.add(pos, neg))


class ReLU(Layer):
    """Rectified Linear Unit activation function layer.

    Formula:
    ``` python
    f(x) = max(x,0)
    f(x) = max_value if x >= max_value
    f(x) = x if threshold <= x < max_value
    f(x) = negative_slope * (x - threshold) otherwise
    ```

    Example:
    ``` python
    relu_layer = keras.layers.ReLU(
        max_value=10,
        negative_slope=0.5,
        threshold=0,
    )
    input = np.array([-10, -5, 0.0, 5, 10])
    result = relu_layer(input)
    # result = [-5. , -2.5,  0. ,  5. , 10.]
    ```

    Args:
        max_value: Float >= 0. Maximum activation value. None means unlimited.
            Defaults to `None`.
        negative_slope: Float >= 0. Negative slope coefficient.
            Defaults to `0.0`.
        threshold: Float >= 0. Threshold value for thresholded activation.
            Defaults to `0.0`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    """

    def __init__(self, max_value=None, negative_slope=0.0, threshold=0.0, **kwargs):
        """Function docstring.

        Args:
            max_value: Description.
            negative_slope: Description.
            threshold: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold

    def call(self, inputs, *args, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        from zero_keras import activations

        return activations.relu(
            inputs,
            negative_slope=self.negative_slope,
            max_value=self.max_value,
            threshold=self.threshold,
        )


class Softmax(Layer):
    """Softmax activation layer.

    Formula:
    ``` python
    exp_x = exp(x - max(x))
    f(x) = exp_x / sum(exp_x)
    ```

    Example:
    >>> softmax_layer = keras.layers.Softmax()
    >>> input = np.array([1.0, 2.0, 1.0])
    >>> result = softmax_layer(input)
    >>> result
    [0.21194157, 0.5761169, 0.21194157]


    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Call arguments:
        inputs: The inputs (logits) to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.

    Returns:
        Softmaxed output with the same shape as `inputs`.

    """

    def __init__(self, axis=-1, **kwargs):
        """Function docstring.

        Args:
            axis: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, *args, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        from zero_keras import activations

        return activations.softmax(inputs, axis=self.axis)


class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit activation layer.

    This layer allows a small gradient when the unit is not active.

    Formula:

    ``` python
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
    ```

    Example:
    ``` python
    leaky_relu_layer = LeakyReLU(negative_slope=0.5)
    input = np.array([-10, -5, 0.0, 5, 10])
    result = leaky_relu_layer(input)
    # result = [-5. , -2.5,  0. ,  5. , 10.]
    ```

    Args:
        negative_slope: Float >= 0.0. Negative slope coefficient.
          Defaults to `0.3`.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.

    """

    def __init__(self, negative_slope=0.3, **kwargs):
        """Function docstring.

        Args:
            negative_slope: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.negative_slope = negative_slope

    def call(self, inputs, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        inputs = _to_tensor(inputs)
        return _wrap(
            ops.maximum(inputs, ops.multiply(_to_tensor(self.negative_slope), inputs))
        )

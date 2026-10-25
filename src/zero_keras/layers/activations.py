"""Activation layers for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer
from zero_keras.types import (
    ActivationSpec,
    ConstraintSpec,
    InitializerSpec,
    RegularizerSpec,
    ShapeTuple,
    TensorLike,
)


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

    def __init__(self, activation: ActivationSpec, **kwargs: TensorLike) -> None:
        """Initialize the Activation layer.

        Args:
            activation: Activation function or identifier string.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        from zero_keras.activations import get

        self.activation = get(activation)

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply ``self.activation`` to ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Activated output tensor.
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

    def __init__(self, alpha: float = 1.0, **kwargs: TensorLike) -> None:
        """Initialize the ELU activation layer.

        Args:
            alpha: Scale factor for negative inputs. Defaults to ``1.0``.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply the Exponential Linear Unit activation to ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: ELU-activated tensor.
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
        alpha_initializer: InitializerSpec = "Zeros",
        alpha_regularizer: RegularizerSpec = None,
        alpha_constraint: ConstraintSpec = None,
        shared_axes: int | Sequence[int] | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the PReLU activation layer.

        Args:
            alpha_initializer: Initializer for the learnable ``alpha`` slope weight.
            alpha_regularizer: Optional regularizer for ``alpha``.
            alpha_constraint: Optional constraint for ``alpha``.
            shared_axes: Axis or axes along which ``alpha`` parameters are shared.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.alpha_initializer = alpha_initializer
        self.shared_axes = shared_axes
        self.alpha_regularizer = alpha_regularizer
        self.alpha_constraint = alpha_constraint
        self.alpha: TensorLike | None = None
        from zero_keras.initializers import get

        self._alpha_init_fn = get(alpha_initializer)

    def build(self, input_shape: ShapeTuple) -> None:
        """Allocate the learnable ``alpha`` slope weight tensor.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if input_shape is not None:
            shape = input_shape[1:]
        else:
            shape = ()
        self.alpha = self._alpha_init_fn(shape=shape)
        super().build(input_shape)

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply Parametric ReLU with slope ``self.alpha`` to ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: PReLU-activated tensor.
        """
        from zero_keras import ops as backend_ops
        from zero_keras.activations import _to_tensor, _wrap

        inputs_t = _to_tensor(inputs)
        alpha_t = self.alpha.data if hasattr(self.alpha, "data") else None
        if alpha_t is None:
            alpha_t = 0.0
        alpha_t = _to_tensor(alpha_t)
        zero_t = backend_ops.asarray(
            0.0, dtype=(inputs_t.dtype if hasattr(inputs_t, "dtype") else "float32")
        )
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

    def __init__(
        self,
        max_value: float | None = None,
        negative_slope: float = 0.0,
        threshold: float = 0.0,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the ReLU activation layer.

        Args:
            max_value: Optional maximum saturation value.
            negative_slope: Slope coefficient for values below ``threshold``.
            threshold: Threshold below which ``negative_slope`` applies.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.max_value = max_value
        self.negative_slope = float(negative_slope)
        self.threshold = float(threshold)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply Rectified Linear Unit activation to ``inputs``.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: ReLU-activated tensor.
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

    def __init__(self, axis: int | Sequence[int] = -1, **kwargs: TensorLike) -> None:
        """Initialize the Softmax activation layer.

        Args:
            axis: Axis or axes along which softmax normalization is computed.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply softmax normalization to ``inputs`` along ``self.axis``.

        Args:
            inputs: Input logits tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Probability distribution tensor summing to 1 along ``self.axis``.
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

    def __init__(self, negative_slope: float = 0.3, **kwargs: TensorLike) -> None:
        """Initialize the LeakyReLU activation layer.

        Args:
            negative_slope: Slope coefficient for negative inputs. Defaults to ``0.3``.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.negative_slope = float(negative_slope)

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply LeakyReLU activation to ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: LeakyReLU-activated tensor.
        """
        inputs = _to_tensor(inputs)
        return _wrap(ops.maximum(inputs, ops.multiply(_to_tensor(self.negative_slope), inputs)))


class ThresholdedReLU(Layer):
    """Thresholded Rectified Linear Unit activation layer."""

    def __init__(self, theta: float = 1.0, **kwargs: TensorLike) -> None:
        """Initialize the ThresholdedReLU activation layer.

        Args:
            theta: Float >= 0.0 threshold location of activation.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.theta = float(theta)

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply ThresholdedReLU activation to ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Thresholded tensor.
        """
        _ = kwargs
        from zero_keras import activations

        return activations.threshold(inputs, threshold=self.theta, default_value=0.0)

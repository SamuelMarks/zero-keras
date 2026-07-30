from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer


class ActivityRegularization(Layer):
    """Layer that applies an update to the cost function based input activity.

    Args:
        l1: L1 regularization factor (positive float).
        l2: L2 regularization factor (positive float).

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.

    """

    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        """Function docstring.

        Args:
            l1: Description.
            l2: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2

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

        loss = 0.0
        if self.l1 != 0.0:
            loss = loss + self.l1 * ops.sum(ops.abs(inputs))
        if self.l2 != 0.0:
            loss = loss + self.l2 * ops.sum(ops.square(inputs))

        if self.l1 != 0.0 or self.l2 != 0.0:
            self.add_loss(loss)

        return _wrap(inputs)


class AlphaDropout(Layer):
    """Applies Alpha Dropout to the input.

    Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
    to their original values, in order to ensure the self-normalizing property
    even after this dropout.
    Alpha Dropout fits well to Scaled Exponential Linear Units (SELU) by
    randomly setting activations to the negative saturation value.

    Args:
        rate: Float between 0 and 1. The multiplicative noise will have
            standard deviation `sqrt(rate / (1 - rate))`.
        noise_shape: 1D integer tensor representing the shape of the
            binary alpha dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the alpha dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding alpha dropout) or in inference mode
            (doing nothing).

    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        """Function docstring.

        Args:
            rate: Description.
            noise_shape: Description.
            seed: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

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
        if training:
            return _wrap(
                ops.dropout(
                    inputs, rate=self.rate, noise_shape=self.noise_shape, seed=self.seed
                )
            )
        return _wrap(inputs)


class GaussianDropout(Layer):
    """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Args:
        rate: Float, drop probability (as with `Dropout`).
            The multiplicative noise will have
            standard deviation `sqrt(rate / (1 - rate))`.
        seed: Integer, optional random seed to enable deterministic behavior.

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
        return _wrap(inputs)


class GaussianNoise(Layer):
    """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    Args:
        stddev: Float, standard deviation of the noise distribution.
        seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding noise) or in inference mode (doing nothing).

    """

    def __init__(self, stddev, **kwargs):
        """Function docstring.

        Args:
            stddev: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.stddev = stddev

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
        return _wrap(inputs)


class SpatialDropout1D(Layer):
    """Spatial 1D version of Dropout.

    This layer performs the same function as Dropout, however, it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, `SpatialDropout1D` will help promote independence
    between feature maps and should be used instead.

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.

    Call arguments:
        inputs: A 3D tensor.
        training: Python boolean indicating whether the layer
            should behave in training mode (applying dropout)
            or in inference mode (pass-through).

    Input shape:
        3D tensor with shape: `(samples, timesteps, channels)`

    Output shape: Same as input.

    Reference:

    - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)

    """

    def __init__(self, rate, data_format=None, **kwargs):
        """Function docstring.

        Args:
            rate: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rate = rate
        self.data_format = data_format or "channels_last"
        self.rank = 1

    def call(self, inputs, training=None, **kwargs):
        """Function docstring.

        Args:
            inputs: Description.
            training: Description.
            kwargs: Description.
        """
        inputs = _to_tensor(inputs)
        if not training:
            return _wrap(inputs)
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            noise_shape = (input_shape[0], 1, input_shape[-1])
        else:  # pragma: no cover
            noise_shape = (input_shape[0], input_shape[1], 1)

        return _wrap(ops.dropout(inputs, rate=self.rate, noise_shape=noise_shape))


class SpatialDropout2D(Layer):
    """Spatial 2D version of Dropout.

    This version performs the same function as Dropout, however, it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, `SpatialDropout2D` will help promote independence
    between feature maps and should be used instead.

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        data_format: `"channels_first"` or `"channels_last"`.
            In `"channels_first"` mode, the channels dimension (the depth)
            is at index 1, in `"channels_last"` mode is it at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.

    Call arguments:
        inputs: A 4D tensor.
        training: Python boolean indicating whether the layer
            should behave in training mode (applying dropout)
            or in inference mode (pass-through).

    Input shape:
        4D tensor with shape: `(samples, channels, rows, cols)` if
            data_format='channels_first'
        or 4D tensor with shape: `(samples, rows, cols, channels)` if
            data_format='channels_last'.

    Output shape: Same as input.

    Reference:

    - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)

    """

    def __init__(self, rate, data_format=None, **kwargs):
        """Function docstring.

        Args:
            rate: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rate = rate
        self.data_format = data_format or "channels_last"
        self.rank = 2

    def call(self, inputs, training=None, **kwargs):
        """Function docstring.

        Args:
            inputs: Description.
            training: Description.
            kwargs: Description.
        """
        inputs = _to_tensor(inputs)
        if not training:
            return _wrap(inputs)
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            noise_shape = (input_shape[0], 1, 1, input_shape[-1])
        else:  # pragma: no cover
            noise_shape = (input_shape[0], input_shape[1], 1, 1)

        return _wrap(ops.dropout(inputs, rate=self.rate, noise_shape=noise_shape))


class SpatialDropout3D(Layer):
    """Spatial 3D version of Dropout.

    This version performs the same function as Dropout, however, it drops
    entire 3D feature maps instead of individual elements. If adjacent voxels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout3D will help promote independence
    between feature maps and should be used instead.

    Args:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        data_format: `"channels_first"` or `"channels_last"`.
            In `"channels_first"` mode, the channels dimension (the depth)
            is at index 1, in `"channels_last"` mode is it at index 4.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.

    Call arguments:
        inputs: A 5D tensor.
        training: Python boolean indicating whether the layer
                should behave in training mode (applying dropout)
                or in inference mode (pass-through).

    Input shape:
        5D tensor with shape: `(samples, channels, dim1, dim2, dim3)` if
            data_format='channels_first'
        or 5D tensor with shape: `(samples, dim1, dim2, dim3, channels)` if
            data_format='channels_last'.

    Output shape: Same as input.

    Reference:

    - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)

    """

    def __init__(self, rate, data_format=None, **kwargs):
        """Function docstring.

        Args:
            rate: Description.
            data_format: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.rate = rate
        self.data_format = data_format or "channels_last"
        self.rank = 3

    def call(self, inputs, training=None, **kwargs):
        """Function docstring.

        Args:
            inputs: Description.
            training: Description.
            kwargs: Description.
        """
        inputs = _to_tensor(inputs)
        if not training:
            return _wrap(inputs)
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            noise_shape = (input_shape[0], 1, 1, 1, input_shape[-1])
        else:  # pragma: no cover
            noise_shape = (input_shape[0], input_shape[1], 1, 1, 1)

        return _wrap(ops.dropout(inputs, rate=self.rate, noise_shape=noise_shape))

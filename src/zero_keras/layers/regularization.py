"""Regularization and noise layers (`SpatialDropout1D/2D/3D`, `GaussianDropout`, `GaussianNoise`, `AlphaDropout`, `ActivityRegularization`) for zero_keras."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer
from zero_keras.types import DataFormatSpec, ShapeTuple, TensorLike


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

    def __init__(self, l1: float = 0.0, l2: float = 0.0, **kwargs: TensorLike) -> None:
        """Initialize the ActivityRegularization layer.

        Args:
            l1: L1 activity regularization factor. Defaults to ``0.0``.
            l2: L2 activity regularization factor. Defaults to ``0.0``.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.l1 = float(l1)
        self.l2 = float(l2)

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Register L1/L2 activity regularization loss and return ``inputs`` unchanged.

        Args:
            inputs: Input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Input tensor unchanged.
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

    def __init__(
        self,
        rate: float,
        noise_shape: ShapeTuple | None = None,
        seed: int | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the AlphaDropout layer.

        Args:
            rate: Drop probability in ``[0, 1]``.
            noise_shape: Optional shape tuple for broadcasting the dropout mask.
            seed: Optional integer random seed.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.noise_shape = noise_shape
        self.seed = seed

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Apply AlphaDropout to ``inputs`` when ``training=True``.

        Args:
            inputs: Input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Dropped-out tensor in training mode, or unchanged input in inference.
        """
        inputs = _to_tensor(inputs)
        if training:
            return _wrap(
                ops.dropout(inputs, rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)
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

    def __init__(
        self,
        rate: float,
        seed: int | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the GaussianDropout layer.

        Args:
            rate: Drop probability determining multiplicative Gaussian noise variance.
            seed: Optional random seed.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.seed = seed

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Apply multiplicative Gaussian noise to ``inputs``.

        Args:
            inputs: Input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Regularized output tensor.
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

    def __init__(
        self,
        stddev: float,
        seed: int | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the GaussianNoise layer.

        Args:
            stddev: Standard deviation of the additive Gaussian noise.
            seed: Optional random seed.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.stddev = float(stddev)
        self.seed = seed

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Apply additive zero-centered Gaussian noise to ``inputs``.

        Args:
            inputs: Input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Regularized output tensor.
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

    def __init__(
        self,
        rate: float,
        seed: int | None = None,
        name: str | None = None,
        dtype: str | None = None,
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SpatialDropout1D layer.

        Args:
            rate: Fraction of 1D feature maps to drop in ``[0, 1]``.
            seed: Optional random seed.
            name: Optional layer name.
            dtype: Optional layer dtype.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.rate = float(rate)
        self.seed = seed
        self.data_format = data_format or "channels_last"
        self.rank = 1

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Apply 1D spatial dropout across whole feature maps during training.

        Args:
            inputs: 3D input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Spatially dropped-out tensor in training mode, or unchanged input in inference.
        """
        inputs = _to_tensor(inputs)
        if not training:
            return _wrap(inputs)
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            noise_shape = (input_shape[0], 1, input_shape[-1])
        else:
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

    def __init__(
        self,
        rate: float,
        data_format: DataFormatSpec = None,
        seed: int | None = None,
        name: str | None = None,
        dtype: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SpatialDropout2D layer.

        Args:
            rate: Fraction of 2D feature maps to drop in ``[0, 1]``.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            seed: Optional random seed.
            name: Optional layer name.
            dtype: Optional layer dtype.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.rate = float(rate)
        self.seed = seed
        self.data_format = data_format or "channels_last"
        self.rank = 2

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Apply 2D spatial dropout across whole feature maps during training.

        Args:
            inputs: 4D input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Spatially dropped-out tensor in training mode, or unchanged input in inference.
        """
        inputs = _to_tensor(inputs)
        if not training:
            return _wrap(inputs)
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            noise_shape = (input_shape[0], 1, 1, input_shape[-1])
        else:
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

    def __init__(
        self,
        rate: float,
        data_format: DataFormatSpec = None,
        seed: int | None = None,
        name: str | None = None,
        dtype: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SpatialDropout3D layer.

        Args:
            rate: Fraction of 3D feature maps to drop in ``[0, 1]``.
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            seed: Optional random seed.
            name: Optional layer name.
            dtype: Optional layer dtype.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.rate = float(rate)
        self.seed = seed
        self.data_format = data_format or "channels_last"
        self.rank = 3

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Apply 3D spatial dropout across whole feature maps during training.

        Args:
            inputs: 5D input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Spatially dropped-out tensor in training mode, or unchanged input in inference.
        """
        inputs = _to_tensor(inputs)
        if not training:
            return _wrap(inputs)
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            noise_shape = (input_shape[0], 1, 1, 1, input_shape[-1])
        else:
            noise_shape = (input_shape[0], input_shape[1], 1, 1, 1)

        return _wrap(ops.dropout(inputs, rate=self.rate, noise_shape=noise_shape))

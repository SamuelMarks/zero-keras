"""Spatial and temporal pooling layers for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer
from zero_keras.types import ConfigDict, DataFormatSpec, TensorLike


class AveragePooling1D(Layer):
    """Average pooling for temporal data.

    Downsamples the input representation by taking the average value over the
    window defined by `pool_size`. The window is shifted by `strides`.  The
    resulting output when using "valid" padding option has a shape of:
    `output_shape = (input_shape - pool_size + 1) / strides)`

    The resulting output shape when using the "same" padding option is:
    `output_shape = input_shape / strides`

    Args:
        pool_size: int, size of the max pooling window.
        strides: int or None. Specifies how much the pooling window moves
            for each pooling step. If None, it will default to `pool_size`.
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

    Input shape:

    - If `data_format="channels_last"`:
        3D tensor with shape `(batch_size, steps, features)`.
    - If `data_format="channels_first"`:
        3D tensor with shape `(batch_size, features, steps)`.

    Output shape:

    - If `data_format="channels_last"`:
        3D tensor with shape `(batch_size, downsampled_steps, features)`.
    - If `data_format="channels_first"`:
        3D tensor with shape `(batch_size, features, downsampled_steps)`.

    Examples:
    `strides=1` and `padding="valid"`:

    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> x = np.reshape(x, [1, 5, 1])
    >>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,
    ...    strides=1, padding="valid")
    >>> avg_pool_1d(x)

    `strides=2` and `padding="valid"`:

    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> x = np.reshape(x, [1, 5, 1])
    >>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,
    ...    strides=2, padding="valid")
    >>> avg_pool_1d(x)

    `strides=1` and `padding="same"`:

    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> x = np.reshape(x, [1, 5, 1])
    >>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,
    ...    strides=1, padding="same")
    >>> avg_pool_1d(x)

    """

    def __init__(
        self,
        pool_size: int | Sequence[int],
        strides: int | Sequence[int] | None = None,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the AveragePooling1D layer.

        Args:
            pool_size: Window size of the 1D average pooling operation.
            strides: Stride step size; defaults to ``pool_size``.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            name: Optional layer name.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.rank = 1
        self.pool_size = (pool_size,) * 1 if isinstance(pool_size, int) else tuple(pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = (self.strides,) * 1 if isinstance(self.strides, int) else tuple(self.strides)
        self.padding = str(padding)
        self.data_format = data_format or "channels_last"

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply 1D average pooling to ``inputs``.

        Args:
            inputs: 3D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Downsampled 3D tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            window_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        else:
            window_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)

        out = ops.avg_pool(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=self.padding.upper(),
        )

        return _wrap(out)


class AveragePooling2D(Layer):
    """Average pooling operation for 2D spatial data.

    Downsamples the input along its spatial dimensions (height and width)
    by taking the average value over an input window
    (of size defined by `pool_size`) for each channel of the input.
    The window is shifted by `strides` along each dimension.

    The resulting output when using the `"valid"` padding option has a spatial
    shape (number of rows or columns) of:
    `output_shape = math.floor((input_shape - pool_size) / strides) + 1`
    (when `input_shape >= pool_size`)

    The resulting output shape when using the `"same"` padding option is:
    `output_shape = math.floor((input_shape - 1) / strides) + 1`

    Args:
        pool_size: int or tuple of 2 integers, factors by which to downscale
            (dim1, dim2). If only one integer is specified, the same
            window length will be used for all dimensions.
        strides: int or tuple of 2 integers, or None. Strides values. If None,
            it will default to `pool_size`. If only one int is specified, the
            same stride size will be used for all dimensions.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.

    Input shape:

    - If `data_format="channels_last"`:
        4D tensor with shape `(batch_size, height, width, channels)`.
    - If `data_format="channels_first"`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:

    - If `data_format="channels_last"`:
        4D tensor with shape
        `(batch_size, pooled_height, pooled_width, channels)`.
    - If `data_format="channels_first"`:
        4D tensor with shape
        `(batch_size, channels, pooled_height, pooled_width)`.

    Examples:
    `strides=(1, 1)` and `padding="valid"`:

    >>> x = np.array([[1., 2., 3.],
    ...               [4., 5., 6.],
    ...               [7., 8., 9.]])
    >>> x = np.reshape(x, [1, 3, 3, 1])
    >>> avg_pool_2d = keras.layers.AveragePooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding="valid")
    >>> avg_pool_2d(x)

    `strides=(2, 2)` and `padding="valid"`:

    >>> x = np.array([[1., 2., 3., 4.],
    ...              [5., 6., 7., 8.],
    ...              [9., 10., 11., 12.]])
    >>> x = np.reshape(x, [1, 3, 4, 1])
    >>> avg_pool_2d = keras.layers.AveragePooling2D(pool_size=(2, 2),
    ...    strides=(2, 2), padding="valid")
    >>> avg_pool_2d(x)

    `stride=(1, 1)` and `padding="same"`:

    >>> x = np.array([[1., 2., 3.],
    ...                  [4., 5., 6.],
    ...                  [7., 8., 9.]])
    >>> x = np.reshape(x, [1, 3, 3, 1])
    >>> avg_pool_2d = keras.layers.AveragePooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding="same")
    >>> avg_pool_2d(x)

    """

    def __init__(
        self,
        pool_size: int | Sequence[int],
        strides: int | Sequence[int] | None = None,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the AveragePooling2D layer.

        Args:
            pool_size: Window size of the 2D average pooling operation.
            strides: Stride step size; defaults to ``pool_size``.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            name: Optional layer name.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.rank = 2
        self.pool_size = (pool_size,) * 2 if isinstance(pool_size, int) else tuple(pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = (self.strides,) * 2 if isinstance(self.strides, int) else tuple(self.strides)
        self.padding = str(padding)
        self.data_format = data_format or "channels_last"

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply 2D average pooling to ``inputs``.

        Args:
            inputs: 4D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Downsampled 4D tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            window_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        else:
            window_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)

        out = ops.avg_pool(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=self.padding.upper(),
        )

        return _wrap(out)


class AveragePooling3D(Layer):
    """Average pooling operation for 3D data (spatial or spatio-temporal).

    Downsamples the input along its spatial dimensions (depth, height, and
    width) by taking the average value over an input window (of size defined by
    `pool_size`) for each channel of the input. The window is shifted by
    `strides` along each dimension.

    Args:
        pool_size: int or tuple of 3 integers, factors by which to downscale
            (dim1, dim2, dim3). If only one integer is specified, the same
            window length will be used for all dimensions.
        strides: int or tuple of 3 integers, or None. Strides values. If None,
            it will default to `pool_size`. If only one int is specified, the
            same stride size will be used for all dimensions.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. If you never set it, then it
            will be `"channels_last"`.

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
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`

    Example:
    ```python
    depth = 30
    height = 30
    width = 30
    channels = 3

    inputs = keras.layers.Input(shape=(depth, height, width, channels))
    layer = keras.layers.AveragePooling3D(pool_size=3)
    outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
    ```

    """

    def __init__(
        self,
        pool_size: int | Sequence[int],
        strides: int | Sequence[int] | None = None,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the AveragePooling3D layer.

        Args:
            pool_size: Window size of the 3D average pooling operation.
            strides: Stride step size; defaults to ``pool_size``.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            name: Optional layer name.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.rank = 3
        self.pool_size = (pool_size,) * 3 if isinstance(pool_size, int) else tuple(pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = (self.strides,) * 3 if isinstance(self.strides, int) else tuple(self.strides)
        self.padding = str(padding)
        self.data_format = data_format or "channels_last"

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply 3D average pooling to ``inputs``.

        Args:
            inputs: 5D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Downsampled 5D tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            window_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        else:
            window_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)

        out = ops.avg_pool(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=self.padding.upper(),
        )

        return _wrap(out)


class MaxPooling1D(Layer):
    """Max pooling operation for 1D temporal data.

    Downsamples the input representation by taking the maximum value over a
    spatial window of size `pool_size`. The window is shifted by `strides`.

    The resulting output when using the `"valid"` padding option has a shape of:
    `output_shape = (input_shape - pool_size + 1) / strides)`.

    The resulting output shape when using the `"same"` padding option is:
    `output_shape = input_shape / strides`

    Args:
        pool_size: int, size of the max pooling window.
        strides: int or None. Specifies how much the pooling window moves
            for each pooling step. If None, it will default to `pool_size`.
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

    Input shape:

    - If `data_format="channels_last"`:
        3D tensor with shape `(batch_size, steps, features)`.
    - If `data_format="channels_first"`:
        3D tensor with shape `(batch_size, features, steps)`.

    Output shape:

    - If `data_format="channels_last"`:
        3D tensor with shape `(batch_size, downsampled_steps, features)`.
    - If `data_format="channels_first"`:
        3D tensor with shape `(batch_size, features, downsampled_steps)`.

    Examples:
    `strides=1` and `padding="valid"`:

    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> x = np.reshape(x, [1, 5, 1])
    >>> max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,
    ...    strides=1, padding="valid")
    >>> max_pool_1d(x)

    `strides=2` and `padding="valid"`:

    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> x = np.reshape(x, [1, 5, 1])
    >>> max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,
    ...    strides=2, padding="valid")
    >>> max_pool_1d(x)

    `strides=1` and `padding="same"`:

    >>> x = np.array([1., 2., 3., 4., 5.])
    >>> x = np.reshape(x, [1, 5, 1])
    >>> max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,
    ...    strides=1, padding="same")
    >>> max_pool_1d(x)

    """

    def __init__(
        self,
        pool_size: int | Sequence[int] = 2,
        strides: int | Sequence[int] | None = None,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MaxPooling1D layer.

        Args:
            pool_size: Window size of the 1D max pooling operation.
            strides: Stride step size; defaults to ``pool_size``.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            name: Optional layer name.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.rank = 1
        self.pool_size = (pool_size,) * 1 if isinstance(pool_size, int) else tuple(pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = (self.strides,) * 1 if isinstance(self.strides, int) else tuple(self.strides)
        self.padding = str(padding)
        self.data_format = data_format or "channels_last"

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply 1D max pooling to ``inputs``.

        Args:
            inputs: 3D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Downsampled 3D tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            window_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        else:
            window_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)

        out = ops.max_pool(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=self.padding.upper(),
        )

        return _wrap(out)


class MaxPooling2D(Layer):
    """Max pooling operation for 2D spatial data.

    Downsamples the input along its spatial dimensions (height and width)
    by taking the maximum value over an input window
    (of size defined by `pool_size`) for each channel of the input.
    The window is shifted by `strides` along each dimension.

    The resulting output when using the `"valid"` padding option has a spatial
    shape (number of rows or columns) of:
    `output_shape = math.floor((input_shape - pool_size) / strides) + 1`
    (when `input_shape >= pool_size`)

    The resulting output shape when using the `"same"` padding option is:
    `output_shape = math.floor((input_shape - 1) / strides) + 1`

    Args:
        pool_size: int or tuple of 2 integers, factors by which to downscale
            (dim1, dim2). If only one integer is specified, the same
            window length will be used for all dimensions.
        strides: int or tuple of 2 integers, or None. Strides values. If None,
            it will default to `pool_size`. If only one int is specified, the
            same stride size will be used for all dimensions.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.

    Input shape:

    - If `data_format="channels_last"`:
        4D tensor with shape `(batch_size, height, width, channels)`.
    - If `data_format="channels_first"`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:

    - If `data_format="channels_last"`:
        4D tensor with shape
        `(batch_size, pooled_height, pooled_width, channels)`.
    - If `data_format="channels_first"`:
        4D tensor with shape
        `(batch_size, channels, pooled_height, pooled_width)`.

    Examples:
    `strides=(1, 1)` and `padding="valid"`:

    >>> x = np.array([[1., 2., 3.],
    ...               [4., 5., 6.],
    ...               [7., 8., 9.]])
    >>> x = np.reshape(x, [1, 3, 3, 1])
    >>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding="valid")
    >>> max_pool_2d(x)

    `strides=(2, 2)` and `padding="valid"`:

    >>> x = np.array([[1., 2., 3., 4.],
    ...               [5., 6., 7., 8.],
    ...               [9., 10., 11., 12.]])
    >>> x = np.reshape(x, [1, 3, 4, 1])
    >>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(2, 2), padding="valid")
    >>> max_pool_2d(x)

    `stride=(1, 1)` and `padding="same"`:

    >>> x = np.array([[1., 2., 3.],
    ...               [4., 5., 6.],
    ...               [7., 8., 9.]])
    >>> x = np.reshape(x, [1, 3, 3, 1])
    >>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding="same")
    >>> max_pool_2d(x)

    """

    def __init__(
        self,
        pool_size: int | Sequence[int] = (2, 2),
        strides: int | Sequence[int] | None = None,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MaxPooling2D layer.

        Args:
            pool_size: Window size of the 2D max pooling operation.
            strides: Stride step size; defaults to ``pool_size``.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            name: Optional layer name.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.rank = 2
        self.pool_size = (pool_size,) * 2 if isinstance(pool_size, int) else tuple(pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = (self.strides,) * 2 if isinstance(self.strides, int) else tuple(self.strides)
        self.padding = str(padding)
        self.data_format = data_format or "channels_last"

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply 2D max pooling to ``inputs``.

        Args:
            inputs: 4D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Downsampled 4D tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            window_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        else:
            window_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)

        out = ops.max_pool(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=self.padding.upper(),
        )

        return _wrap(out)


class MaxPooling3D(Layer):
    """Max pooling operation for 3D data (spatial or spatio-temporal).

    Downsamples the input along its spatial dimensions (depth, height, and
    width) by taking the maximum value over an input window (of size defined by
    `pool_size`) for each channel of the input. The window is shifted by
    `strides` along each dimension.

    Args:
        pool_size: int or tuple of 3 integers, factors by which to downscale
            (dim1, dim2, dim3). If only one integer is specified, the same
            window length will be used for all dimensions.
        strides: int or tuple of 3 integers, or None. Strides values. If None,
            it will default to `pool_size`. If only one int is specified, the
            same stride size will be used for all dimensions.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. If you never set it, then it
            will be `"channels_last"`.

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
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`

    Example:
    ```python
    depth = 30
    height = 30
    width = 30
    channels = 3

    inputs = keras.layers.Input(shape=(depth, height, width, channels))
    layer = keras.layers.MaxPooling3D(pool_size=3)
    outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
    ```

    """

    def __init__(
        self,
        pool_size: int | Sequence[int] = (2, 2, 2),
        strides: int | Sequence[int] | None = None,
        padding: str = "valid",
        data_format: DataFormatSpec = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MaxPooling3D layer.

        Args:
            pool_size: Window size of the 3D max pooling operation.
            strides: Stride step size; defaults to ``pool_size``.
            padding: Padding mode (``"valid"`` or ``"same"``).
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            name: Optional layer name.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.rank = 3
        self.pool_size = (pool_size,) * 3 if isinstance(pool_size, int) else tuple(pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = (self.strides,) * 3 if isinstance(self.strides, int) else tuple(self.strides)
        self.padding = str(padding)
        self.data_format = data_format or "channels_last"

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply 3D max pooling to ``inputs``.

        Args:
            inputs: 5D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Downsampled 5D tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            window_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        else:
            window_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)

        out = ops.max_pool(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=self.padding.upper(),
        )

        return _wrap(out)


class GlobalAveragePooling1D(Layer):
    """Global average pooling operation for temporal data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            temporal dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Call arguments:
        inputs: A 3D tensor.
        mask: Binary tensor of shape `(batch_size, steps)` indicating whether
            a given step should be masked (excluded from the average).

    Input shape:

    - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
    - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, features)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            3D tensor with shape `(batch_size, 1, features)`
        - If `data_format="channels_first"`:
            3D tensor with shape `(batch_size, features, 1)`

    Example:
    >>> x = np.random.rand(2, 3, 4)
    >>> y = keras.layers.GlobalAveragePooling1D()(x)
    >>> y.shape
    (2, 4)

    """

    def __init__(
        self, data_format: DataFormatSpec = None, keepdims: bool = False, **kwargs: TensorLike
    ) -> None:
        """Initialize the GlobalAveragePooling1D layer.

        Args:
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            keepdims: Whether to retain reduced spatial dimensions with size 1.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.data_format = data_format or "channels_last"
        self.keepdims = bool(keepdims)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the global mean across the 1D temporal axis of ``inputs``.

        Args:
            inputs: 3D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Pooled 2D (or 3D if ``keepdims=True``) tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            axes = tuple(range(2, 2 + self.rank))
        else:
            axes = tuple(range(1, 1 + self.rank))

        out = ops.mean(inputs, axis=axes, keepdims=self.keepdims)

        return _wrap(out)


class GlobalAveragePooling2D(Layer):
    """Global average pooling operation for 2D data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, height, weight)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            spatial dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:

    - If `data_format='channels_last'`:
        4D tensor with shape:
        `(batch_size, height, width, channels)`
    - If `data_format='channels_first'`:
        4D tensor with shape:
        `(batch_size, channels, height, width)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, channels)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            4D tensor with shape `(batch_size, 1, 1, channels)`
        - If `data_format="channels_first"`:
            4D tensor with shape `(batch_size, channels, 1, 1)`

    Example:
    >>> x = np.random.rand(2, 4, 5, 3)
    >>> y = keras.layers.GlobalAveragePooling2D()(x)
    >>> y.shape
    (2, 3)

    """

    def __init__(
        self, data_format: DataFormatSpec = None, keepdims: bool = False, **kwargs: TensorLike
    ) -> None:
        """Initialize the GlobalAveragePooling2D layer.

        Args:
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            keepdims: Whether to retain reduced spatial dimensions with size 1.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.data_format = data_format or "channels_last"
        self.keepdims = bool(keepdims)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the global mean across the 2D spatial axes of ``inputs``.

        Args:
            inputs: 4D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Pooled 2D (or 4D if ``keepdims=True``) tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            axes = tuple(range(2, 2 + self.rank))
        else:
            axes = tuple(range(1, 1 + self.rank))

        out = ops.mean(inputs, axis=axes, keepdims=self.keepdims)

        return _wrap(out)


class GlobalAveragePooling3D(Layer):
    """Global average pooling operation for 3D data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. If you never set it, then it
            will be `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            spatial dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:

    - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, channels)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            5D tensor with shape `(batch_size, 1, 1, 1, channels)`
        - If `data_format="channels_first"`:
            5D tensor with shape `(batch_size, channels, 1, 1, 1)`

    Example:
    >>> x = np.random.rand(2, 4, 5, 4, 3)
    >>> y = keras.layers.GlobalAveragePooling3D()(x)
    >>> y.shape
    (2, 3)

    """

    def __init__(
        self, data_format: DataFormatSpec = None, keepdims: bool = False, **kwargs: TensorLike
    ) -> None:
        """Initialize the GlobalAveragePooling3D layer.

        Args:
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            keepdims: Whether to retain reduced spatial dimensions with size 1.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = 3
        self.data_format = data_format or "channels_last"
        self.keepdims = bool(keepdims)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the global mean across the 3D spatial axes of ``inputs``.

        Args:
            inputs: 5D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Pooled 2D (or 5D if ``keepdims=True``) tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            axes = tuple(range(2, 2 + self.rank))
        else:
            axes = tuple(range(1, 1 + self.rank))

        out = ops.mean(inputs, axis=axes, keepdims=self.keepdims)

        return _wrap(out)


class GlobalMaxPooling1D(Layer):
    """Global max pooling operation for temporal data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            temporal dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:

    - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
    - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, features)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            3D tensor with shape `(batch_size, 1, features)`
        - If `data_format="channels_first"`:
            3D tensor with shape `(batch_size, features, 1)`

    Example:
    >>> x = np.random.rand(2, 3, 4)
    >>> y = keras.layers.GlobalMaxPooling1D()(x)
    >>> y.shape
    (2, 4)

    """

    def __init__(
        self, data_format: DataFormatSpec = None, keepdims: bool = False, **kwargs: TensorLike
    ) -> None:
        """Initialize the GlobalMaxPooling1D layer.

        Args:
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            keepdims: Whether to retain reduced spatial dimensions with size 1.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = 1
        self.data_format = data_format or "channels_last"
        self.keepdims = bool(keepdims)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the global maximum across the 1D temporal axis of ``inputs``.

        Args:
            inputs: 3D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Pooled 2D (or 3D if ``keepdims=True``) tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            axes = tuple(range(2, 2 + self.rank))
        else:
            axes = tuple(range(1, 1 + self.rank))

        out = ops.max(inputs, axis=axes, keepdims=self.keepdims)

        return _wrap(out)


class GlobalMaxPooling2D(Layer):
    """Global max pooling operation for 2D data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, height, weight)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            spatial dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:

    - If `data_format='channels_last'`:
        4D tensor with shape:
        `(batch_size, height, width, channels)`
    - If `data_format='channels_first'`:
        4D tensor with shape:
        `(batch_size, channels, height, width)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, channels)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            4D tensor with shape `(batch_size, 1, 1, channels)`
        - If `data_format="channels_first"`:
            4D tensor with shape `(batch_size, channels, 1, 1)`

    Example:
    >>> x = np.random.rand(2, 4, 5, 3)
    >>> y = keras.layers.GlobalMaxPooling2D()(x)
    >>> y.shape
    (2, 3)

    """

    def __init__(
        self, data_format: DataFormatSpec = None, keepdims: bool = False, **kwargs: TensorLike
    ) -> None:
        """Initialize the GlobalMaxPooling2D layer.

        Args:
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            keepdims: Whether to retain reduced spatial dimensions with size 1.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = 2
        self.data_format = data_format or "channels_last"
        self.keepdims = bool(keepdims)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the global maximum across the 2D spatial axes of ``inputs``.

        Args:
            inputs: 4D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Pooled 2D (or 4D if ``keepdims=True``) tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            axes = tuple(range(2, 2 + self.rank))
        else:
            axes = tuple(range(1, 1 + self.rank))

        out = ops.max(inputs, axis=axes, keepdims=self.keepdims)

        return _wrap(out)


class GlobalMaxPooling3D(Layer):
    """Global max pooling operation for 3D data.

    Args:
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. If you never set it, then it
            will be `"channels_last"`.
        keepdims: A boolean, whether to keep the temporal dimension or not.
            If `keepdims` is `False` (default), the rank of the tensor is
            reduced for spatial dimensions. If `keepdims` is `True`, the
            spatial dimension are retained with length 1.
            The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:

    - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:

    - If `keepdims=False`:
        2D tensor with shape `(batch_size, channels)`.
    - If `keepdims=True`:
        - If `data_format="channels_last"`:
            5D tensor with shape `(batch_size, 1, 1, 1, channels)`
        - If `data_format="channels_first"`:
            5D tensor with shape `(batch_size, channels, 1, 1, 1)`

    Example:
    >>> x = np.random.rand(2, 4, 5, 4, 3)
    >>> y = keras.layers.GlobalMaxPooling3D()(x)
    >>> y.shape
    (2, 3)

    """

    def __init__(
        self, data_format: DataFormatSpec = None, keepdims: bool = False, **kwargs: TensorLike
    ) -> None:
        """Initialize the GlobalMaxPooling3D layer.

        Args:
            data_format: Channel ordering (``"channels_last"`` or ``"channels_first"``).
            keepdims: Whether to retain reduced spatial dimensions with size 1.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = 3
        self.data_format = data_format or "channels_last"
        self.keepdims = bool(keepdims)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the global maximum across the 3D spatial axes of ``inputs``.

        Args:
            inputs: 5D input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Pooled 2D (or 5D if ``keepdims=True``) tensor.
        """
        inputs = _to_tensor(inputs)

        if self.data_format == "channels_first":
            axes = tuple(range(2, 2 + self.rank))
        else:
            axes = tuple(range(1, 1 + self.rank))

        out = ops.max(inputs, axis=axes, keepdims=self.keepdims)

        return _wrap(out)


class _AdaptivePoolingBase(Layer):
    """Base layer for 1D/2D/3D adaptive average and max pooling."""

    def __init__(
        self,
        rank: int,
        mode: str,
        output_size: int | Sequence[int],
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the adaptive pooling base layer.

        Args:
            rank: Spatial rank (1, 2, or 3).
            mode: Pooling mode (``"avg"`` or ``"max"``).
            output_size: Target spatial output size(s).
            data_format: Data format (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.rank = rank
        self.mode = mode
        if isinstance(output_size, int):
            self.output_size: tuple[int, ...] = (output_size,) * rank
        else:
            self.output_size = tuple(int(v) for v in output_size)
        self.data_format = data_format or "channels_last"

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply adaptive pooling over the spatial dimensions of ``inputs``.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Pooled output tensor.
        """
        _ = args, kwargs
        t = _to_tensor(inputs)
        if self.mode == "avg":
            out = ops.adaptive_average_pool(t, self.output_size, data_format=self.data_format)
        else:
            out = ops.adaptive_max_pool(t, self.output_size, data_format=self.data_format)
        return _wrap(out)

    def get_config(self) -> ConfigDict:
        """Return serializable layer configuration.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "output_size": self.output_size,
                "data_format": self.data_format,
            }
        )
        return cfg


class AdaptiveAveragePooling1D(_AdaptivePoolingBase):
    """Adaptive average pooling layer for 1D temporal data."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize AdaptiveAveragePooling1D.

        Args:
            output_size: Target 1D spatial output size.
            data_format: Data format (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(1, "avg", output_size, data_format=data_format, **kwargs)


class AdaptiveAveragePooling2D(_AdaptivePoolingBase):
    """Adaptive average pooling layer for 2D spatial data."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize AdaptiveAveragePooling2D.

        Args:
            output_size: Target 2D spatial output size.
            data_format: Data format (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(2, "avg", output_size, data_format=data_format, **kwargs)


class AdaptiveAveragePooling3D(_AdaptivePoolingBase):
    """Adaptive average pooling layer for 3D volumetric data."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize AdaptiveAveragePooling3D.

        Args:
            output_size: Target 3D spatial output size.
            data_format: Data format (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(3, "avg", output_size, data_format=data_format, **kwargs)


class AdaptiveMaxPooling1D(_AdaptivePoolingBase):
    """Adaptive max pooling layer for 1D temporal data."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize AdaptiveMaxPooling1D.

        Args:
            output_size: Target 1D spatial output size.
            data_format: Data format (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(1, "max", output_size, data_format=data_format, **kwargs)


class AdaptiveMaxPooling2D(_AdaptivePoolingBase):
    """Adaptive max pooling layer for 2D spatial data."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize AdaptiveMaxPooling2D.

        Args:
            output_size: Target 2D spatial output size.
            data_format: Data format (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(2, "max", output_size, data_format=data_format, **kwargs)


class AdaptiveMaxPooling3D(_AdaptivePoolingBase):
    """Adaptive max pooling layer for 3D volumetric data."""

    def __init__(
        self,
        output_size: int | Sequence[int],
        data_format: DataFormatSpec = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize AdaptiveMaxPooling3D.

        Args:
            output_size: Target 3D spatial output size.
            data_format: Data format (``"channels_last"`` or ``"channels_first"``).
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(3, "max", output_size, data_format=data_format, **kwargs)

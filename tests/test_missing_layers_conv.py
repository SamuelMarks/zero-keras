"""Unit and behavioral tests for missing layers conv in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_conv1d_basic() -> None:
    """Test ``conv1d_basic`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Conv1D(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 8, 4)

    layer_cf = layers.Conv1D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = random.uniform((2, 3, 10), dtype="float32")
    out_cf = layer_cf(x_cf)
    assert out_cf.shape == (2, 4, 8)


def test_conv1d_transpose_basic() -> None:
    """Test ``conv1d_transpose_basic`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Conv1DTranspose(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 3), dtype="float32")
    _out = layer(x)

    layer_cf = layers.Conv1DTranspose(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = random.uniform((2, 3, 10), dtype="float32")
    _out_cf = layer_cf(x_cf)


def test_conv2d_basic() -> None:
    """Test ``conv2d_basic`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Conv2D(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 8, 8, 4)

    layer_cf = layers.Conv2D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = random.uniform((2, 3, 10, 10), dtype="float32")
    _out_cf = layer_cf(x_cf)


def test_conv2d_transpose_basic() -> None:
    """Test ``conv2d_transpose_basic`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Conv2DTranspose(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    _out = layer(x)

    layer_cf = layers.Conv2DTranspose(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = random.uniform((2, 3, 10, 10), dtype="float32")
    _out_cf = layer_cf(x_cf)


def test_conv3d_basic() -> None:
    """Test ``conv3d_basic`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Conv3D(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 10, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 8, 8, 8, 4)

    layer_cf = layers.Conv3D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = random.uniform((2, 3, 10, 10, 10), dtype="float32")
    _out_cf = layer_cf(x_cf)


def test_conv3d_transpose_basic() -> None:
    """Test ``conv3d_transpose_basic`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Conv3DTranspose(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 10, 10, 3), dtype="float32")
    _out = layer(x)

    layer_cf = layers.Conv3DTranspose(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = random.uniform((2, 3, 10, 10, 10), dtype="float32")
    _out_cf = layer_cf(x_cf)


def test_cropping1d() -> None:
    """Test ``cropping1d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Cropping1D(cropping=1)
    x = random.uniform((2, 10, 3), dtype="float32")
    _out = layer(x)
    layer_cf = layers.Cropping1D(cropping=1, data_format="channels_first")
    _out_cf = layer_cf(random.uniform((2, 3, 10), dtype="float32"))


def test_cropping2d() -> None:
    """Test ``cropping2d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Cropping2D(cropping=1)
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    _out = layer(x)
    layer_cf = layers.Cropping2D(cropping=1, data_format="channels_first")
    _out_cf = layer_cf(random.uniform((2, 3, 10, 10), dtype="float32"))


def test_cropping3d() -> None:
    """Test ``cropping3d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.Cropping3D(cropping=1)
    x = random.uniform((2, 10, 10, 10, 3), dtype="float32")
    _out = layer(x)
    layer_cf = layers.Cropping3D(cropping=1, data_format="channels_first")
    _out_cf = layer_cf(random.uniform((2, 3, 10, 10, 10), dtype="float32"))


def test_depthwise1d() -> None:
    """Test ``depthwise1d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.DepthwiseConv1D(kernel_size=3)
    x = random.uniform((2, 10, 3), dtype="float32")
    _out = layer(x)


def test_depthwise2d() -> None:
    """Test ``depthwise2d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.DepthwiseConv2D(kernel_size=3)
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    _out = layer(x)


def test_einsum_dense() -> None:
    """Test ``einsum_dense`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.EinsumDense(equation="abc,cd->abd", output_shape=(10, 4))
    x = random.uniform((2, 10, 3), dtype="float32")
    _out = layer(x)


def test_alpha_dropout() -> None:
    """Test ``alpha_dropout`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.AlphaDropout(0.5)
    layer(random.uniform((2, 10), dtype="float32"), training=True)


def test_separable_conv1d() -> None:
    """Test ``separable_conv1d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.SeparableConv1D(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 8, 4)


def test_separable_conv2d() -> None:
    """Test ``separable_conv2d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.SeparableConv2D(filters=4, kernel_size=3)
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 8, 8, 4)


def test_upsampling1d() -> None:
    """Test ``upsampling1d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.UpSampling1D(size=2)
    x = random.uniform((2, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 20, 3)


def test_upsampling2d() -> None:
    """Test ``upsampling2d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.UpSampling2D(size=(2, 2))
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 20, 20, 3)


def test_upsampling3d() -> None:
    """Test ``upsampling3d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.UpSampling3D(size=(2, 2, 2))
    x = random.uniform((2, 10, 10, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 20, 20, 20, 3)


def test_zeropadding1d() -> None:
    """Test ``zeropadding1d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.ZeroPadding1D(padding=1)
    x = random.uniform((2, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 12, 3)


def test_zeropadding2d() -> None:
    """Test ``zeropadding2d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.ZeroPadding2D(padding=(1, 1))
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 12, 12, 3)


def test_zeropadding3d() -> None:
    """Test ``zeropadding3d`` behavior and edge cases in ``test_missing_layers_conv``."""
    layer = layers.ZeroPadding3D(padding=(1, 1, 1))
    x = random.uniform((2, 10, 10, 10, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 12, 12, 12, 3)


def test_conv_built_and_call_direct() -> None:
    """Test ``conv_built_and_call_direct`` behavior and edge cases in ``test_missing_layers_conv``."""
    classes_to_test = [
        layers.Conv1D,
        layers.Conv1DTranspose,
        layers.Conv2D,
        layers.Conv2DTranspose,
        layers.Conv3D,
        layers.Conv3DTranspose,
        layers.DepthwiseConv1D,
        layers.DepthwiseConv2D,
        layers.SeparableConv1D,
        layers.SeparableConv2D,
    ]
    for cls in classes_to_test:
        kwargs = {"filters": 2, "kernel_size": 3}
        if "Depthwise" in cls.__name__:
            del kwargs["filters"]
        layer = cls(**kwargs)
        shape: tuple[int, ...] = (2, 5, 3)
        if "2D" in cls.__name__:
            shape = (2, 5, 5, 3)
        elif "3D" in cls.__name__:
            shape = (2, 5, 5, 5, 3)

        layer.build(shape)
        layer.build(shape)

        layer.built = False
        x = random.uniform(shape, dtype="float32")
        layer.call(x)


def test_padding_tuple_cases() -> None:
    """Test ``padding_tuple_cases`` behavior and edge cases in ``test_missing_layers_conv``."""
    layers.ZeroPadding1D(padding=(1,))
    layers.ZeroPadding2D(padding=(1, 2))
    layers.ZeroPadding3D(padding=(1, 2, 3))
    layers.Cropping1D(cropping=(1,))
    layers.Cropping2D(cropping=(1, 2))
    layers.Cropping3D(cropping=(1, 2, 3))


def test_channels_first_padding_cropping() -> None:
    """Test ``channels_first_padding_cropping`` behavior and edge cases in ``test_missing_layers_conv``."""
    x1 = random.uniform((2, 3, 10), dtype="float32")
    out = layers.ZeroPadding1D(padding=1, data_format="channels_first")(x1)
    assert out.shape == (2, 3, 12)
    out = layers.Cropping1D(cropping=1, data_format="channels_first")(x1)
    assert out.shape == (2, 3, 8)

    x2 = random.uniform((2, 3, 10, 10), dtype="float32")
    out = layers.ZeroPadding2D(padding=1, data_format="channels_first")(x2)
    assert out.shape == (2, 3, 12, 12)
    out = layers.Cropping2D(cropping=1, data_format="channels_first")(x2)
    assert out.shape == (2, 3, 8, 8)

    x3 = random.uniform((2, 3, 10, 10, 10), dtype="float32")
    out = layers.ZeroPadding3D(padding=1, data_format="channels_first")(x3)
    assert out.shape == (2, 3, 12, 12, 12)
    out = layers.Cropping3D(cropping=1, data_format="channels_first")(x3)
    assert out.shape == (2, 3, 8, 8, 8)


def test_padding_cropping_tuple_of_tuples() -> None:
    """Test ``padding_cropping_tuple_of_tuples`` behavior and edge cases in ``test_missing_layers_conv``."""
    layers.ZeroPadding1D(padding=((1, 1),))
    layers.ZeroPadding2D(padding=((1, 1), (2, 2)))
    layers.ZeroPadding3D(padding=((1, 1), (2, 2), (3, 3)))
    layers.Cropping1D(cropping=((1, 1),))
    layers.Cropping2D(cropping=((1, 1), (2, 2)))
    layers.Cropping3D(cropping=((1, 1), (2, 2), (3, 3)))

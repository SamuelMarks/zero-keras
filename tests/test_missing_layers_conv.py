"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import numpy as np

from zero_keras import layers


def test_conv1d_basic() -> TensorLike:
    """Function docstring."""
    layer = layers.Conv1D(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 8, 4)

    # channels_first
    layer_cf = layers.Conv1D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)
    assert out_cf.shape == (2, 4, 8)


def test_conv1d_transpose_basic() -> TensorLike:
    """Function docstring."""
    layer = layers.Conv1DTranspose(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)

    layer_cf = layers.Conv1DTranspose(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv2d_basic() -> TensorLike:
    """Function docstring."""
    layer = layers.Conv2D(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 8, 8, 4)

    layer_cf = layers.Conv2D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv2d_transpose_basic() -> TensorLike:
    """Function docstring."""
    layer = layers.Conv2DTranspose(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)

    layer_cf = layers.Conv2DTranspose(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv3d_basic() -> TensorLike:
    """Function docstring."""
    layer = layers.Conv3D(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 8, 8, 8, 4)

    layer_cf = layers.Conv3D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv3d_transpose_basic() -> TensorLike:
    """Function docstring."""
    layer = layers.Conv3DTranspose(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)

    layer_cf = layers.Conv3DTranspose(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_cropping1d() -> TensorLike:
    """Function docstring."""
    layer = layers.Cropping1D(cropping=1)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)
    layer_cf = layers.Cropping1D(cropping=1, data_format="channels_first")
    out_cf = layer_cf(np.random.uniform(size=(2, 3, 10)).astype(np.float32))


def test_cropping2d() -> TensorLike:
    """Function docstring."""
    layer = layers.Cropping2D(cropping=1)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    layer_cf = layers.Cropping2D(cropping=1, data_format="channels_first")
    out_cf = layer_cf(np.random.uniform(size=(2, 3, 10, 10)).astype(np.float32))


def test_cropping3d() -> TensorLike:
    """Function docstring."""
    layer = layers.Cropping3D(cropping=1)
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    layer_cf = layers.Cropping3D(cropping=1, data_format="channels_first")
    out_cf = layer_cf(np.random.uniform(size=(2, 3, 10, 10, 10)).astype(np.float32))


def test_depthwise1d() -> TensorLike:
    """Function docstring."""
    layer = layers.DepthwiseConv1D(kernel_size=3)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)


def test_depthwise2d() -> TensorLike:
    """Function docstring."""
    layer = layers.DepthwiseConv2D(kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)


def test_einsum_dense() -> TensorLike:
    """Function docstring."""
    layer = layers.EinsumDense(equation="abc,cd->abd", output_shape=(10, 4))
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)


def test_alpha_dropout() -> TensorLike:
    """Function docstring."""
    layer = layers.AlphaDropout(0.5)
    layer(np.random.uniform(size=(2, 10)).astype(np.float32), training=True)


def test_separable_conv1d() -> TensorLike:
    """Execute ``test_separable_conv1d``."""
    layer = layers.SeparableConv1D(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 8, 4)


def test_separable_conv2d() -> TensorLike:
    """Execute ``test_separable_conv2d``."""
    layer = layers.SeparableConv2D(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 8, 8, 4)


def test_upsampling1d() -> TensorLike:
    """Execute ``test_upsampling1d``."""
    layer = layers.UpSampling1D(size=2)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 20, 3)


def test_upsampling2d() -> TensorLike:
    """Execute ``test_upsampling2d``."""
    layer = layers.UpSampling2D(size=(2, 2))
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 20, 20, 3)


def test_upsampling3d() -> TensorLike:
    """Execute ``test_upsampling3d``."""
    layer = layers.UpSampling3D(size=(2, 2, 2))
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 20, 20, 20, 3)


def test_zeropadding1d() -> TensorLike:
    """Execute ``test_zeropadding1d``."""
    layer = layers.ZeroPadding1D(padding=1)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 12, 3)


def test_zeropadding2d() -> TensorLike:
    """Execute ``test_zeropadding2d``."""
    layer = layers.ZeroPadding2D(padding=(1, 1))
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 12, 12, 3)


def test_zeropadding3d() -> TensorLike:
    """Execute ``test_zeropadding3d``."""
    layer = layers.ZeroPadding3D(padding=(1, 1, 1))
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 12, 12, 12, 3)


def test_conv_built_and_call_direct() -> TensorLike:
    """Execute ``test_conv_built_and_call_direct``."""
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
        shape = (2, 5, 3)
        if "2D" in cls.__name__:
            shape = (2, 5, 5, 3)
        elif "3D" in cls.__name__:
            shape = (2, 5, 5, 5, 3)

        layer.build(shape)
        layer.build(shape)

        layer.built = False
        x = np.random.uniform(size=shape).astype(np.float32)
        layer.call(x)


def test_padding_tuple_cases() -> TensorLike:
    """Execute ``test_padding_tuple_cases``."""
    layers.ZeroPadding1D(padding=(1,))
    layers.ZeroPadding2D(padding=(1, 2))
    layers.ZeroPadding3D(padding=(1, 2, 3))
    layers.Cropping1D(cropping=(1,))
    layers.Cropping2D(cropping=(1, 2))
    layers.Cropping3D(cropping=(1, 2, 3))


def test_channels_first_padding_cropping() -> TensorLike:
    """Execute ``test_channels_first_padding_cropping``."""
    x1 = np.random.uniform(size=(2, 3, 10)).astype(np.float32)
    out = layers.ZeroPadding1D(padding=1, data_format="channels_first")(x1)
    assert out.shape == (2, 3, 12)
    out = layers.Cropping1D(cropping=1, data_format="channels_first")(x1)
    assert out.shape == (2, 3, 8)

    x2 = np.random.uniform(size=(2, 3, 10, 10)).astype(np.float32)
    out = layers.ZeroPadding2D(padding=1, data_format="channels_first")(x2)
    assert out.shape == (2, 3, 12, 12)
    out = layers.Cropping2D(cropping=1, data_format="channels_first")(x2)
    assert out.shape == (2, 3, 8, 8)

    x3 = np.random.uniform(size=(2, 3, 10, 10, 10)).astype(np.float32)
    out = layers.ZeroPadding3D(padding=1, data_format="channels_first")(x3)
    assert out.shape == (2, 3, 12, 12, 12)
    out = layers.Cropping3D(cropping=1, data_format="channels_first")(x3)
    assert out.shape == (2, 3, 8, 8, 8)


def test_padding_cropping_tuple_of_tuples() -> TensorLike:
    """Execute ``test_padding_cropping_tuple_of_tuples``."""
    layers.ZeroPadding1D(padding=((1, 1),))
    layers.ZeroPadding2D(padding=((1, 1), (2, 2)))
    layers.ZeroPadding3D(padding=((1, 1), (2, 2), (3, 3)))
    layers.Cropping1D(cropping=((1, 1),))
    layers.Cropping2D(cropping=((1, 1), (2, 2)))
    layers.Cropping3D(cropping=((1, 1), (2, 2), (3, 3)))

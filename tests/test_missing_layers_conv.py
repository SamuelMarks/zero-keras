"""Module docstring."""

import numpy as np
from zero_keras import layers


def test_conv1d_basic():
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


def test_conv1d_transpose_basic():
    """Function docstring."""
    layer = layers.Conv1DTranspose(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)

    layer_cf = layers.Conv1DTranspose(
        filters=4, kernel_size=3, data_format="channels_first"
    )
    x_cf = np.random.uniform(size=(2, 3, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv2d_basic():
    """Function docstring."""
    layer = layers.Conv2D(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 8, 8, 4)

    layer_cf = layers.Conv2D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv2d_transpose_basic():
    """Function docstring."""
    layer = layers.Conv2DTranspose(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)

    layer_cf = layers.Conv2DTranspose(
        filters=4, kernel_size=3, data_format="channels_first"
    )
    x_cf = np.random.uniform(size=(2, 3, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv3d_basic():
    """Function docstring."""
    layer = layers.Conv3D(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 8, 8, 8, 4)

    layer_cf = layers.Conv3D(filters=4, kernel_size=3, data_format="channels_first")
    x_cf = np.random.uniform(size=(2, 3, 10, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_conv3d_transpose_basic():
    """Function docstring."""
    layer = layers.Conv3DTranspose(filters=4, kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)

    layer_cf = layers.Conv3DTranspose(
        filters=4, kernel_size=3, data_format="channels_first"
    )
    x_cf = np.random.uniform(size=(2, 3, 10, 10, 10)).astype(np.float32)
    out_cf = layer_cf(x_cf)


def test_cropping1d():
    """Function docstring."""
    layer = layers.Cropping1D(cropping=1)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)
    layer_cf = layers.Cropping1D(cropping=1, data_format="channels_first")
    out_cf = layer_cf(np.random.uniform(size=(2, 3, 10)).astype(np.float32))


def test_cropping2d():
    """Function docstring."""
    layer = layers.Cropping2D(cropping=1)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    layer_cf = layers.Cropping2D(cropping=1, data_format="channels_first")
    out_cf = layer_cf(np.random.uniform(size=(2, 3, 10, 10)).astype(np.float32))


def test_cropping3d():
    """Function docstring."""
    layer = layers.Cropping3D(cropping=1)
    x = np.random.uniform(size=(2, 10, 10, 10, 3)).astype(np.float32)
    out = layer(x)
    layer_cf = layers.Cropping3D(cropping=1, data_format="channels_first")
    out_cf = layer_cf(np.random.uniform(size=(2, 3, 10, 10, 10)).astype(np.float32))


def test_depthwise1d():
    """Function docstring."""
    layer = layers.DepthwiseConv1D(kernel_size=3)
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)


def test_depthwise2d():
    """Function docstring."""
    layer = layers.DepthwiseConv2D(kernel_size=3)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    out = layer(x)


def test_einsum_dense():
    """Function docstring."""
    layer = layers.EinsumDense(equation="abc,cd->abd", output_shape=(10, 4))
    x = np.random.uniform(size=(2, 10, 3)).astype(np.float32)
    out = layer(x)


def test_alpha_dropout():
    """Function docstring."""
    layer = layers.AlphaDropout(0.5)
    layer(np.random.uniform(size=(2, 10)).astype(np.float32), training=True)

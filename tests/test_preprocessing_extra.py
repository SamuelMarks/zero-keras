from unittest import mock

import numpy as np

from zero_keras import layers


def test_discretization():
    layer = layers.Discretization(bin_boundaries=[0.0, 0.5, 1.0])
    x = np.random.rand(2, 5).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5)


def test_discretization_none():
    layer = layers.Discretization()
    x = np.random.rand(2, 5).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5)


def test_max_num_bounding_boxes():
    layer = layers.MaxNumBoundingBoxes(max_number=10)
    x = np.random.rand(2, 5, 4).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5, 4)


def test_string_lookup_memoryview():
    layer = layers.StringLookup()
    layer.adapt(["a", "b", "c"])
    x = memoryview(b"a")
    out = layer(x)
    assert out is not None


def test_string_lookup_2d_and_modes():
    layer_one_hot = layers.StringLookup(
        vocabulary=["a", "b", "c"], output_mode="one_hot"
    )
    x2d = np.array([["a", "d"], ["b", "c"]])
    x1d = np.array(["a", "d"])
    out = layer_one_hot(x1d)

    layer_multi = layers.StringLookup(
        vocabulary=["a", "b", "c"], output_mode="multi_hot"
    )
    out_multi = layer_multi(x2d)


def test_random_flip():
    layer = layers.RandomFlip()
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.random_flip.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_rotation():
    layer = layers.RandomRotation(0.2)
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.random_rotation.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_gaussian_blur():
    layer = layers.RandomGaussianBlur(0.2, 3, 1.0)
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.gaussian_blur.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_grayscale():
    layer = layers.RandomGrayscale(0.2)
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    with mock.patch("zero_keras.ops.image") as mock_image:
        from ml_switcheroo_compiler import ops

        mock_image.rgb_to_grayscale.return_value = ops.convert_to_tensor(
            np.random.rand(2, 5, 5, 1).astype(np.float32)
        )
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_saturation():
    layer = layers.RandomSaturation(0.2)
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.adjust_saturation.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_shear():
    layer = layers.RandomShear(0.2)
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.affine_transform.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None

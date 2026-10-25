"""Module `test_preprocessing_extra` for zero_keras."""

from __future__ import annotations

from unittest import mock

from zero_keras import layers, random


def test_discretization() -> None:
    """Verify or execute `test discretization`."""
    layer = layers.Discretization(bin_boundaries=[0.0, 0.5, 1.0])
    x = random.uniform((2, 5), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5)


def test_discretization_none() -> None:
    """Verify or execute `test discretization none`."""
    layer = layers.Discretization()
    x = random.uniform((2, 5), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5)


def test_max_num_bounding_boxes() -> None:
    """Test MaxNumBoundingBoxes padding, truncation, and exact-length behavior."""
    layer = layers.MaxNumBoundingBoxes(max_number=10, padding_value=-1.0)
    x = random.uniform((2, 5, 4), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 10, 4)

    layer_trunc = layers.MaxNumBoundingBoxes(max_number=3)
    out_trunc = layer_trunc(x)
    assert out_trunc.shape == (2, 3, 4)

    layer_exact = layers.MaxNumBoundingBoxes(max_number=5)
    out_exact = layer_exact(x)
    assert out_exact.shape == (2, 5, 4)


def test_string_lookup_memoryview() -> None:
    """Verify or execute `test string lookup memoryview`."""
    layer = layers.StringLookup()
    layer.adapt(["a", "b", "c"])
    x = memoryview(b"a")
    out = layer(x)
    assert out is not None


def test_string_lookup_2d_and_modes() -> None:
    """Verify or execute `test string lookup 2d and modes`."""
    layer_one_hot = layers.StringLookup(vocabulary=["a", "b", "c"], output_mode="one_hot")
    x2d = [["a", "d"], ["b", "c"]]
    x1d = ["a", "d"]
    _out = layer_one_hot(x1d)

    layer_multi = layers.StringLookup(vocabulary=["a", "b", "c"], output_mode="multi_hot")
    _out_multi = layer_multi(x2d)


def test_random_flip() -> None:
    """Verify or execute `test random flip`."""
    layer = layers.RandomFlip()
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.random_flip.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_rotation() -> None:
    """Verify or execute `test random rotation`."""
    layer = layers.RandomRotation(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.random_rotation.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_gaussian_blur() -> None:
    """Verify or execute `test gaussian blur`."""
    layer = layers.RandomGaussianBlur(0.2, 3, 1.0)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.gaussian_blur.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_grayscale() -> None:
    """Verify or execute `test random grayscale`."""
    layer = layers.RandomGrayscale(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.rgb_to_grayscale.return_value = random.uniform((2, 5, 5, 1), dtype="float32")
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_saturation() -> None:
    """Verify or execute `test random saturation`."""
    layer = layers.RandomSaturation(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.adjust_saturation.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_shear() -> None:
    """Verify or execute `test random shear`."""
    layer = layers.RandomShear(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.affine_transform.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None

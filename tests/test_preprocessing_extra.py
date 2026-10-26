"""Module `test_preprocessing_extra` for zero_keras."""

from __future__ import annotations

from unittest import mock

from zero_keras import layers, random


def test_discretization() -> None:
    """Test ``discretization`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.Discretization(bin_boundaries=[0.0, 0.5, 1.0])
    x = random.uniform((2, 5), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5)


def test_discretization_none() -> None:
    """Test ``discretization_none`` behavior and edge cases in ``test_preprocessing_extra``."""
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
    """Test ``string_lookup_memoryview`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.StringLookup()
    layer.adapt(["a", "b", "c"])
    x = memoryview(b"a")
    out = layer(x)
    assert out is not None


def test_string_lookup_2d_and_modes() -> None:
    """Test ``string_lookup_2d_and_modes`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer_one_hot = layers.StringLookup(vocabulary=["a", "b", "c"], output_mode="one_hot")
    x2d = [["a", "d"], ["b", "c"]]
    x1d = ["a", "d"]
    _out = layer_one_hot(x1d)

    layer_multi = layers.StringLookup(vocabulary=["a", "b", "c"], output_mode="multi_hot")
    _out_multi = layer_multi(x2d)


def test_random_flip() -> None:
    """Test ``random_flip`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.RandomFlip()
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.random_flip.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_rotation() -> None:
    """Test ``random_rotation`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.RandomRotation(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.random_rotation.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_gaussian_blur() -> None:
    """Test ``gaussian_blur`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.RandomGaussianBlur(0.2, 3, 1.0)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.gaussian_blur.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_grayscale() -> None:
    """Test ``random_grayscale`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.RandomGrayscale(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.rgb_to_grayscale.return_value = random.uniform((2, 5, 5, 1), dtype="float32")
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_saturation() -> None:
    """Test ``random_saturation`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.RandomSaturation(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.adjust_saturation.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None


def test_random_shear() -> None:
    """Test ``random_shear`` behavior and edge cases in ``test_preprocessing_extra``."""
    layer = layers.RandomShear(0.2)
    x = random.uniform((2, 5, 5, 3), dtype="float32")
    with mock.patch("zero_keras.ops.image") as mock_image:
        mock_image.affine_transform.return_value = x
        out = layer(x, training=True)
        assert out is not None
        out2 = layer(x, training=False)
        assert out2 is not None

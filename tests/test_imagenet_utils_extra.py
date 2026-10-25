"""Module `test_imagenet_utils_extra` for zero_keras."""

from __future__ import annotations
import pytest

from zero_keras.applications import imagenet_utils


def test_obtain_input_shape_extra() -> None:
    # require_flatten = False
    """Verify or execute `test obtain input shape extra`."""
    shape = imagenet_utils.obtain_input_shape(
        input_shape=(224, 224, 3),
        default_size=224,
        min_size=32,
        data_format="channels_last",
        require_flatten=False,
        weights="imagenet",
    )
    assert shape == (224, 224, 3)


if __name__ == "__main__":
    pytest.main(["-v", "test_imagenet_utils_extra.py"])

"""Module `test_convnext_extra` for zero_keras."""

from __future__ import annotations
import pytest

from zero_keras.applications.convnext import ConvNeXt


def test_convnext_extra_coverage() -> None:
    # layer_scale_init_value=None, include_preprocessing=False
    """Verify or execute `test convnext extra coverage`."""
    model = ConvNeXt(
        depths=[1, 1, 1, 1],
        projection_dims=[16, 32, 64, 128],
        layer_scale_init_value=None,
        include_preprocessing=False,
        include_top=False,
        weights=None,
    )
    # include_preprocessing=True, num_channels != 3
    model = ConvNeXt(
        depths=[1, 1, 1, 1],
        projection_dims=[16, 32, 64, 128],
        include_preprocessing=True,
        input_shape=(32, 32, 1),
        include_top=False,
        weights=None,
    )
    # total_depths <= 1
    model = ConvNeXt(
        depths=[1, 0, 0, 0],
        projection_dims=[16, 16, 16, 16],
        include_preprocessing=False,
        include_top=False,
        weights=None,
    )
    # include_top=True
    model = ConvNeXt(
        depths=[1, 1, 1, 1],
        projection_dims=[16, 32, 64, 128],
        include_preprocessing=False,
        include_top=True,
        classes=10,
        weights=None,
    )


if __name__ == "__main__":
    pytest.main(["-v", "test_convnext_extra.py"])

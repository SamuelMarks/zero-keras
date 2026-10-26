"""Unit and behavioral tests for coverage conv bias in zero_keras."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.layers import Conv1D, Conv2D, Conv3D


def test_layers_conv_channels_first_with_bias() -> None:
    """Test ``layers_conv_channels_first_with_bias`` behavior and edge cases in ``test_coverage_conv_bias``."""
    c1 = Conv1D(2, 2, data_format="channels_first", use_bias=True)
    c1.build((None, 3, 10))
    c1(ops.ones((1, 3, 10)))

    c2 = Conv2D(2, 2, data_format="channels_first", use_bias=True)
    c2.build((None, 3, 10, 10))
    c2(ops.ones((1, 3, 10, 10)))

    c3 = Conv3D(2, 2, data_format="channels_first", use_bias=True)
    c3.build((None, 3, 10, 10, 10))
    c3(ops.ones((1, 3, 10, 10, 10)))

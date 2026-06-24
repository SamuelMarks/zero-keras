"""Module docstring."""

from zero_keras.layers import Conv1D, Conv2D, Conv3D
import numpy as np
from ml_switcheroo_compiler.ops import asarray


def test_layers_conv_channels_first_with_bias():
    """Function docstring."""
    # 2793->2794
    c1 = Conv1D(2, 2, data_format="channels_first", use_bias=True)
    c1.build((None, 3, 10))
    c1(asarray(np.ones((1, 3, 10))))

    c2 = Conv2D(2, 2, data_format="channels_first", use_bias=True)
    c2.build((None, 3, 10, 10))
    c2(asarray(np.ones((1, 3, 10, 10))))

    c3 = Conv3D(2, 2, data_format="channels_first", use_bias=True)
    c3.build((None, 3, 10, 10, 10))
    c3(asarray(np.ones((1, 3, 10, 10, 10))))

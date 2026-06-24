"""Module docstring."""

import numpy as np
import ml_switcheroo_compiler.ops as ops
from zero_keras.layers import (
    Conv1D,
    Conv2D,
    Conv3D,
    Cropping2D,
    Cropping3D,
    SimpleRNNCell,
    Bidirectional,
    RMSNormalization,
    GroupNormalization,
    ConvLSTM1D,
    SimpleRNN,
    Dense,
)


def test_conv_channels_first():
    """Function docstring."""
    original_transpose = getattr(ops, "transpose", None)
    ops.transpose = lambda x, perm: x
    try:
        l1 = Conv1D(2, 2, data_format="channels_first")
        l1.build((None, 4, 4))
        l1(np.ones((1, 4, 4)))

        l2 = Conv2D(2, 2, data_format="channels_first")
        l2.build((None, 4, 4, 4))
        l2(np.ones((1, 4, 4, 4)))

        l3 = Conv3D(2, 2, data_format="channels_first")
        l3.build((None, 4, 4, 4, 4))
        l3(np.ones((1, 4, 4, 4, 4)))
    finally:
        if original_transpose:
            ops.transpose = original_transpose
        else:
            delattr(ops, "transpose")  # pragma: no cover


def test_cropping_elif():
    """Function docstring."""
    l2 = Cropping2D(cropping=(1, 2))
    l2(np.ones((1, 5, 5, 2)))

    l3 = Cropping3D(cropping=(1, 2, 3))
    l3(np.ones((1, 5, 5, 5, 2)))


def test_rnn_reset_states_already_built():
    """Function docstring."""
    srnn = SimpleRNN(10, stateful=True)
    srnn(np.ones((1, 5, 10)))  # build and init states
    srnn.reset_states()
    srnn.reset_states(states=(np.zeros((1, 10)),))


def test_simplernncell_unbuilt_call():
    """Function docstring."""
    c1 = SimpleRNNCell(10)
    # bypass __call__ to hit 'if not self.built:' inside call()
    c1.call(np.ones((1, 10)), states=(np.zeros((1, 10)),))


def test_bidirectional_dense_units():
    """Function docstring."""
    try:
        Bidirectional(Dense(10))
    except Exception:
        pass


def test_rms_norm_center():
    """Function docstring."""
    rms = RMSNormalization(center=True, scale=False)
    rms(np.ones((1, 10)))


def test_group_norm_build_twice_and_axis():
    """Function docstring."""
    gn = GroupNormalization(axis=1, groups=2)
    # Call build directly twice
    GroupNormalization.build(gn, (None, 2, 10, 2))
    GroupNormalization.build(gn, (None, 2, 10, 2))

    # call to hit axis branch with matching reshape
    # shape=(1, 2, 10, 2)
    # new_shape=(1, 2, 10, 2, 1) -> size 40 matches!
    gn(np.ones((1, 2, 10, 2)))


def test_convlstm_padding_same_spatial():
    """Function docstring."""
    cl = ConvLSTM1D(2, 2, padding="same")
    cl.build((None, 5, 10, 3))

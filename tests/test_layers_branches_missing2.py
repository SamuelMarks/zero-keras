"""Module docstring."""

import numpy as np
import zero_keras.layers as layers


def test_preset_weights_convs():
    """Function docstring."""
    # DepthwiseConv1D
    l = layers.DepthwiseConv1D(2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 2))

    # DepthwiseConv2D
    l = layers.DepthwiseConv2D(2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 2))

    # Conv1DTranspose
    l = layers.Conv1DTranspose(2, 3)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 2))

    # Conv2DTranspose
    l = layers.Conv2DTranspose(2, 3)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 2))

    # Conv3DTranspose
    l = layers.Conv3DTranspose(2, 3)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 5, 2))


def test_preset_weights_rnn_cells():
    """Function docstring."""
    # SimpleRNNCell
    c = layers.SimpleRNNCell(4)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))

    # GRUCell with reset_after=True
    c = layers.GRUCell(4, reset_after=True)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))

    # GRUCell with reset_after=False
    c = layers.GRUCell(4, reset_after=False)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))

    # LSTMCell
    c = layers.LSTMCell(4)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))


def test_simplernn_activation_none():
    """Function docstring."""
    l = layers.SimpleRNN(4, activation=None)
    l(np.random.rand(2, 3, 2))


def test_convlstm_bias_none():
    """Function docstring."""
    l = layers.ConvLSTM1D(2, 3, use_bias=False)
    l(np.random.rand(2, 3, 5, 2))


def test_wrapper_already_built():
    """Function docstring."""
    # TimeDistributed already built wrapper
    l = layers.Dense(4)
    t = layers.TimeDistributed(l)
    t.build((None, 3, 2))
    t.build((None, 3, 2))  # trigger if self.built: return

    # SpectralNormalization already built wrapper
    l = layers.Dense(4)
    s = layers.SpectralNormalization(l)
    s.build((None, 2))
    s.build((None, 2))  # trigger if self.built: return


def test_bidirectional_args():
    """Function docstring."""

    class DummyCell:
        """Class docstring."""

        def __init__(self, units):
            """Function docstring.

            Args:
                units: Description.
            """
            self.units = units

    class DummyLayer(layers.Layer):
        """Class docstring."""

        def __init__(self, cell):
            """Function docstring.

            Args:
                cell: Description.
            """
            super().__init__()
            self.cell = cell
            self.return_sequences = False
            self.return_state = False
            self.go_backwards = False
            self.stateful = False

    l = DummyLayer(DummyCell(4))
    b = layers.Bidirectional(l)

"""Tests for missing coverage in layers."""

from zero_keras.layers import (
    Dense,
    Conv1D,
    Conv2D,
    Conv3D,
    Conv1DTranspose,
    Conv2DTranspose,
    Conv3DTranspose,
    SimpleRNNCell,
    GRUCell,
    LSTMCell,
    SpectralNormalization,
    Bidirectional,
    RNN,
    Layer,
)
import numpy as np


def test_activation_none_branches():
    """Test branches where activation is None."""
    # Dense
    layer = Dense(2)
    layer.activation = None
    layer(np.ones((2, 2)))

    # Conv1D
    layer = Conv1D(2, 2)
    layer.activation = None
    layer(np.ones((2, 4, 2)))

    # Conv2D
    layer = Conv2D(2, 2)
    layer.activation = None
    layer(np.ones((2, 4, 4, 2)))

    # Conv3D
    layer = Conv3D(2, 2)
    layer.activation = None
    layer(np.ones((2, 4, 4, 4, 2)))

    # Conv1DTranspose
    layer = Conv1DTranspose(2, 2)
    layer.activation = None
    layer(np.ones((2, 4, 2)))

    # Conv2DTranspose
    layer = Conv2DTranspose(2, 2)
    layer.activation = None
    layer(np.ones((2, 4, 4, 2)))

    # Conv3DTranspose
    layer = Conv3DTranspose(2, 2)
    layer.activation = None
    layer(np.ones((2, 4, 4, 4, 2)))

    # SimpleRNNCell
    layer = SimpleRNNCell(2)
    layer.activation = None
    layer(np.ones((2, 2)), [np.ones((2, 2))])


def test_rnn_cell_no_bias_branch():
    """Test RNN cells where use_bias is False or getattr returns None."""
    # SimpleRNNCell 5539-5540
    # True branch (getattr is None)
    layer = SimpleRNNCell(2, use_bias=False)
    layer.build((None, 2))
    # False branch
    layer = SimpleRNNCell(2, use_bias=False)
    layer.bias = "dummy"
    layer.build((None, 2))

    # GRUCell 5810
    layer = GRUCell(2, use_bias=False)
    layer.bias = "dummy"
    layer.build((None, 2))

    # LSTMCell 6131, 6142
    # To cover bias_init with unit_forget_bias=False, we need use_bias=True (default) and getattr=None (default)
    layer = LSTMCell(2, unit_forget_bias=False)
    layer.build((None, 2))

    # False branch for getattr(self, "bias") is None
    layer = LSTMCell(2, use_bias=False)
    layer.bias = "dummy"
    layer.build((None, 2))


def test_bidirectional_rnn_cell_units():
    """Test Bidirectional wrapper missing branches."""
    # 6405
    layer = Bidirectional(RNN(SimpleRNNCell(3)))
    assert layer.backward_layer.cell == 3

    class DummyBidirectionalLayer(Layer):
        """Dummy layer for testing Bidirectional branches."""

        def call(self, inputs):
            """Call function."""
            return inputs

    layer2 = DummyBidirectionalLayer()
    layer2.args = ()
    layer2.return_sequences = False
    layer2.return_state = False
    Bidirectional(layer2)


def test_spectral_norm_no_weights_branch():
    """Test SpectralNormalization where u is not in _weights."""

    # 12831->12835
    class DummyWrapper(SpectralNormalization):
        """Dummy wrapper to skip adding weight to _weights."""

        def add_weight(self, *args, **kwargs):
            """Override add_weight."""
            weight = super().add_weight(*args, **kwargs)
            if weight in self._weights:
                self._weights.remove(weight)
            return weight

    layer = Dense(2)
    sn = DummyWrapper(layer)
    sn.build((None, 3))

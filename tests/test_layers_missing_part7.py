import numpy as np
import ml_switcheroo_compiler.ops as ops
from zero_keras.layers import (
    Dot,
    Conv1DTranspose,
    Conv2DTranspose,
    Conv3DTranspose,
    Cropping1D,
    Cropping2D,
    Cropping3D,
    EinsumDense,
    RNN,
    SimpleRNNCell,
    LSTM,
    GRUCell,
    LSTMCell,
    Bidirectional,
    Normalization,
    GroupNormalization,
    RandomFlip,
    RandomRotation,
    RandAugment,
    LayerNormalization,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
    ConvLSTM1D,
    Layer,
    SimpleRNN,
)


def test_dot_tuple_axes():
    layer = Dot(axes=(1, 1))
    layer([np.ones((2, 3)), np.ones((2, 3))])


def test_conv_transpose_channels_first():
    l1 = Conv1DTranspose(2, 2, data_format="channels_first")
    l1(np.ones((1, 2, 4)))

    l2 = Conv2DTranspose(2, 2, data_format="channels_first")
    l2(np.ones((1, 2, 4, 4)))

    l3 = Conv3DTranspose(2, 2, data_format="channels_first")
    l3(np.ones((1, 2, 4, 4, 4)))


def test_cropping_tuples():
    l1 = Cropping1D(cropping=((1, 2),))
    l1(np.ones((1, 5, 2)))

    l2 = Cropping2D(cropping=((1, 2), (3, 4)))
    l2(np.ones((1, 5, 5, 2)))

    l3 = Cropping3D(cropping=((1, 2), (3, 4), (5, 6)))
    l3(np.ones((1, 5, 5, 5, 2)))


def test_einsum_dense_bias():
    layer = EinsumDense("abc,cd->abd", output_shape=(None, 64), bias_axes="d")
    layer.build((None, 32, 128))
    # just build to cover the bias creation


class DummyCell(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, inputs, states, training=None, **kwargs):
        return inputs, states


def test_rnn_missing_coverage():
    rnn = RNN(DummyCell(10))
    rnn(np.ones((1, 5, 10)))
    rnn(np.ones((1, 5, 10)), initial_state=np.zeros((1, 10)))

    # stateful reset_states
    srnn = SimpleRNN(10, stateful=True)
    srnn.build((1, 5, 10))
    srnn.reset_states()
    srnn.reset_states(states=(np.zeros((1, 10)),))


def test_lstm_return_state():
    lstm = LSTM(10, return_state=True)
    lstm(np.ones((1, 5, 10)))


def test_cells_build():
    c1 = SimpleRNNCell(10)
    c1(np.ones((1, 10)), states=(np.zeros((1, 10)),))

    c2 = GRUCell(10)
    c2.build((None, 10))
    c2.build((None, 10))

    c3 = LSTMCell(10)
    c3.build((None, 10))
    c3.build((None, 10))


def test_bidirectional_missing():
    # Mock ops.reverse since ml_switcheroo_compiler might not have it
    original_reverse = getattr(ops, "reverse", None)
    ops.reverse = lambda x, dims: x
    try:
        bidi = Bidirectional(LSTM(10, return_sequences=True), merge_mode="sum")
        bidi(
            np.ones((1, 5, 10)),
            initial_state=[
                np.zeros((1, 10)),
                np.zeros((1, 10)),
                np.zeros((1, 10)),
                np.zeros((1, 10)),
            ],
        )
    finally:
        if original_reverse:
            ops.reverse = original_reverse
        else:
            delattr(ops, "reverse")

    bidi2 = Bidirectional(LSTM(10), backward_layer=LSTM(10, go_backwards=True))
    bidi2(np.ones((1, 5, 10)))


def test_normalization():
    norm = Normalization(invert=True)
    norm(np.ones((1, 10)))


def test_group_normalization():
    gn1 = GroupNormalization(groups=2, scale=False, center=True)
    gn1(np.ones((1, 10, 10, 4)))
    gn2 = GroupNormalization(groups=2, scale=False, center=False)
    gn2(np.ones((1, 10, 10, 4)))


def test_random_layers():
    l1 = RandomFlip("horizontal")
    l1(np.ones((1, 10, 10, 3)))
    l2 = RandomRotation(0.2)
    l2(np.ones((1, 10, 10, 3)))
    l4 = RandAugment(2, 9)
    l4(np.ones((1, 10, 10, 3)))


def test_layer_normalization():
    ln = LayerNormalization(axis=1)
    ln.build((None, 5, 10))
    ln.build((None, 5, 10))
    ln(np.ones((1, 5, 10)))


def test_zeropadding():
    z1 = ZeroPadding1D(data_format="channels_first")
    z1(np.ones((1, 2, 4)))

    z2 = ZeroPadding2D(padding=(1, 2))
    z2(np.ones((1, 4, 4, 2)))
    z3 = ZeroPadding2D(data_format="channels_first")
    z3(np.ones((1, 2, 4, 4)))

    z4 = ZeroPadding3D(padding=(1, 2, 3))
    z4(np.ones((1, 4, 4, 4, 2)))
    z5 = ZeroPadding3D(data_format="channels_first")
    z5(np.ones((1, 2, 4, 4, 4)))


def test_convlstm():
    cl = ConvLSTM1D(2, 2, padding="same")
    cl.build((None, None, None, 3))

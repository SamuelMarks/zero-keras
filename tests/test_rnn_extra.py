import numpy as np

from zero_keras import layers


def test_simplernn_isolated():
    cell = layers.SimpleRNNCell(4)
    layer = layers.SimpleRNN(4)
    x = np.random.rand(2, 5, 3).astype(np.float32)
    assert layer(x).shape == (2, 4)

    cell.build((None, 3))
    out, _states = cell(x[:, 0, :], [np.zeros((2, 4), dtype=np.float32)])
    assert out.shape == (2, 4)


def test_gru_isolated():
    cell = layers.GRUCell(4)
    layer = layers.GRU(4)
    x = np.random.rand(2, 5, 3).astype(np.float32)
    assert layer(x).shape == (2, 4)


def test_lstm_isolated():
    cell = layers.LSTMCell(4)
    layer = layers.LSTM(4)
    x = np.random.rand(2, 5, 3).astype(np.float32)
    assert layer(x).shape == (2, 4)


def test_stacked_rnn_cells():
    cells = [layers.LSTMCell(4), layers.LSTMCell(4)]
    layer = layers.RNN(layers.StackedRNNCells(cells))
    x = np.random.rand(2, 5, 3).astype(np.float32)
    assert layer(x).shape == (2, 4)

    stacked = layers.StackedRNNCells(cells)
    assert len(stacked.state_size) == 4


def test_rnn_stateful():
    layer = layers.SimpleRNN(4, stateful=True)
    layer.build((2, 5, 3))
    x = np.random.rand(2, 5, 3).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 4)
    out2 = layer(x)
    assert out2.shape == (2, 4)


from unittest.mock import patch


@patch("zero_keras.ops.gru_cell", return_value=(None, [None, None]))
@patch("zero_keras.ops.lstm_cell", return_value=(None, [None, None]))
@patch("zero_keras.ops.rnn")
def test_all_rnn_and_attention_coverage(mock_rnn, mock_lstm, mock_gru):
    import numpy as np

    from zero_keras import layers

    mock_rnn.return_value = (np.zeros((2, 5, 4)), [np.zeros((2, 4)), np.zeros((2, 4))])

    # RNN Cells
    x = np.random.uniform(size=(2, 3)).astype(np.float32)

    cell1 = layers.SimpleRNNCell(4)
    cell1.build((2, 3))
    cell1.call(x, [np.zeros((2, 4))])

    cell2 = layers.GRUCell(4)
    cell2.build((2, 3))
    cell2.call(x, [np.zeros((2, 4))])

    cell3 = layers.LSTMCell(4)
    cell3.build((2, 3))
    cell3.call(x, [np.zeros((2, 4)), np.zeros((2, 4))])

    # RNN Layers
    x_seq = np.random.uniform(size=(2, 5, 3)).astype(np.float32)

    for cls in [layers.SimpleRNN, layers.GRU, layers.LSTM]:
        l = cls(4, return_sequences=True, return_state=True)
        l(x_seq)

    # Attention
    l = layers.Attention()
    l([x_seq, x_seq])

    l = layers.AdditiveAttention()
    l([x_seq, x_seq])

    l = layers.MultiHeadAttention(num_heads=2, key_dim=2)
    l(x_seq, x_seq)

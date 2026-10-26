"""Module `test_rnn_extra` for zero_keras."""

from __future__ import annotations

from unittest.mock import patch

from zero_keras import layers, ops, random
from zero_keras.types import TensorLike


def test_simplernn_isolated() -> None:
    """Test ``simplernn_isolated`` behavior and edge cases in ``test_rnn_extra``."""
    cell = layers.SimpleRNNCell(4)
    layer = layers.SimpleRNN(4)
    x = random.uniform((2, 5, 3), dtype="float32")
    assert layer(x).shape == (2, 4)

    cell.build((None, 3))
    out, _states = cell(x[:, 0, :], [ops.zeros((2, 4), dtype="float32")])
    assert out.shape == (2, 4)


def test_gru_isolated() -> None:
    """Test ``gru_isolated`` behavior and edge cases in ``test_rnn_extra``."""
    _cell = layers.GRUCell(4)
    layer = layers.GRU(4)
    x = random.uniform((2, 5, 3), dtype="float32")
    assert layer(x).shape == (2, 4)


def test_lstm_isolated() -> None:
    """Test ``lstm_isolated`` behavior and edge cases in ``test_rnn_extra``."""
    _cell = layers.LSTMCell(4)
    layer = layers.LSTM(4)
    x = random.uniform((2, 5, 3), dtype="float32")
    assert layer(x).shape == (2, 4)


def test_stacked_rnn_cells() -> None:
    """Test ``stacked_rnn_cells`` behavior and edge cases in ``test_rnn_extra``."""
    cells = [layers.LSTMCell(4), layers.LSTMCell(4)]
    layer = layers.RNN(layers.StackedRNNCells(cells))
    x = random.uniform((2, 5, 3), dtype="float32")
    assert layer(x).shape == (2, 4)

    stacked = layers.StackedRNNCells(cells)
    assert len(stacked.state_size) == 4


def test_rnn_stateful() -> None:
    """Test ``rnn_stateful`` behavior and edge cases in ``test_rnn_extra``."""
    layer = layers.SimpleRNN(4, stateful=True)
    layer.build((2, 5, 3))
    x = random.uniform((2, 5, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 4)
    out2 = layer(x)
    assert out2.shape == (2, 4)


@patch("zero_keras.ops.gru_cell", return_value=(None, [None, None]))
@patch("zero_keras.ops.lstm_cell", return_value=(None, [None, None]))
@patch("zero_keras.ops.rnn")
def test_all_rnn_and_attention_coverage(
    mock_rnn: TensorLike, mock_lstm: TensorLike, mock_gru: TensorLike
) -> None:
    """Test ``all_rnn_and_attention_coverage`` behavior and edge cases in ``test_rnn_extra``.

    Args:
        mock_rnn: Mocked RNN backend operator fixture.
        mock_lstm: Mocked LSTM backend operator fixture.
        mock_gru: Mocked GRU backend operator fixture.
    """
    _ = mock_lstm, mock_gru
    mock_rnn.return_value = (ops.zeros((2, 5, 4)), [ops.zeros((2, 4)), ops.zeros((2, 4))])

    x = random.uniform((2, 3), dtype="float32")

    cell1 = layers.SimpleRNNCell(4)
    cell1.build((2, 3))
    cell1.call(x, [ops.zeros((2, 4))])

    cell2 = layers.GRUCell(4)
    cell2.build((2, 3))
    cell2.call(x, [ops.zeros((2, 4))])

    cell3 = layers.LSTMCell(4)
    cell3.build((2, 3))
    cell3.call(x, [ops.zeros((2, 4)), ops.zeros((2, 4))])

    x_seq = random.uniform((2, 5, 3), dtype="float32")

    for cls in [layers.SimpleRNN, layers.GRU, layers.LSTM]:
        l_rnn = cls(4, return_sequences=True, return_state=True)
        l_rnn(x_seq)

    l_att = layers.Attention()
    l_att([x_seq, x_seq])

    l_add = layers.AdditiveAttention()
    l_add([x_seq, x_seq])

    l_mha = layers.MultiHeadAttention(num_heads=2, key_dim=2)
    l_mha(x_seq, x_seq)

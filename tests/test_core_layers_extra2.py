from zero_keras.core_layers import KerasTensor
from unittest.mock import patch


@patch("ml_switcheroo_compiler.ops.ones")
def test_keras_tensor_eq_no_data2(mock_ones):
    mock_ones.return_value = "mock_ones"
    tensor = KerasTensor(shape=None)
    other = KerasTensor(shape=None)
    result = tensor == other
    assert result == "mock_ones"

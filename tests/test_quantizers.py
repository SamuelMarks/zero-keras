import numpy as np

from zero_keras import ops, quantizers


def test_quantizers():
    """Test quantizers module."""
    # Test quantize_and_dequantize
    inputs = ops.convert_to_tensor([1.0, 2.0, 3.0])
    scale = ops.convert_to_tensor(1.0)
    dtype_max = ops.convert_to_tensor(127.0)
    res = quantizers.quantize_and_dequantize(inputs, scale, dtype_max)
    assert np.allclose(ops.convert_to_numpy(res), [1.0, 2.0, 3.0])

    # Test abs_max_quantize
    q, s = quantizers.abs_max_quantize(inputs, axis=0)
    assert q.shape == (3,)
    assert s.shape == (1,)

    # Test compute_float8_amax_history
    history = ops.convert_to_tensor([0.0, 0.0, 0.0])
    new_history = quantizers.compute_float8_amax_history(inputs, history)
    assert np.allclose(ops.convert_to_numpy(new_history), [3.0, 0.0, 0.0])

    # Test compute_float8_scale
    new_scale = quantizers.compute_float8_scale(new_history, scale, dtype_max)
    assert float(ops.convert_to_numpy(new_scale)) == 127.0 / 3.0

    # Test fake_quant_with_min_max_vars
    fq = quantizers.fake_quant_with_min_max_vars(inputs, 1.5, 2.5)
    assert np.allclose(ops.convert_to_numpy(fq), [1.5, 2.0, 2.5])

    # Test AbsMaxQuantizer
    q_layer = quantizers.AbsMaxQuantizer(axis=0)
    res_q = q_layer(inputs)
    assert res_q.shape == (3,)
    config = q_layer.get_config()
    assert config["axis"] == 0
    q_layer_2 = quantizers.AbsMaxQuantizer.from_config(config)
    assert q_layer_2.axis == 0

    # Test serialization/deserialization
    assert quantizers.serialize(q_layer) == q_layer
    assert quantizers.deserialize(config) == config
    assert quantizers.get("test") == "test"

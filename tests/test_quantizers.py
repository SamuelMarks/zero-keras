"""Module `test_quantizers` for zero_keras."""

from __future__ import annotations

from zero_keras import ops, quantizers

from .conftest import assert_allclose_zero


def test_quantizers() -> None:
    """Test quantizers module."""
    inputs = ops.convert_to_tensor([1.0, 2.0, 3.0])
    scale = ops.convert_to_tensor(1.0)
    dtype_max = ops.convert_to_tensor(127.0)
    res = quantizers.quantize_and_dequantize(inputs, scale, dtype_max)
    assert_allclose_zero(ops.convert_to_numpy(res), [1.0, 2.0, 3.0])

    q, s = quantizers.abs_max_quantize(inputs, axis=0)
    assert q.shape == (3,)
    assert s.shape == (1,)

    history = ops.convert_to_tensor([0.0, 0.0, 0.0])
    new_history = quantizers.compute_float8_amax_history(inputs, history)
    assert_allclose_zero(ops.convert_to_numpy(new_history), [3.0, 0.0, 0.0])

    new_scale = quantizers.compute_float8_scale(new_history, scale, dtype_max)
    assert float(ops.convert_to_numpy(new_scale)) == 127.0 / 3.0

    fq = quantizers.fake_quant_with_min_max_vars(inputs, 1.5, 2.5)
    assert_allclose_zero(ops.convert_to_numpy(fq), [1.5, 2.0, 2.5])

    q_layer = quantizers.AbsMaxQuantizer(axis=0)
    res_q = q_layer(inputs)
    assert res_q.shape == (3,)
    config = q_layer.get_config()
    assert config["axis"] == 0
    q_layer_2 = quantizers.AbsMaxQuantizer.from_config(config)
    assert q_layer_2.axis == 0

    assert quantizers.serialize(q_layer) == q_layer
    assert quantizers.deserialize(config) == config
    assert quantizers.get("test") == "test"

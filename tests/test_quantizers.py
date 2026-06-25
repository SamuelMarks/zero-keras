from zero_keras import quantizers


def test_quantizers():
    """Test quantizers module."""
    quantizers.AbsMaxQuantizer()
    quantizers.Quantizer()
    assert quantizers.abs_max_quantize(1, axis=0) == 1
    assert quantizers.compute_float8_amax_history(1, 1) == 1
    assert quantizers.compute_float8_scale(1, 1, 1) == 1
    assert quantizers.deserialize("test") == "test"
    assert quantizers.fake_quant_with_min_max_vars(1, 0, 1) == 1
    assert quantizers.get("test") == "test"
    assert quantizers.quantize_and_dequantize(1, 1, 1) == 1
    assert quantizers.serialize("test") == "test"

"""quantizers API."""

from zero_keras.core_layers import Quantizer


class AbsMaxQuantizer(Quantizer):
    """AbsMaxQuantizer docstring."""

    pass


def abs_max_quantize(x, axis):
    """abs_max_quantize docstring."""
    return x


def compute_float8_amax_history(x, amax_history):
    """compute_float8_amax_history docstring."""
    return amax_history


def compute_float8_scale(amax_history, scale, dtype_max, margin=0):
    """compute_float8_scale docstring."""
    return scale


def deserialize(config, custom_objects=None):
    """deserialize docstring."""
    return config


def fake_quant_with_min_max_vars(inputs, min, max, num_bits=8, narrow_range=False):
    """fake_quant_with_min_max_vars docstring."""
    return inputs


def get(identifier):
    """get docstring."""
    return identifier


def quantize_and_dequantize(inputs, scale, dtype_max, num_bits=8, narrow_range=False):
    """quantize_and_dequantize docstring."""
    return inputs


def serialize(quantizer):
    """serialize docstring."""
    return quantizer


__all__ = [
    "AbsMaxQuantizer",
    "Quantizer",
    "abs_max_quantize",
    "compute_float8_amax_history",
    "compute_float8_scale",
    "deserialize",
    "fake_quant_with_min_max_vars",
    "get",
    "quantize_and_dequantize",
    "serialize",
]

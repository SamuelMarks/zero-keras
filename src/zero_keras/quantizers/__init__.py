"""quantizers API."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from zero_keras import ops
from zero_keras.core_layers import Quantizer


class AbsMaxQuantizer(Quantizer):
    """AbsMaxQuantizer docstring."""

    def __init__(self, axis: int = -1, **kwargs: Any) -> None:
        """docstring."""
        super().__init__()
        self.axis = axis
        self._name = kwargs.get("name", "AbsMaxQuantizer")

    def __call__(self, x: Any) -> Any:
        """Call docstring."""
        return abs_max_quantize(x, self.axis)[0]

    def get_config(self) -> dict[str, Any]:
        """get_config docstring."""
        config = super().get_config() if hasattr(super(), "get_config") else {}
        config.update({"axis": self.axis, "name": self._name})
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AbsMaxQuantizer:
        """from_config docstring."""
        return cls(**config)


def abs_max_quantize(x: Any, axis: int) -> tuple[Any, Any]:
    """abs_max_quantize docstring.

    Args:
        x: Input tensor.
        axis: Axis.
    """
    scale = ops.max(ops.abs(x), axis=axis, keepdims=True)
    # Avoid division by zero
    scale = ops.maximum(scale, ops.cast(1e-7, scale.dtype))
    # Max value for int8 is 127
    quantized_x = ops.round((x / scale) * ops.cast(127, x.dtype))
    quantized_x = ops.cast(quantized_x, "int8")
    return quantized_x, scale


def compute_float8_amax_history(x: Any, amax_history: Any) -> Any:
    """compute_float8_amax_history docstring.

    Args:
        x: Input tensor.
        amax_history: History tensor.
    """
    amax = ops.max(ops.abs(x))
    amax = ops.cast(amax, amax_history.dtype)
    amax_expanded = ops.expand_dims(amax, axis=0)
    # shift history
    new_history = ops.concatenate([amax_expanded, amax_history[:-1]], axis=0)
    return new_history


def compute_float8_scale(
    amax_history: Any, scale: Any, dtype_max: Any, margin: int = 0
) -> Any:
    """compute_float8_scale docstring.

    Args:
        amax_history: History tensor.
        scale: Current scale.
        dtype_max: Max value for dtype.
        margin: Margin.
    """
    amax = ops.max(amax_history)
    dtype_max = ops.cast(dtype_max, amax.dtype)
    new_scale = dtype_max / ops.maximum(amax, ops.cast(1e-7, amax.dtype))
    return new_scale


def deserialize(config: Any, custom_objects: Any | None = None) -> Any:
    """deserialize docstring."""
    return config


def fake_quant_with_min_max_vars(
    inputs: Any, min: Any, max: Any, num_bits: int = 8, narrow_range: bool = False
) -> Any:
    """fake_quant_with_min_max_vars docstring."""
    # fake quant is mostly clipping in this mock
    return ops.clip(inputs, min, max)


def get(identifier: Any) -> Any:
    """get docstring."""
    return identifier


def quantize_and_dequantize(
    inputs: Any,
    scale: Any,
    dtype_max: Any,
    num_bits: int = 8,
    narrow_range: bool = False,
) -> Any:
    """quantize_and_dequantize docstring."""
    # QAT mock: quantize to nearest, then dequantize
    scaled = inputs / ops.maximum(scale, ops.cast(1e-7, scale.dtype))
    q = ops.round(scaled)
    dtype_max = ops.cast(dtype_max, q.dtype)
    q = ops.clip(q, -dtype_max, dtype_max)
    return q * scale


def serialize(quantizer: Any) -> Any:
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

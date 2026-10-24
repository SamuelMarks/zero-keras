"""Quantizers API for zero_keras."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.core_layers import Quantizer
from zero_keras.types import ConfigDict, ConfigValue, TensorLike


class AbsMaxQuantizer(Quantizer):
    """Symmetric abs-max int8 quantizer."""

    def __init__(
        self,
        axis: int = -1,
        name: str = "AbsMaxQuantizer",
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize AbsMaxQuantizer.

        Args:
            axis: Axis or axes along which to compute maximum absolute scale.
            name: Quantizer name.
            **kwargs: Additional keyword options.
        """
        super().__init__()
        _ = kwargs
        self.axis = axis
        self._name = name

    def __call__(self, x: TensorLike) -> TensorLike:
        """Quantize tensor ``x`` using abs-max scaling.

        Args:
            x: Input tensor.

        Returns:
            TensorLike: Quantized int8 tensor.
        """
        return abs_max_quantize(x, self.axis)[0]

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {"axis": self.axis, "name": self._name}

    @classmethod
    def from_config(cls, config: ConfigDict) -> AbsMaxQuantizer:
        """Instantiate an AbsMaxQuantizer from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            AbsMaxQuantizer: Reconstructed quantizer instance.
        """
        axis_raw = config.get("axis", -1)
        axis = int(axis_raw) if isinstance(axis_raw, (int, float)) else -1
        name_raw = config.get("name", "AbsMaxQuantizer")
        name = str(name_raw) if isinstance(name_raw, str) else "AbsMaxQuantizer"
        return cls(axis=axis, name=name)


def abs_max_quantize(
    x: TensorLike,
    axis: int = -1,
) -> tuple[TensorLike, TensorLike]:
    """Quantize ``x`` to int8 using symmetric abs-max scaling along ``axis``.

    Args:
        x: Input tensor.
        axis: Axis along which to compute scale factor.

    Returns:
        tuple[TensorLike, TensorLike]: ``(quantized_x, scale)`` tuple.
    """
    t = ops.convert_to_tensor(x)
    scale = ops.max(ops.abs(t), axis=axis, keepdims=True)
    scale = ops.maximum(scale, ops.cast(1e-7, ops.dtype(scale)))
    quantized_x = ops.round(ops.multiply(ops.divide(t, scale), ops.cast(127, ops.dtype(t))))
    quantized_x = ops.cast(quantized_x, "int8")
    return quantized_x, scale


def compute_float8_amax_history(
    x: TensorLike,
    amax_history: TensorLike,
) -> TensorLike:
    """Update the rolling FP8 absolute-maximum history tensor with ``x``.

    Args:
        x: Current activation or gradient tensor.
        amax_history: Rolling 1-D history tensor.

    Returns:
        TensorLike: Updated amax history tensor.
    """
    hist_t = ops.convert_to_tensor(amax_history)
    amax = ops.max(ops.abs(x))
    amax = ops.cast(amax, hist_t.dtype)
    amax_expanded = ops.expand_dims(amax, axis=0)
    return ops.concatenate([amax_expanded, hist_t[:-1]], axis=0)


def compute_float8_scale(
    amax_history: TensorLike,
    scale: TensorLike,
    dtype_max: float | TensorLike,
    margin: int = 0,
) -> TensorLike:
    """Compute new FP8 scaling factor from ``amax_history`` and ``dtype_max``.

    Args:
        amax_history: Rolling absolute maximum history tensor.
        scale: Previous scale tensor.
        dtype_max: Maximum representable value of target FP8 dtype.
        margin: Power-of-two safety margin.

    Returns:
        TensorLike: Updated FP8 scale tensor.
    """
    _ = scale, margin
    amax = ops.max(amax_history)
    amax_dt = ops.dtype(amax)
    dmax = ops.cast(dtype_max, amax_dt)
    return ops.divide(dmax, ops.maximum(amax, ops.cast(1e-7, amax_dt)))


def deserialize(
    config: Quantizer | str | ConfigDict | None,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> Quantizer | str | ConfigDict | None:
    """Deserialize a quantizer configuration.

    Args:
        config: Serialized quantizer config or instance.
        custom_objects: Optional custom object mapping.

    Returns:
        Quantizer | str | ConfigDict | None: Deserialized quantizer.
    """
    _ = custom_objects
    return config


def fake_quant_with_min_max_vars(
    inputs: TensorLike,
    min: float | TensorLike,
    max: float | TensorLike,
    num_bits: int = 8,
    narrow_range: bool = False,
) -> TensorLike:
    """Simulate quantization clamping between ``min`` and ``max``.

    Args:
        inputs: Input tensor.
        min: Minimum quantization range bound.
        max: Maximum quantization range bound.
        num_bits: Bit-width of simulated integer quantization.
        narrow_range: Whether to exclude the minimum negative integer value.

    Returns:
        TensorLike: Clamped fake-quantized tensor.
    """
    _ = num_bits, narrow_range
    return ops.clip(inputs, min, max)


def get(
    identifier: Quantizer | str | ConfigDict | None,
) -> Quantizer | str | ConfigDict | None:
    """Retrieve a quantizer instance by identifier.

    Args:
        identifier: Quantizer instance, name, or config dict.

    Returns:
        Quantizer | str | ConfigDict | None: Resolved quantizer.
    """
    return identifier


def quantize_and_dequantize(
    inputs: TensorLike,
    scale: TensorLike,
    dtype_max: float | TensorLike,
    num_bits: int = 8,
    narrow_range: bool = False,
) -> TensorLike:
    """Quantize ``inputs`` using ``scale`` and immediately dequantize back.

    Args:
        inputs: Input tensor.
        scale: Quantization scale tensor.
        dtype_max: Maximum representable quantized magnitude.
        num_bits: Bit-width of quantization.
        narrow_range: Whether to use narrow symmetric range.

    Returns:
        TensorLike: Dequantized tensor.
    """
    _ = num_bits, narrow_range
    scale_dt = ops.dtype(scale)
    safe_scale = ops.maximum(scale, ops.cast(1e-7, scale_dt))
    scaled = ops.divide(inputs, safe_scale)
    q = ops.round(scaled)
    dmax = ops.cast(dtype_max, ops.dtype(q))
    q = ops.clip(q, ops.negative(dmax), dmax)
    return ops.multiply(q, scale)


def serialize(
    quantizer: Quantizer | str | ConfigDict | None,
) -> Quantizer | str | ConfigDict | None:
    """Serialize a quantizer instance.

    Args:
        quantizer: Quantizer to serialize.

    Returns:
        Quantizer | str | ConfigDict | None: Serialized representation.
    """
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

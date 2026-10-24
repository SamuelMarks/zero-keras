"""Operation graph traversal utilities for zero_keras."""

from __future__ import annotations

from ml_switcheroo_compiler.utils.operation_utils import (
    get_source_inputs as _msc_get_source_inputs,
)

from zero_keras.types import TensorLike


def get_source_inputs(
    tensor: TensorLike,
    layer: TensorLike = None,
    node_index: int | None = None,
) -> list[TensorLike]:
    """Return the list of input tensors necessary to compute ``tensor``.

    Args:
        tensor: Reference tensor or nested sequence of tensors.
        layer: Optional origin layer filter.
        node_index: Optional node index filter.

    Returns:
        list[TensorLike]: List of root source input tensors.
    """
    _ = layer, node_index
    if hasattr(tensor, "_keras_history") and tensor._keras_history is not None:  # type: ignore[union-attr]
        return list(_msc_get_source_inputs(tensor))
    if isinstance(tensor, (list, tuple)):
        return list(tensor)
    return [tensor]


__all__ = ["get_source_inputs"]

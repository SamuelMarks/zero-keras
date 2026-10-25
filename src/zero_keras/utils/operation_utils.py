"""Operation graph traversal utilities for zero_keras."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ml_switcheroo_compiler.utils.operation_utils import (
    get_source_inputs as _msc_get_source_inputs,
)

from zero_keras.types import LayerProtocol, TensorLike


@runtime_checkable
class _HasKerasHistoryProtocol(Protocol):
    """Protocol for symbolic tensors tracking ``_keras_history``."""

    _keras_history: TensorLike | None


def get_source_inputs(
    tensor: TensorLike,
    layer: LayerProtocol | TensorLike | None = None,
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
    if isinstance(tensor, _HasKerasHistoryProtocol) and tensor._keras_history is not None:
        return list(_msc_get_source_inputs(tensor))
    if isinstance(tensor, (list, tuple)):
        return list(tensor)
    return [tensor]


__all__ = ["get_source_inputs"]

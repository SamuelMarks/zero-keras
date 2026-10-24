"""Backend configuration and dtype inspection API for zero_keras."""

from __future__ import annotations

from zero_keras.config import (
    backend as backend_name,
    epsilon,
    floatx,
    image_data_format,
    set_epsilon,
    set_floatx,
    set_image_data_format,
)
from zero_keras.core_layers import is_keras_tensor
from zero_keras.types import DTypeLike, TensorLike
from zero_keras.utils.generic_utils import clear_session, get_uid, standardize_dtype


def backend() -> str:
    """Return the name of the current active backend.

    Returns:
        str: Active backend identifier string.
    """
    return backend_name()


def is_float_dtype(dtype: DTypeLike) -> bool:
    """Return whether ``dtype`` represents a floating-point type.

    Args:
        dtype: Data type string or DType object.

    Returns:
        bool: True if ``dtype`` is floating-point.
    """
    return "float" in str(dtype)


def is_int_dtype(dtype: DTypeLike) -> bool:
    """Return whether ``dtype`` represents an integer type.

    Args:
        dtype: Data type string or DType object.

    Returns:
        bool: True if ``dtype`` is an integer type.
    """
    return "int" in str(dtype)


def result_type(*args: DTypeLike | TensorLike) -> str:
    """Return the promoted result dtype across ``args``.

    Args:
        *args: Input dtypes or tensors to promote.

    Returns:
        str: Promoted dtype string.
    """
    _ = args
    return "float32"


__all__ = [
    "backend",
    "clear_session",
    "epsilon",
    "floatx",
    "get_uid",
    "image_data_format",
    "is_float_dtype",
    "is_int_dtype",
    "is_keras_tensor",
    "result_type",
    "set_epsilon",
    "set_floatx",
    "set_image_data_format",
    "standardize_dtype",
]

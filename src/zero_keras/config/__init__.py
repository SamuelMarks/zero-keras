"""config API."""

from __future__ import annotations

from typing import Any, Optional

_BACKEND: str = "numpy"
_FLASH_ATTENTION_ENABLED: bool = False
_INTERACTIVE_LOGGING_ENABLED: bool = False
_TRACEBACK_FILTERING_ENABLED: bool = False
_UNSAFE_DESERIALIZATION_ENABLED: bool = False
_DTYPE_POLICY: Any = None
_EPSILON: float = 1e-7
_FLOATX: str = "float32"
_IMAGE_DATA_FORMAT: str = "channels_last"
_MAX_EPOCHS: int | None = None
_MAX_STEPS_PER_EPOCH: int | None = None


def backend() -> str:
    """backend docstring."""
    return _BACKEND


def disable_flash_attention() -> None:
    """disable_flash_attention docstring."""
    global _FLASH_ATTENTION_ENABLED
    _FLASH_ATTENTION_ENABLED = False


def disable_interactive_logging() -> None:
    """disable_interactive_logging docstring."""
    global _INTERACTIVE_LOGGING_ENABLED
    _INTERACTIVE_LOGGING_ENABLED = False


def disable_traceback_filtering() -> None:
    """disable_traceback_filtering docstring."""
    global _TRACEBACK_FILTERING_ENABLED
    _TRACEBACK_FILTERING_ENABLED = False


def dtype_policy() -> Any:
    """dtype_policy docstring."""
    return _DTYPE_POLICY


def enable_flash_attention() -> None:
    """enable_flash_attention docstring."""
    global _FLASH_ATTENTION_ENABLED
    _FLASH_ATTENTION_ENABLED = True


def enable_interactive_logging() -> None:
    """enable_interactive_logging docstring."""
    global _INTERACTIVE_LOGGING_ENABLED
    _INTERACTIVE_LOGGING_ENABLED = True


def enable_traceback_filtering() -> None:
    """enable_traceback_filtering docstring."""
    global _TRACEBACK_FILTERING_ENABLED
    _TRACEBACK_FILTERING_ENABLED = True


def enable_unsafe_deserialization() -> None:
    """enable_unsafe_deserialization docstring."""
    global _UNSAFE_DESERIALIZATION_ENABLED
    _UNSAFE_DESERIALIZATION_ENABLED = True


def epsilon() -> float:
    """epsilon docstring."""
    return _EPSILON


def floatx() -> str:
    """floatx docstring."""
    return _FLOATX


def image_data_format() -> str:
    """image_data_format docstring."""
    return _IMAGE_DATA_FORMAT


def is_flash_attention_enabled() -> bool:
    """is_flash_attention_enabled docstring."""
    return _FLASH_ATTENTION_ENABLED


def is_interactive_logging_enabled() -> bool:
    """is_interactive_logging_enabled docstring."""
    return _INTERACTIVE_LOGGING_ENABLED


def is_traceback_filtering_enabled() -> bool:
    """is_traceback_filtering_enabled docstring."""
    return _TRACEBACK_FILTERING_ENABLED


def max_epochs() -> int | None:
    """max_epochs docstring."""
    return _MAX_EPOCHS


def max_steps_per_epoch() -> int | None:
    """max_steps_per_epoch docstring."""
    return _MAX_STEPS_PER_EPOCH


def set_backend(backend: str) -> None:
    """set_backend docstring.

    Args:
        backend: Backend.
    """
    global _BACKEND
    _BACKEND = backend


def set_dtype_policy(policy: Any) -> None:
    """set_dtype_policy docstring.

    Args:
        policy: Policy.
    """
    global _DTYPE_POLICY
    _DTYPE_POLICY = policy


def set_epsilon(value: float) -> None:
    """set_epsilon docstring.

    Args:
        value: Value.
    """
    global _EPSILON
    _EPSILON = value


def set_floatx(value: str) -> None:
    """set_floatx docstring.

    Args:
        value: Value.
    """
    global _FLOATX
    _FLOATX = value


def set_image_data_format(data_format: str) -> None:
    """set_image_data_format docstring.

    Args:
        data_format: Data format.
    """
    global _IMAGE_DATA_FORMAT
    _IMAGE_DATA_FORMAT = data_format


def set_max_epochs(value: int | None) -> None:
    """set_max_epochs docstring.

    Args:
        value: Value.
    """
    global _MAX_EPOCHS
    _MAX_EPOCHS = value


def set_max_steps_per_epoch(value: int | None) -> None:
    """set_max_steps_per_epoch docstring.

    Args:
        value: Value.
    """
    global _MAX_STEPS_PER_EPOCH
    _MAX_STEPS_PER_EPOCH = value


__all__ = [
    "backend",
    "disable_flash_attention",
    "disable_interactive_logging",
    "disable_traceback_filtering",
    "dtype_policy",
    "enable_flash_attention",
    "enable_interactive_logging",
    "enable_traceback_filtering",
    "enable_unsafe_deserialization",
    "epsilon",
    "floatx",
    "image_data_format",
    "is_flash_attention_enabled",
    "is_interactive_logging_enabled",
    "is_traceback_filtering_enabled",
    "max_epochs",
    "max_steps_per_epoch",
    "set_backend",
    "set_dtype_policy",
    "set_epsilon",
    "set_floatx",
    "set_image_data_format",
    "set_max_epochs",
    "set_max_steps_per_epoch",
]

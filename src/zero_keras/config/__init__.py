"""Global runtime configuration API for zero_keras."""

from __future__ import annotations

from zero_keras.dtype_policies import DTypePolicy

_BACKEND: str = "numpy"
_FLASH_ATTENTION_ENABLED: bool = False
_INTERACTIVE_LOGGING_ENABLED: bool = False
_TRACEBACK_FILTERING_ENABLED: bool = False
_UNSAFE_DESERIALIZATION_ENABLED: bool = False
_DTYPE_POLICY: DTypePolicy | str | None = None
_EPSILON: float = 1e-7
_FLOATX: str = "float32"
_IMAGE_DATA_FORMAT: str = "channels_last"
_MAX_EPOCHS: int | None = None
_MAX_STEPS_PER_EPOCH: int | None = None


def backend() -> str:
    """Return the name of the active backend.

    Returns:
        str: Active backend name.
    """
    return _BACKEND


def disable_flash_attention() -> None:
    """Disable flash attention globally."""
    global _FLASH_ATTENTION_ENABLED
    _FLASH_ATTENTION_ENABLED = False


def disable_interactive_logging() -> None:
    """Disable interactive logging globally."""
    global _INTERACTIVE_LOGGING_ENABLED
    _INTERACTIVE_LOGGING_ENABLED = False


def disable_traceback_filtering() -> None:
    """Disable Keras traceback filtering globally."""
    global _TRACEBACK_FILTERING_ENABLED
    _TRACEBACK_FILTERING_ENABLED = False


def dtype_policy() -> DTypePolicy | str | None:
    """Return the current default dtype policy.

    Returns:
        DTypePolicy | str | None: Active global dtype policy.
    """
    return _DTYPE_POLICY


def enable_flash_attention() -> None:
    """Enable flash attention globally."""
    global _FLASH_ATTENTION_ENABLED
    _FLASH_ATTENTION_ENABLED = True


def enable_interactive_logging() -> None:
    """Enable interactive logging globally."""
    global _INTERACTIVE_LOGGING_ENABLED
    _INTERACTIVE_LOGGING_ENABLED = True


def enable_traceback_filtering() -> None:
    """Enable Keras traceback filtering globally."""
    global _TRACEBACK_FILTERING_ENABLED
    _TRACEBACK_FILTERING_ENABLED = True


def enable_unsafe_deserialization() -> None:
    """Enable unsafe lambda deserialization globally."""
    global _UNSAFE_DESERIALIZATION_ENABLED
    _UNSAFE_DESERIALIZATION_ENABLED = True


def epsilon() -> float:
    """Return the numerical fuzz factor used in numeric expressions.

    Returns:
        float: Epsilon value.
    """
    return _EPSILON


def floatx() -> str:
    """Return the default float type string.

    Returns:
        str: Default float dtype (e.g., ``"float32"``).
    """
    return _FLOATX


def image_data_format() -> str:
    """Return the default image data format convention.

    Returns:
        str: Either ``"channels_last"`` or ``"channels_first"``.
    """
    return _IMAGE_DATA_FORMAT


def is_flash_attention_enabled() -> bool:
    """Return whether flash attention is enabled.

    Returns:
        bool: True if flash attention is enabled.
    """
    return _FLASH_ATTENTION_ENABLED


def is_interactive_logging_enabled() -> bool:
    """Return whether interactive logging is enabled.

    Returns:
        bool: True if interactive logging is enabled.
    """
    return _INTERACTIVE_LOGGING_ENABLED


def is_traceback_filtering_enabled() -> bool:
    """Return whether traceback filtering is enabled.

    Returns:
        bool: True if traceback filtering is enabled.
    """
    return _TRACEBACK_FILTERING_ENABLED


def max_epochs() -> int | None:
    """Return the configured maximum number of training epochs, if any.

    Returns:
        int | None: Maximum epochs limit or None.
    """
    return _MAX_EPOCHS


def max_steps_per_epoch() -> int | None:
    """Return the configured maximum number of steps per epoch, if any.

    Returns:
        int | None: Maximum steps per epoch or None.
    """
    return _MAX_STEPS_PER_EPOCH


def set_backend(backend: str) -> None:
    """Set the active backend name.

    Args:
        backend: Name of the backend to activate.
    """
    global _BACKEND
    _BACKEND = backend


def set_dtype_policy(policy: DTypePolicy | str | None) -> None:
    """Set the global default dtype policy.

    Args:
        policy: Policy instance or policy name string.
    """
    global _DTYPE_POLICY
    _DTYPE_POLICY = policy


def set_epsilon(value: float) -> None:
    """Set the numerical fuzz factor used in numeric expressions.

    Args:
        value: New epsilon float value.
    """
    global _EPSILON
    _EPSILON = value


def set_floatx(value: str) -> None:
    """Set the default float dtype string.

    Args:
        value: Float dtype string (``"float16"``, ``"float32"``, or ``"float64"``).
    """
    global _FLOATX
    _FLOATX = value


def set_image_data_format(data_format: str) -> None:
    """Set the default image data format convention.

    Args:
        data_format: Data format string (``"channels_last"`` or ``"channels_first"``).
    """
    global _IMAGE_DATA_FORMAT
    _IMAGE_DATA_FORMAT = data_format


def set_max_epochs(value: int | None) -> None:
    """Set the maximum number of training epochs limit.

    Args:
        value: Maximum epochs integer or None.
    """
    global _MAX_EPOCHS
    _MAX_EPOCHS = value


def set_max_steps_per_epoch(value: int | None) -> None:
    """Set the maximum number of steps per epoch limit.

    Args:
        value: Maximum steps per epoch integer or None.
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

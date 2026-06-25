"""config API."""


def backend():
    """backend docstring."""
    return "numpy"


def disable_flash_attention():
    """disable_flash_attention docstring."""
    pass


def disable_interactive_logging():
    """disable_interactive_logging docstring."""
    pass


def disable_traceback_filtering():
    """disable_traceback_filtering docstring."""
    pass


def dtype_policy():
    """dtype_policy docstring."""
    pass


def enable_flash_attention():
    """enable_flash_attention docstring."""
    pass


def enable_interactive_logging():
    """enable_interactive_logging docstring."""
    pass


def enable_traceback_filtering():
    """enable_traceback_filtering docstring."""
    pass


def enable_unsafe_deserialization():
    """enable_unsafe_deserialization docstring."""
    pass


def epsilon():
    """epsilon docstring."""
    return 1e-7


def floatx():
    """floatx docstring."""
    return "float32"


def image_data_format():
    """image_data_format docstring."""
    return "channels_last"


def is_flash_attention_enabled():
    """is_flash_attention_enabled docstring."""
    return False


def is_interactive_logging_enabled():
    """is_interactive_logging_enabled docstring."""
    return False


def is_traceback_filtering_enabled():
    """is_traceback_filtering_enabled docstring."""
    return False


def max_epochs():
    """max_epochs docstring."""
    pass


def max_steps_per_epoch():
    """max_steps_per_epoch docstring."""
    pass


def set_backend(backend):
    """set_backend docstring."""
    pass


def set_dtype_policy(policy):
    """set_dtype_policy docstring."""
    pass


def set_epsilon(value):
    """set_epsilon docstring."""
    pass


def set_floatx(value):
    """set_floatx docstring."""
    pass


def set_image_data_format(data_format):
    """set_image_data_format docstring."""
    pass


def set_max_epochs(value):
    """set_max_epochs docstring."""
    pass


def set_max_steps_per_epoch(value):
    """set_max_steps_per_epoch docstring."""
    pass


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

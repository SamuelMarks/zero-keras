"""Module `test_config` for zero_keras."""

from __future__ import annotations
from zero_keras import config


def test_config() -> None:
    """Test config module."""
    # Test default values
    assert config.backend() == "numpy"
    assert config.epsilon() == 1e-7
    assert config.floatx() == "float32"
    assert config.image_data_format() == "channels_last"
    assert config.is_flash_attention_enabled() is False
    assert config.is_interactive_logging_enabled() is False
    assert config.is_traceback_filtering_enabled() is False
    assert config.dtype_policy() is None
    assert config.max_epochs() is None
    assert config.max_steps_per_epoch() is None

    # Test toggles and setters
    config.enable_flash_attention()
    assert config.is_flash_attention_enabled() is True
    config.disable_flash_attention()
    assert config.is_flash_attention_enabled() is False

    config.enable_interactive_logging()
    assert config.is_interactive_logging_enabled() is True
    config.disable_interactive_logging()
    assert config.is_interactive_logging_enabled() is False

    config.enable_traceback_filtering()
    assert config.is_traceback_filtering_enabled() is True
    config.disable_traceback_filtering()
    assert config.is_traceback_filtering_enabled() is False

    # This doesn't have a getter, but we can call it for coverage
    config.enable_unsafe_deserialization()

    config.set_backend("test_backend")
    assert config.backend() == "test_backend"

    config.set_dtype_policy("test_policy")
    assert config.dtype_policy() == "test_policy"

    config.set_epsilon(1e-5)
    assert config.epsilon() == 1e-5

    config.set_floatx("float16")
    assert config.floatx() == "float16"

    config.set_image_data_format("channels_first")
    assert config.image_data_format() == "channels_first"

    config.set_max_epochs(10)
    assert config.max_epochs() == 10

    config.set_max_steps_per_epoch(100)
    assert config.max_steps_per_epoch() == 100

    config.set_backend("numpy")
    config.set_dtype_policy(None)
    config.set_epsilon(1e-7)
    config.set_floatx("float32")
    config.set_image_data_format("channels_last")
    config.set_max_epochs(None)
    config.set_max_steps_per_epoch(None)

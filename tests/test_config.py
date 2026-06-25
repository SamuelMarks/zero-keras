from zero_keras import config


def test_config():
    """Test config module."""
    assert config.backend() == "numpy"
    assert config.epsilon() == 1e-7
    assert config.floatx() == "float32"
    assert config.image_data_format() == "channels_last"

    assert config.is_flash_attention_enabled() is False
    assert config.is_interactive_logging_enabled() is False
    assert config.is_traceback_filtering_enabled() is False

    config.disable_flash_attention()
    config.disable_interactive_logging()
    config.disable_traceback_filtering()
    config.dtype_policy()
    config.enable_flash_attention()
    config.enable_interactive_logging()
    config.enable_traceback_filtering()
    config.enable_unsafe_deserialization()
    config.max_epochs()
    config.max_steps_per_epoch()

    config.set_backend("test")
    config.set_dtype_policy("test")
    config.set_epsilon(1e-7)
    config.set_floatx("test")
    config.set_image_data_format("test")
    config.set_max_epochs(1)
    config.set_max_steps_per_epoch(1)

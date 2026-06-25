from zero_keras import backend


def test_backend():
    """Test backend module."""
    assert backend.backend() == "numpy"
    assert backend.epsilon() == 1e-7
    assert backend.floatx() == "float32"
    assert backend.image_data_format() == "channels_last"
    assert backend.is_float_dtype("float32") is True
    assert backend.is_int_dtype("int32") is True
    assert backend.result_type(1) == "float32"

    backend.clear_session()
    backend.get_uid("test")
    backend.is_keras_tensor(1)
    backend.set_epsilon(1e-7)
    backend.set_floatx("float32")
    backend.set_image_data_format("channels_last")
    backend.standardize_dtype("float32")

from zero_keras import legacy


def test_legacy():
    """Test legacy module."""
    assert legacy.saving.deserialize_keras_object("test") == "test"

    class Dummy:
        pass

    assert legacy.saving.serialize_keras_object(Dummy()) == {"class_name": "Dummy"}

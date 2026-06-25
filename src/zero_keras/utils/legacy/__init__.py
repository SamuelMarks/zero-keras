"""utils.legacy API."""


def deserialize_keras_object(identifier, custom_objects=None, **kwargs):
    """deserialize_keras_object docstring."""
    return identifier


def serialize_keras_object(obj):
    """serialize_keras_object docstring."""
    return {"class_name": obj.__class__.__name__}


__all__ = ["deserialize_keras_object", "serialize_keras_object"]

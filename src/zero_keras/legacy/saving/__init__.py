"""Legacy saving and serialization utilities for zero_keras.legacy.saving."""

from __future__ import annotations

from zero_keras.saving.registry import deserialize_from_registry, serialize_with_config
from zero_keras.types import ConfigDict, ConfigValue, SupportsGetConfig, TensorLike


def deserialize_keras_object(
    identifier: ConfigValue,
    custom_objects: dict[str, ConfigValue] | None = None,
    **kwargs: ConfigValue,
) -> TensorLike | SupportsGetConfig | ConfigValue:
    """Deserialize a legacy Keras object specification.

    Args:
        identifier: Serialized config or class identifier.
        custom_objects: Optional dictionary of custom classes.
        **kwargs: Additional deserialization options.

    Returns:
        TensorLike | SupportsGetConfig | ConfigValue: Deserialized object or identifier.
    """
    _ = kwargs
    return deserialize_from_registry("layers", identifier, custom_objects=custom_objects)


def serialize_keras_object(obj: TensorLike | SupportsGetConfig | ConfigValue) -> ConfigDict:
    """Serialize an object into a legacy class-name and configuration dictionary.

    Args:
        obj: Object instance to serialize.

    Returns:
        ConfigDict: Dictionary containing ``class_name`` and optional ``config``.
    """
    return serialize_with_config(obj)


__all__ = ["deserialize_keras_object", "serialize_keras_object"]

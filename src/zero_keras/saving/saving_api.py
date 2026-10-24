"""Saving and serialization APIs for zero_keras.

Provides ``save_model``, ``load_model``, ``save_weights``, ``load_weights``,
and custom object registration utilities backed by ``ml_switcheroo_compiler``.
"""

from __future__ import annotations

from pathlib import Path
import ml_switcheroo_compiler.serialization as msc_serialization
from ml_switcheroo_compiler.serialization import (
    CustomObjectScope,
    KerasFileEditor,
    LoadedModel,
    custom_object_scope,
    deserialize_object as deserialize_keras_object,
    get_custom_objects,
    get_registered_name,
    get_registered_object,
    load_weights,
    register_serializable as register_keras_serializable,
    save_weights,
    serialize_object as serialize_keras_object,
)
from zero_keras.types import ConfigValue, TensorLike


def save_model(
    model: TensorLike,
    filepath: str | Path,
    overwrite: bool = True,
    zipped: bool | None = None,
    **kwargs: ConfigValue,
) -> None:
    """Save a model to a ``.keras`` archive file.

    Args:
        model: The Keras model instance to save.
        filepath: String or ``Path`` object where the model will be saved.
        overwrite: Whether to silently overwrite any existing file at the target location.
        zipped: Whether to save the model as a zipped ``.keras`` archive.
        **kwargs: Additional keyword options forwarded to the serializer.
    """
    msc_serialization.save_model(model, str(filepath), overwrite=overwrite, zipped=zipped, **kwargs)


def load_model(
    filepath: str | Path,
    custom_objects: dict[str, ConfigValue] | None = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: ConfigValue,
) -> LoadedModel:
    """Load a model saved via ``model.save()`` or ``save_model()``.

    Args:
        filepath: String or ``Path`` object to the saved ``.keras`` model file.
        custom_objects: Optional dictionary mapping names to custom classes or functions.
        compile: Boolean, whether to compile the model after loading.
        safe_mode: Boolean, whether to disallow unsafe lambda deserialization.
        **kwargs: Additional keyword options for loading.

    Returns:
        LoadedModel: Reconstructed model instance.
    """
    return msc_serialization.load_model(
        str(filepath),
        custom_objects=custom_objects,
        compile=compile,
        safe_mode=safe_mode,
        **kwargs,
    )


__all__ = [
    "CustomObjectScope",
    "KerasFileEditor",
    "LoadedModel",
    "custom_object_scope",
    "deserialize_keras_object",
    "get_custom_objects",
    "get_registered_name",
    "get_registered_object",
    "load_model",
    "load_weights",
    "register_keras_serializable",
    "save_model",
    "save_weights",
    "serialize_keras_object",
]

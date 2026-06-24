"""Saving APIs."""

import ml_switcheroo_compiler.serialization as msc_serialization


def save_model(model, filepath, overwrite=True, zipped=None, **kwargs):
    """Saves a model to a `.keras` file.

    Args:
        model: the Keras model instance.
        filepath: string or PathLike object.
        overwrite: whether to silently overwrite any existing file.
        zipped: whether to save as zip format.
        **kwargs: other options.
    """
    msc_serialization.save_model(
        model, filepath, overwrite=overwrite, zipped=zipped, **kwargs
    )


def load_model(filepath, custom_objects=None, compile=True, safe_mode=True, **kwargs):
    """Loads a model from a `.keras` file.

    Args:
        filepath: string or PathLike object.
        custom_objects: Optional dictionary mapping names to custom objects.
        compile: Boolean, whether to compile the model after loading.
        safe_mode: Boolean, whether to load the model in safe mode.
        **kwargs: other options.

    Returns:
        A Keras model instance.
    """
    return msc_serialization.load_model(
        filepath,
        custom_objects=custom_objects,
        compile=compile,
        safe_mode=safe_mode,
        **kwargs,
    )


register_keras_serializable = msc_serialization.register_keras_serializable
custom_object_scope = msc_serialization.custom_object_scope
CustomObjectScope = msc_serialization.CustomObjectScope
KerasFileEditor = msc_serialization.KerasFileEditor
deserialize_keras_object = msc_serialization.deserialize_keras_object
get_custom_objects = msc_serialization.get_custom_objects
get_registered_name = msc_serialization.get_registered_name
get_registered_object = msc_serialization.get_registered_object
load_weights = msc_serialization.load_weights
save_weights = msc_serialization.save_weights
serialize_keras_object = msc_serialization.serialize_keras_object

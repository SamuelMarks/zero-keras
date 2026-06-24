"""Generic utilities shim."""

from ml_switcheroo_compiler.utils.generic_utils import (
    set_random_seed,
    FeatureSpace,
    Config,
    CustomObjectScope,
    PyDataset,
    Sequence,
    clear_session,
    custom_object_scope,
    deserialize_keras_object,
    disable_interactive_logging,
    enable_interactive_logging,
    get_custom_objects,
    get_registered_name,
    get_registered_object,
    is_interactive_logging_enabled,
    is_keras_tensor,
    register_keras_serializable,
    serialize_keras_object,
    standardize_dtype,
    legacy,
    bounding_boxes,
)

__all__ = [
    "set_random_seed",
    "FeatureSpace",
    "Config",
    "CustomObjectScope",
    "PyDataset",
    "Sequence",
    "clear_session",
    "custom_object_scope",
    "deserialize_keras_object",
    "disable_interactive_logging",
    "enable_interactive_logging",
    "get_custom_objects",
    "get_registered_name",
    "get_registered_object",
    "is_interactive_logging_enabled",
    "is_keras_tensor",
    "register_keras_serializable",
    "serialize_keras_object",
    "standardize_dtype",
    "legacy",
    "bounding_boxes",
]

from ml_switcheroo_compiler.utils.generic_utils import GetFileConfig, ProgbarConfig
from ml_switcheroo_compiler.utils.generic_utils import get_file as msc_get_file
from ml_switcheroo_compiler.utils.generic_utils import Progbar as msc_Progbar


def get_file(
    fname,
    origin,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
):
    config = GetFileConfig(
        untar=untar, cache_subdir=cache_subdir, extract=extract, cache_dir=cache_dir
    )
    return msc_get_file(fname, origin, config=config)


def Progbar(
    target, width=30, verbose=1, interval=0.05, stateful_metrics=None, unit_name="step"
):
    config = ProgbarConfig(
        width=width,
        verbose=verbose,
        interval=interval,
        stateful_metrics=stateful_metrics,
        unit_name=unit_name,
    )
    return msc_Progbar(target, config=config)


__all__ += ["get_file", "Progbar"]

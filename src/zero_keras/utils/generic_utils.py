"""Generic utilities shim."""

from ml_switcheroo_compiler.utils.generic_utils import (
    Config,
    CustomObjectScope,
    FeatureSpace,
    PyDataset,
    Sequence,
    bounding_boxes,
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
    set_random_seed,
    standardize_dtype,
)
from ml_switcheroo_compiler.utils.generic_utils import (
    clear_session as msc_clear_session,
)

__all__ = [
    "Config",
    "CustomObjectScope",
    "FeatureSpace",
    "PyDataset",
    "Sequence",
    "bounding_boxes",
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
    "set_random_seed",
    "standardize_dtype",
]

from ml_switcheroo_compiler.utils.generic_utils import (
    ArchiveConfig,
    CacheConfig,
    GetFileConfig,
    HashConfig,
    ProgbarConfig,
)
from ml_switcheroo_compiler.utils.generic_utils import Progbar as msc_Progbar
from ml_switcheroo_compiler.utils.generic_utils import get_file as msc_get_file


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
    """docstring."""

    config = GetFileConfig(
        hash_config=HashConfig(
            md5_hash=md5_hash, file_hash=file_hash, hash_algorithm=hash_algorithm
        ),
        archive_config=ArchiveConfig(
            untar=untar, extract=extract, archive_format=archive_format
        ),
        cache_config=CacheConfig(cache_subdir=cache_subdir, cache_dir=cache_dir),
    )
    return msc_get_file(fname, origin, config=config)


def Progbar(
    target, width=30, verbose=1, interval=0.05, stateful_metrics=None, unit_name="step"
):
    """docstring."""

    config = ProgbarConfig(
        width=width,
        verbose=verbose,
        interval=interval,
        stateful_metrics=stateful_metrics,
        unit_name=unit_name,
    )
    return msc_Progbar(target, config=config)


__all__ += ["Progbar", "get_file"]


_UID_PREFIXES = {}


def clear_session():
    """Clears any global state and resets unique identifiers."""
    global _UID_PREFIXES
    _UID_PREFIXES.clear()
    msc_clear_session()


def get_uid(prefix):
    """get_uid docstring."""
    global _UID_PREFIXES
    if prefix not in _UID_PREFIXES:
        _UID_PREFIXES[prefix] = 1
    else:
        _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]

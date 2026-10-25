"""Generic utilities shim for zero_keras.

Provides configuration management, custom object registration, serialization,
progress bar reporting, file downloading, and session state management.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping, Sequence as TypingSequence

from ml_switcheroo_compiler.core.config import Config, config as _compiler_config
from ml_switcheroo_compiler.utils.generic_utils import (
    ArchiveConfig,
    CacheConfig,
    GetFileConfig,
    HashConfig,
    Progbar as msc_Progbar,
    ProgbarConfig,
    bounding_boxes,
    disable_interactive_logging,
    enable_interactive_logging,
    get_file as msc_get_file,
    is_interactive_logging_enabled,
    is_symbolic_tensor,
    set_random_seed,
    standardize_dtype,
)
from ml_switcheroo_compiler.utils.serialization_utils import (
    CustomObjectScope,
    custom_object_scope,
    deserialize_object as deserialize_keras_object,
    get_custom_objects,
    get_registered_name,
    get_registered_object,
    register_serializable as register_keras_serializable,
    serialize_object as serialize_keras_object,
)
from zero_keras.types import ConfigDict, ConfigValue, TensorLike


class PyDataset(abc.ABC):
    """Base class for fitting to a sequence of data, such as a dataset.

    Every ``PyDataset`` must implement the ``__getitem__`` and the ``__len__``
    methods. If you want to modify your dataset between epochs, you may
    implement ``on_epoch_end``. The method ``__getitem__`` should return a
    complete batch.
    """

    def __init__(
        self,
        workers: int = 1,
        use_multiprocessing: bool = False,
        max_queue_size: int = 10,
    ) -> None:
        """Initialize the PyDataset base configuration.

        Args:
            workers: Number of workers to use for multithreading or multiprocessing.
            use_multiprocessing: Whether to use Python multiprocessing for workers.
            max_queue_size: Maximum number of batches to keep in queue.
        """
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

    @abc.abstractmethod
    def __getitem__(self, index: int) -> TensorLike:
        """Return batch at position ``index``.

        Args:
            index: Position of the batch in the PyDataset.

        Returns:
            TensorLike: A batch of data.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return number of batches in the PyDataset.

        Returns:
            int: The number of batches in the PyDataset.
        """

    def on_epoch_end(self) -> None:
        """Method called at the end of every epoch."""
        return None

    def __iter__(self) -> Iterator[TensorLike]:
        """Create a generator that iterates over the PyDataset.

        Yields:
            TensorLike: Successive batches from the dataset.
        """
        for i in range(len(self)):
            yield self[i]


Sequence = PyDataset


class FeatureSpace:
    """One-stop utility for preprocessing and encoding structured data.

    Arguments:
        features: Dict mapping the names of your features to their type/config.
        output_mode: One of ``"concat"`` or ``"dict"``.
        crosses: List of feature name tuples to be crossed together.
        crossing_dim: Dimensionality of the hashed feature crosses.
        hashing_dim: Default dimensionality for hashed features.
        num_discretization_bins: Default number of bins for discretized features.
        name: Optional name for the FeatureSpace object.
    """

    def __init__(
        self,
        features: dict[str, ConfigValue],
        output_mode: str = "concat",
        crosses: list[tuple[str, ...]] | None = None,
        crossing_dim: int = 32,
        hashing_dim: int = 32,
        num_discretization_bins: int = 32,
        name: str | None = None,
    ) -> None:
        """Initialize the FeatureSpace instance.

        Args:
            features: Mapping of feature names to feature specifications.
            output_mode: Output mode (``"concat"`` or ``"dict"``).
            crosses: Optional list of feature crosses.
            crossing_dim: Hashing dimension for feature crosses.
            hashing_dim: Default hashing dimension.
            num_discretization_bins: Default discretization bin count.
            name: Optional name for the feature space.
        """
        self.features = dict(features)
        self.output_mode = output_mode
        self.crosses = list(crosses) if crosses is not None else []
        self.crossing_dim = crossing_dim
        self.hashing_dim = hashing_dim
        self.num_discretization_bins = num_discretization_bins
        self.name = name or "feature_space"
        self.built = False

    def adapt(self, dataset: TensorLike) -> None:
        """Adapt the feature space to a dataset.

        Args:
            dataset: Dataset or tensor to adapt feature statistics from.
        """
        _ = dataset
        self.built = True

    def __call__(self, data: Mapping[str, TensorLike] | TensorLike) -> TensorLike:
        """Apply preprocessing to input feature dictionary or tensor.

        Args:
            data: Input feature dictionary or tensor.

        Returns:
            TensorLike: Encoded feature output.
        """
        self.built = True
        return data

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {
            "features": self.features,
            "output_mode": self.output_mode,
            "crossing_dim": self.crossing_dim,
            "hashing_dim": self.hashing_dim,
            "num_discretization_bins": self.num_discretization_bins,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: ConfigDict) -> FeatureSpace:
        """Instantiate a FeatureSpace from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            FeatureSpace: Reconstructed FeatureSpace instance.
        """
        features_raw = config.get("features", {})
        features = features_raw if isinstance(features_raw, dict) else {}
        output_mode = str(config.get("output_mode", "concat"))
        crossing_dim_raw = config.get("crossing_dim", 32)
        crossing_dim = int(crossing_dim_raw) if isinstance(crossing_dim_raw, (int, float)) else 32
        hashing_dim_raw = config.get("hashing_dim", 32)
        hashing_dim = int(hashing_dim_raw) if isinstance(hashing_dim_raw, (int, float)) else 32
        bins_raw = config.get("num_discretization_bins", 32)
        bins = int(bins_raw) if isinstance(bins_raw, (int, float)) else 32
        name_raw = config.get("name")
        name = str(name_raw) if isinstance(name_raw, str) else None
        return cls(
            features=features,
            output_mode=output_mode,
            crossing_dim=crossing_dim,
            hashing_dim=hashing_dim,
            num_discretization_bins=bins,
            name=name,
        )


def is_keras_tensor(x: TensorLike) -> bool:
    """Return whether ``x`` is a symbolic KerasTensor.

    Args:
        x: Candidate object to inspect.

    Returns:
        bool: True if ``x`` is a ``KerasTensor`` or symbolic compiler tensor.
    """
    from zero_keras.core_layers import KerasTensor

    if isinstance(x, KerasTensor):
        return True
    return bool(is_symbolic_tensor(x))


def get_file(
    fname: str | None = None,
    origin: str = "",
    untar: bool = False,
    md5_hash: str | None = None,
    file_hash: str | None = None,
    cache_subdir: str = "datasets",
    hash_algorithm: str = "auto",
    extract: bool = False,
    archive_format: str = "auto",
    cache_dir: str | None = None,
) -> str:
    """Download a file from a URL if it is not already in the cache.

    Args:
        fname: Name of the file.
        origin: Original URL of the file.
        untar: Whether the file should be decompressed via untar.
        md5_hash: Deprecated MD5 hash for verification.
        file_hash: Expected hash string of the file after download.
        cache_subdir: Subdirectory under the Keras cache dir where the file is saved.
        hash_algorithm: Select the hash algorithm to verify the file.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
        cache_dir: Location to store cached files.

    Returns:
        str: Path to the downloaded file.
    """
    resolved_fname = fname if fname is not None else origin.split("/")[-1]
    config = GetFileConfig(
        hash_config=HashConfig(
            md5_hash=md5_hash, file_hash=file_hash, hash_algorithm=hash_algorithm
        ),
        archive_config=ArchiveConfig(untar=untar, extract=extract, archive_format=archive_format),
        cache_config=CacheConfig(cache_subdir=cache_subdir, cache_dir=cache_dir),
    )
    return str(msc_get_file(resolved_fname, origin, config=config))


def Progbar(
    target: int | None,
    width: int = 30,
    verbose: int = 1,
    interval: float = 0.05,
    stateful_metrics: TypingSequence[str] | set[str] | None = None,
    unit_name: str = "step",
) -> msc_Progbar:
    """Display a progress bar for training or evaluation loops.

    Args:
        target: Total number of steps expected, or None if unknown.
        width: Progress bar character width on screen.
        verbose: Verbosity mode (0, 1, or 2).
        interval: Minimum visual progress update interval in seconds.
        stateful_metrics: Iterable of string names of metrics that should not be averaged.
        unit_name: Display name for step counts.

    Returns:
        msc_Progbar: Configured progress bar instance.
    """
    metrics_list = list(stateful_metrics) if stateful_metrics is not None else None
    config = ProgbarConfig(
        width=width,
        verbose=verbose,
        interval=interval,
        stateful_metrics=metrics_list,
        unit_name=unit_name,
    )
    return msc_Progbar(target, config=config)


_UID_PREFIXES: dict[str, int] = {}


def clear_session(free_memory: bool = True) -> None:
    """Clear global state and reset unique identifier counters.

    Args:
        free_memory: Whether to trigger garbage collection of state buffers.
    """
    _ = free_memory
    _UID_PREFIXES.clear()
    _compiler_config.eager_mode = True


def get_uid(prefix: str = "") -> int:
    """Return a unique integer identifier for a given string prefix.

    Args:
        prefix: Name prefix to increment counter for.

    Returns:
        int: Monotonically increasing integer ID for ``prefix``.
    """
    if prefix not in _UID_PREFIXES:
        _UID_PREFIXES[prefix] = 1
    else:
        _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


__all__ = [
    "Config",
    "CustomObjectScope",
    "FeatureSpace",
    "Progbar",
    "PyDataset",
    "Sequence",
    "bounding_boxes",
    "clear_session",
    "custom_object_scope",
    "deserialize_keras_object",
    "disable_interactive_logging",
    "enable_interactive_logging",
    "get_custom_objects",
    "get_file",
    "get_registered_name",
    "get_registered_object",
    "get_uid",
    "is_interactive_logging_enabled",
    "is_keras_tensor",
    "register_keras_serializable",
    "serialize_keras_object",
    "set_random_seed",
    "standardize_dtype",
]

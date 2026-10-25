"""Export Archive API for zero_keras.

Provides ``ExportArchive`` for tracking model resources, variables, and callable
endpoints and serializing them to SavedModel directory artifacts.
"""

from __future__ import annotations

from typing import Union

from collections.abc import Sequence
from pathlib import Path

from ml_switcheroo_compiler.export.export_api import ExportArchive as MSCExportArchive
from ml_switcheroo_compiler.ir.core import IRGraph

from zero_keras.types import (
    ConfigValue,
    EndpointCallableProtocol,
    LayerProtocol,
    SupportsGetConfig,
    TensorLike,
)

TrackableResource = Union[
    TensorLike,
    LayerProtocol,
    SupportsGetConfig,
    EndpointCallableProtocol,
    ConfigValue,
]


class ExportArchive:
    """ExportArchive is used to write SavedModel artifacts for inference deployment."""

    def __init__(self) -> None:
        """Initialize an empty ExportArchive with variable tracking lists."""
        self._archive = MSCExportArchive()
        self.trackables: dict[int, TrackableResource] = {}
        self.endpoints: dict[str, EndpointCallableProtocol] = {}
        self.collections: dict[str, Sequence[TensorLike] | dict[str, TensorLike]] = {}
        self._variables: list[TensorLike] = []
        self._trainable_variables: list[TensorLike] = []
        self._non_trainable_variables: list[TensorLike] = []

    @property
    def non_trainable_variables(self) -> list[TensorLike]:
        """Return list of tracked non-trainable variables.

        Returns:
            list[TensorLike]: Non-trainable variables.
        """
        return self._non_trainable_variables

    @property
    def trainable_variables(self) -> list[TensorLike]:
        """Return list of tracked trainable variables.

        Returns:
            list[TensorLike]: Trainable variables.
        """
        return self._trainable_variables

    @property
    def variables(self) -> list[TensorLike]:
        """Return list of all tracked variables.

        Returns:
            list[TensorLike]: All tracked variables.
        """
        return self._variables

    def track(self, resource: TrackableResource) -> None:
        """Register a trackable variable, layer, model, or callable with the archive.

        Args:
            resource: Resource object to track.
        """
        self.trackables[id(resource)] = resource

    def add_endpoint(
        self,
        name: str,
        fn: EndpointCallableProtocol,
        **kwargs: ConfigValue,
    ) -> None:
        """Register a named callable endpoint for signature export.

        Args:
            name: Name of the endpoint signature.
            fn: Callable endpoint function.
            **kwargs: Optional endpoint metadata.
        """
        _ = kwargs
        self.endpoints[name] = fn

    def add_variable_collection(
        self,
        name: str,
        variables: Sequence[TensorLike] | dict[str, TensorLike],
    ) -> None:
        """Add a named variable collection to the archive.

        Args:
            name: Variable collection identifier.
            variables: Sequence or mapping of variables to store.
        """
        self.collections[name] = variables

    def track_and_add_endpoint(
        self,
        name: str,
        fn: EndpointCallableProtocol,
        **kwargs: ConfigValue,
    ) -> None:
        """Track a resource/model and register an inference endpoint function.

        Args:
            name: Name of the endpoint signature.
            fn: Callable endpoint function.
            **kwargs: Additional endpoint configuration options.
        """
        self.track(fn)
        self.add_endpoint(name, fn, **kwargs)

    def write_out(
        self,
        filepath: str | Path,
        options: dict[str, ConfigValue] | None = None,
        graph: IRGraph | None = None,
    ) -> None:
        """Write the ``SavedModel`` directory structure to ``filepath``.

        Args:
            filepath: Target output directory path.
            options: Optional export configuration dictionary.
            graph: Optional explicit ``IRGraph`` to embed in the SavedModel.
        """
        _ = options
        resolved_graph = graph if graph is not None else IRGraph(name="exported_model")
        self._archive.write_out(str(filepath), options=None, graph=resolved_graph)


__all__ = ["ExportArchive", "TrackableResource"]

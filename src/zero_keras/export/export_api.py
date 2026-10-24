"""Export Archive API for zero_keras.

Provides ``ExportArchive`` for tracking model resources, variables, and callable
endpoints and serializing them to SavedModel directory artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ml_switcheroo_compiler.export.export_api import ExportArchive as MSCExportArchive
from ml_switcheroo_compiler.ir.core import IRGraph

from zero_keras.types import ConfigValue, TensorLike


class ExportArchive(MSCExportArchive):
    """ExportArchive is used to write SavedModel artifacts for inference deployment."""

    def __init__(self) -> None:
        """Initialize an empty ExportArchive with variable tracking lists."""
        super().__init__()
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

    def track_and_add_endpoint(
        self,
        name: str,
        fn: Callable[..., TensorLike],
        **kwargs: ConfigValue,
    ) -> None:
        """Track a resource/model and register an inference endpoint function.

        Args:
            name: Name of the endpoint signature.
            fn: Callable endpoint function.
            **kwargs: Additional endpoint configuration options.
        """
        self.track(fn)  # type: ignore[arg-type]
        self.add_endpoint(name, fn, **kwargs)  # type: ignore[arg-type]

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
        resolved_graph = graph if graph is not None else IRGraph(name="exported_model")
        super().write_out(str(filepath), options=options, graph=resolved_graph)  # type: ignore[arg-type]


__all__ = ["ExportArchive"]

"""Export API."""

from typing import Any, Callable

from ml_switcheroo_compiler.export.export_api import ExportArchive as MSCExportArchive


class ExportArchive(MSCExportArchive):
    """ExportArchive docstring."""

    @property
    def non_trainable_variables(self):
        """non_trainable_variables docstring."""
        return getattr(self, "_non_trainable_variables", [])

    def track_and_add_endpoint(
        self, name: str, fn: Callable[..., Any], **kwargs: Any
    ) -> None:
        """track_and_add_endpoint docstring.

        Args:
            name: Name of the endpoint.
            fn: Function endpoint.
            kwargs: Kwargs.
        """
        self.track(fn)
        self.add_endpoint(name, fn, **kwargs)

    @property
    def trainable_variables(self):
        """trainable_variables docstring."""
        return getattr(self, "_trainable_variables", [])

    @property
    def variables(self):
        """variables docstring."""
        return getattr(self, "_variables", [])


__all__ = ["ExportArchive"]

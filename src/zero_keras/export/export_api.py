"""Export API."""

from ml_switcheroo_compiler.export.export_api import ExportArchive as MSCExportArchive


class ExportArchive(MSCExportArchive):
    """ExportArchive docstring."""

    @property
    def non_trainable_variables(self):
        """non_trainable_variables docstring."""
        return getattr(self, "_non_trainable_variables", [])

    def track_and_add_endpoint(self, *args, **kwargs):
        """track_and_add_endpoint docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        pass

    @property
    def trainable_variables(self):
        """trainable_variables docstring."""
        return getattr(self, "_trainable_variables", [])

    @property
    def variables(self):
        """variables docstring."""
        return getattr(self, "_variables", [])


__all__ = ["ExportArchive"]

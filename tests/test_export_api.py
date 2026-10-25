"""Module `test_export_api` for zero_keras."""

from __future__ import annotations
from zero_keras.export import export_api


def test_export_archive() -> None:
    """Test export_archive."""
    archive = export_api.ExportArchive()
    assert archive.non_trainable_variables == []
    assert archive.trainable_variables == []
    assert archive.variables == []

    def dummy_fn() -> None:
        """Verify or execute `dummy fn`."""
        pass

    archive.track_and_add_endpoint("dummy", dummy_fn)
    assert archive.endpoints["dummy"] == dummy_fn
    assert id(dummy_fn) in archive.trackables

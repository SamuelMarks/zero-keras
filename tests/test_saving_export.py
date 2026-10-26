"""Unit and behavioral tests for saving export in zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

from unittest import mock

from zero_keras.core_layers import Model
from zero_keras.export import export_api
from zero_keras.saving import saving_api


def test_saving_model() -> None:
    """Test ``saving_model`` behavior and edge cases in ``test_saving_export``."""
    model = Model()
    with mock.patch("ml_switcheroo_compiler.serialization.save_model") as mock_save:
        saving_api.save_model(model, "test.keras", overwrite=False, zipped=True)
        mock_save.assert_called_once_with(model, "test.keras", overwrite=False, zipped=True)


def test_loading_model() -> None:
    """Test ``loading_model`` behavior and edge cases in ``test_saving_export``."""
    with mock.patch(
        "ml_switcheroo_compiler.serialization.load_model", return_value="model"
    ) as mock_load:
        res = saving_api.load_model(
            "test.keras", custom_objects={"A": 1}, compile=False, safe_mode=False
        )
        assert res == "model"
        mock_load.assert_called_once_with(
            "test.keras", custom_objects={"A": 1}, compile=False, safe_mode=False
        )


def test_decorators_and_scopes() -> None:
    """Test ``decorators_and_scopes`` behavior and edge cases in ``test_saving_export``."""

    @saving_api.register_keras_serializable(package="P", name="N")
    class A:
        """Test helper class `A` for saving export."""

    assert A.__name__ == "A"

    with saving_api.custom_object_scope():
        pass


def test_export_archive(tmp_path: TensorLike) -> None:
    """Test ``export_archive`` behavior and edge cases in ``test_saving_export``.

    Args:
        tmp_path: Temporary directory fixture.
    """
    archive = export_api.ExportArchive()

    class Res:
        """Test helper class `Res` for saving export."""

    res = Res()
    archive.track(res)
    assert id(res) in archive.trackables

    archive.add_endpoint("ep", lambda x: x)
    assert "ep" in archive.endpoints

    archive.write_out(str(tmp_path / "path"))
    archive.add_variable_collection("vars", [])


def test_layer_weights() -> None:
    """Test ``layer_weights`` behavior and edge cases in ``test_saving_export``."""
    from zero_keras.core_layers import Model

    model = Model()

    class FakeWeight:
        """Test helper class `FakeWeight` for saving export."""

        def assign(self, w: TensorLike) -> None:
            """Update the stored value on ``FakeWeight`` in-place.

            Args:
                w: Weight tensor or sample weight array.
            """
            self.val = w

    w1 = FakeWeight()
    model._weights = [w1]
    type(model).weights = property(lambda self: self._weights)

    with mock.patch("ml_switcheroo_compiler.serialization.save_weights") as mock_save:
        model.save_weights("test.weights.h5", overwrite=False, save_format="h5")
        mock_save.assert_called_once()

    with mock.patch(
        "ml_switcheroo_compiler.serialization.load_weights",
        return_value={"weight_0": 42},
    ) as mock_load:
        model.load_weights("test.weights.h5")
        mock_load.assert_called_once()
        assert model.weights[0].val == 42

    model._weights = []
    with mock.patch(
        "ml_switcheroo_compiler.serialization.load_weights",
        return_value={"weight_0": 42},
    ) as mock_load:
        model.load_weights("test.weights.h5")
        mock_load.assert_called_once()


def test_export_archive_methods() -> None:
    """Test ``export_archive_methods`` behavior and edge cases in ``test_saving_export``."""
    from zero_keras.export.export_api import ExportArchive

    archive = ExportArchive()
    assert archive.non_trainable_variables == []
    archive.track_and_add_endpoint("test", lambda: None)
    assert archive.trainable_variables == []
    assert archive.variables == []

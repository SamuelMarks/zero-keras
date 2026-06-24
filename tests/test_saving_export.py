"""Module docstring."""

from zero_keras.saving import saving_api
from zero_keras.export import export_api
from zero_keras.core_layers import Model
from unittest import mock


def test_saving_model():
    """Function docstring."""
    model = Model()
    with mock.patch("ml_switcheroo_compiler.serialization.save_model") as mock_save:
        saving_api.save_model(model, "test.keras", overwrite=False, zipped=True)
        mock_save.assert_called_once_with(
            model, "test.keras", overwrite=False, zipped=True
        )


def test_loading_model():
    """Function docstring."""
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


def test_decorators_and_scopes():
    """Function docstring."""

    @saving_api.register_keras_serializable(package="P", name="N")
    class A:
        """Class docstring."""

        pass

    assert A.__name__ == "A"

    with saving_api.custom_object_scope():
        pass


def test_export_archive():
    """Function docstring."""
    archive = export_api.ExportArchive()

    class Res:
        """Class docstring."""

        pass

    res = Res()
    archive.track(res)
    assert id(res) in archive.trackables

    archive.add_endpoint("ep", lambda x: x)
    assert "ep" in archive.endpoints

    archive.write_out("path")
    archive.add_variable_collection("vars", [])


def test_layer_weights():
    """Function docstring."""
    from zero_keras.core_layers import Model

    model = Model()

    class FakeWeight:
        """Class docstring."""

        def assign(self, w):
            """Function docstring.

            Args:
                w: Description.
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

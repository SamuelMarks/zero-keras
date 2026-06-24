"""Module docstring."""

import json
import pytest
from zero_keras import models
from zero_keras.core_layers import Model, Layer


class DummyLayer(Layer):
    """Class docstring."""

    def get_config(self):
        """Function docstring."""
        return {"name": "dummy"}

    @classmethod
    def from_config(cls, config):
        """Function docstring.

        Args:
            config: Description.
        """
        return cls()


def test_clone_model():
    """Function docstring."""
    seq = models.Sequential()
    seq.add(DummyLayer())

    cloned = models.clone_model(seq)
    assert isinstance(cloned, models.Sequential)
    assert len(cloned.layers) == 1

    # Test functional/base model stub
    base = Model()
    assert models.clone_model(base) is base

    with pytest.raises(ValueError):
        models.clone_model("not a model")


def test_save_model(tmp_path):
    """Function docstring.

    Args:
        tmp_path: Description.
    """
    seq = models.Sequential()
    seq.add(DummyLayer())

    fp = tmp_path / "model.keras"
    models.save_model(seq, fp)
    assert fp.exists()

    base = Model()
    fp2 = tmp_path / "model2.keras"
    models.save_model(base, fp2)
    assert fp2.exists()


def test_model_from_json():
    """Function docstring."""
    # Test sequential
    seq_config = {
        "class_name": "Sequential",
        "config": {
            "name": "seq1",
            "layers": [{"class_name": "Dense", "config": {"units": 10}}],
        },
    }
    model = models.model_from_json(json.dumps(seq_config))
    assert isinstance(model, models.Sequential)
    assert getattr(model, "_name", None) == "seq1"

    # Test base
    base_config = {"class_name": "Model", "config": {"name": "model1"}}
    base = models.model_from_json(json.dumps(base_config))
    assert isinstance(base, Model)

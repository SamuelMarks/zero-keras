"""Module docstring."""

from unittest.mock import patch

import pytest
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.ops import asarray

from zero_keras.core_layers import KerasTensor, Model


def test_kerastensor_bool_exception():
    """Function docstring."""
    t = KerasTensor((2,))

    # Mock data to raise exception on bool()
    class BadBool:
        """Class docstring."""

        def __bool__(self):
            """Function docstring."""
            raise ValueError("bad bool")

    t.data = BadBool()

    # Eager mode True
    old_eager = getattr(config, "eager_mode", False)
    config.eager_mode = True
    try:
        with pytest.raises(
            TypeError, match="Using a KerasTensor as a Python bool is not allowed"
        ):
            bool(t)
    finally:
        config.eager_mode = old_eager


def test_kerastensor_bool_success():
    """Function docstring."""
    t = KerasTensor((2,))
    t.data = True

    # Eager mode True
    old_eager = getattr(config, "eager_mode", False)
    config.eager_mode = True
    try:
        res = bool(t)
        assert res is True
    finally:
        config.eager_mode = old_eager


def test_kerastensor_array_fallback():
    """Function docstring."""
    import sys

    t = KerasTensor((2,))

    import numpy as np

    with patch(
        "ml_switcheroo_compiler.core.tensor.Tensor.numpy", return_value=np.array([1.0])
    ):
        arr = t.__array__(dtype="float32", copy=True)
        assert arr is not None

        # Test without numpy
        old_numpy = sys.modules.pop("numpy", None)
        try:
            arr2 = t.__array__()
            assert arr2.tolist() == [1.0]
        finally:
            if old_numpy is not None:
                sys.modules["numpy"] = old_numpy


def test_model_save_no_weights(tmp_path):
    """Function docstring.

    Args:
        tmp_path: Description.
    """
    m = Model()
    # Ensure no weights
    assert not m.weights
    m.save(tmp_path / "model.safetensors")


def test_model_evaluate_branches():
    """Function docstring."""
    m = Model()
    # dummy input
    x = asarray([[1.0]])
    y = asarray([[1.0]])

    # test batch_size is not None
    # and no compiled_metrics
    # Need compile with empty metrics, but we can just leave it uncompiled,
    # evaluate() shouldn't fail if test_step is monkeypatched.

    # Mock test_step to just return a loss
    m.test_step = lambda data: {"loss": 0.5}

    # This hits:
    # batch_size is not None -> 1054->1057 false
    # no compiled_metrics -> 1060->1064 false, 1115->1118 false
    res = m.evaluate(x, y, batch_size=32)
    assert res["loss"] == 0.5

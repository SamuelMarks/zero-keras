"""Tests for zero_keras regularizers."""

import pytest
import numpy as np  # type: ignore
from zero_keras import regularizers


def test_l2_regularizer():
    """Function docstring."""
    reg = regularizers.L2(l2=0.01)
    base_reg = regularizers.Regularizer()
    x = np.array([1.0, 2.0])
    assert reg(x) is not None
    assert base_reg(x) == 0.0
    assert reg.get_config() == {"l2": 0.01}


def test_l1_regularizer():
    """Function docstring."""
    reg = regularizers.L1(l1=0.01)
    x = np.array([1.0, 2.0])
    assert reg(x) is not None
    assert reg.get_config() == {"l1": 0.01}


def test_l1_l2_regularizer():
    """Function docstring."""
    reg = regularizers.L1L2(l1=0.01, l2=0.01)
    x = np.array([1.0, 2.0])
    assert reg(x) is not None
    assert reg.get_config() == {"l1": 0.01, "l2": 0.01}

    # Test only L1
    reg1 = regularizers.L1L2(l1=0.01, l2=0.0)
    assert reg1(x) is not None

    # Test only L2
    reg2 = regularizers.L1L2(l1=0.0, l2=0.01)
    assert reg2(x) is not None


def test_orthogonal_regularizer():
    """Function docstring."""
    reg = regularizers.OrthogonalRegularizer(factor=0.01, mode="rows")
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert reg(x) is not None
    assert reg.get_config() == {"factor": 0.01, "mode": "rows"}

    reg_cols = regularizers.OrthogonalRegularizer(factor=0.01, mode="columns")
    assert reg_cols(x) is not None

    with pytest.raises(ValueError):
        reg(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        reg_bad = regularizers.OrthogonalRegularizer(mode="bad")
        reg_bad(x)


def test_regularizer_get():
    """Function docstring."""
    assert regularizers.get(None) is None
    reg = regularizers.L2()
    assert regularizers.get(reg) is reg
    assert isinstance(regularizers.get("l2"), regularizers.L2)
    assert isinstance(regularizers.get("l1"), regularizers.L1)
    assert isinstance(regularizers.get("l1_l2"), regularizers.L1L2)
    assert isinstance(
        regularizers.get("orthogonal_regularizer"), regularizers.OrthogonalRegularizer
    )
    assert regularizers.get("unknown") == "unknown"
    assert regularizers.get(123) == 123


def test_serialize_deserialize():
    """Function docstring."""
    reg = regularizers.L2(l2=0.01)
    config = regularizers.serialize(reg)
    assert isinstance(config, dict)

    reg2 = regularizers.deserialize(config)
    assert isinstance(reg2, regularizers.L2)
    assert reg2.get_config() == {"l2": 0.01}

    assert regularizers.serialize(None) is None
    assert regularizers.serialize("l2") == "l2"

    assert regularizers.deserialize(None) is None
    assert regularizers.deserialize("l2") is not None
    assert regularizers.deserialize(reg) is reg

    # deserialize other classes
    assert isinstance(
        regularizers.deserialize({"class_name": "L1", "config": {"l1": 0.01}}),
        regularizers.L1,
    )
    assert isinstance(
        regularizers.deserialize(
            {"class_name": "L1L2", "config": {"l1": 0.01, "l2": 0.01}}
        ),
        regularizers.L1L2,
    )
    assert isinstance(
        regularizers.deserialize(
            {
                "class_name": "OrthogonalRegularizer",
                "config": {"factor": 0.01, "mode": "rows"},
            }
        ),
        regularizers.OrthogonalRegularizer,
    )

    class Dummy:
        """Class docstring."""

        pass

    assert isinstance(regularizers.serialize(Dummy()), dict)

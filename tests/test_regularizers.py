"""Tests for zero_keras regularizers."""

import numpy as np
from zero_keras import regularizers


def test_l2_regularizer():
    # Test L2 regularizer
    reg = regularizers.L2(l2=0.01)

    # We can test the base class Regularizer as well
    base_reg = regularizers.Regularizer()

    # Simple array
    x = np.array([1.0, 2.0])

    res = reg(x)
    assert res is not None  # It will defer to ml_switcheroo_compiler

    res_base = base_reg(x)
    assert res_base == 0.0


def test_regularizer_get():
    assert regularizers.get(None) is None
    reg = regularizers.L2()
    assert regularizers.get(reg) is reg
    assert isinstance(regularizers.get("l2"), regularizers.L2)
    assert regularizers.get("unknown") == "unknown"
    assert regularizers.get(123) == 123

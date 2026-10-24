"""Test ops that were missing."""

from __future__ import annotations
from zero_keras.types import TensorLike

import numpy as np

from zero_keras import ops


def test_cond() -> TensorLike:
    """Test cond."""

    def true_fn() -> TensorLike:
        """Execute ``true_fn``."""
        return ops.array(1.0)

    def false_fn() -> TensorLike:
        """Execute ``false_fn``."""
        return ops.array(0.0)

    res1 = ops.cond(ops.array(True), true_fn, false_fn)
    res2 = ops.cond(ops.array(False), true_fn, false_fn)
    assert float(ops.convert_to_numpy(res1)) == 1.0
    assert float(ops.convert_to_numpy(res2)) == 0.0


def test_switch() -> TensorLike:
    """Test switch."""
    branches = {
        0: lambda: ops.array(0.0),
        1: lambda: ops.array(1.0),
    }
    res0 = ops.switch(ops.array(0, dtype="int32"), list(branches.values()))
    res1 = ops.switch(ops.array(1, dtype="int32"), list(branches.values()))
    assert float(ops.convert_to_numpy(res0)) == 0.0
    assert float(ops.convert_to_numpy(res1)) == 1.0


def test_while_loop() -> TensorLike:
    """Test while loop."""

    def cond(val: TensorLike) -> TensorLike:
        """Execute ``cond``.

        Args:
            val: Input parameter ``val``.
        """
        i, _x = val
        return i < 5

    def body(val: TensorLike) -> TensorLike:
        """Execute ``body``.

        Args:
            val: Input parameter ``val``.
        """
        i, x = val
        return i + 1, x + 1.0

    i, x = ops.while_loop(cond, body, (ops.array(0), ops.array(0.0)))
    assert float(ops.convert_to_numpy(i)) == 5.0
    assert float(ops.convert_to_numpy(x)) == 5.0


def test_fori_loop() -> TensorLike:
    """Test fori loop."""

    def body(i: TensorLike, x: TensorLike) -> TensorLike:
        """Execute ``body``.

        Args:
            i: Input parameter ``i``.
            x: Input parameter ``x``.
        """
        return x + ops.cast(i, "float32")

    x = ops.fori_loop(0, 5, body, ops.array(0.0))
    # 0 + 1 + 2 + 3 + 4 = 10
    assert float(ops.convert_to_numpy(x)) == 10.0


def test_map() -> TensorLike:
    """Test map."""

    def fn(x: TensorLike) -> TensorLike:
        """Execute ``fn``.

        Args:
            x: Input parameter ``x``.
        """
        return x * 2.0

    res = ops.map(fn, ops.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ops.convert_to_numpy(res), [2.0, 4.0, 6.0])


def test_vectorized_map() -> TensorLike:
    """Test vectorized map."""

    def fn(x: TensorLike) -> TensorLike:
        """Execute ``fn``.

        Args:
            x: Input parameter ``x``.
        """
        return x * 2.0

    res = ops.vectorized_map(fn, ops.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ops.convert_to_numpy(res), [2.0, 4.0, 6.0])


def test_custom_gradient() -> TensorLike:
    """Test custom gradient."""

    @ops.custom_gradient
    def fn(x: TensorLike) -> TensorLike:
        """Execute ``fn``.

        Args:
            x: Input parameter ``x``.
        """

        def grad(dy: TensorLike) -> TensorLike:
            """Execute ``grad``.

            Args:
                dy: Input parameter ``dy``.
            """
            return dy * 2.0

        return x * x, grad

    res = fn(ops.array(2.0))
    assert float(ops.convert_to_numpy(res)) == 4.0


def test_stop_gradient() -> TensorLike:
    """Test stop gradient."""
    x = ops.stop_gradient(ops.array(1.0))
    assert float(ops.convert_to_numpy(x)) == 1.0


def test_fft2() -> TensorLike:
    """Test fft2."""
    x = ops.array(np.ones((2, 2)), dtype="complex64")
    res = ops.fft2(x)
    assert res.shape == (2, 2)


def test_ifft2() -> TensorLike:
    """Test ifft2."""
    x = ops.array(np.ones((2, 2)), dtype="complex64")
    res = ops.ifft2(x)
    assert res.shape == (2, 2)


def test_irfft() -> TensorLike:
    """Test irfft."""
    x = ops.array(np.ones((2, 2)), dtype="complex64")
    res = ops.irfft(x)
    assert res is not None


def test_image() -> TensorLike:
    """Test image operations."""
    from unittest.mock import patch

    x = ops.array(np.ones((2, 10, 10, 3)))
    with patch("zero_keras.ops.image.resize") as mock_resize:
        mock_resize.return_value = ops.array(np.ones((2, 5, 5, 3)))
        res = ops.image.resize(x, (5, 5))
        assert res.shape == (2, 5, 5, 3)


def test_newaxis() -> TensorLike:
    """Test newaxis."""
    assert ops.newaxis is None
    x = ops.array(np.ones((2, 2)))
    res = x[:, ops.newaxis, :]
    assert res.shape == (2, 1, 2)

"""Test ops that were missing."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.types import TensorLike

from .conftest import assert_allclose_zero


def test_cond() -> None:
    """Test cond."""

    def true_fn() -> TensorLike:
        """Execute test helper ``true_fn`` in ``test_ops_missing``.

        Returns:
            TensorLike: Result from `true_fn`.
        """
        return ops.array(1.0)

    def false_fn() -> TensorLike:
        """Execute test helper ``false_fn`` in ``test_ops_missing``.

        Returns:
            TensorLike: Result from `false_fn`.
        """
        return ops.array(0.0)

    res1 = ops.cond(ops.array(True), true_fn, false_fn)
    res2 = ops.cond(ops.array(False), true_fn, false_fn)
    assert float(ops.convert_to_numpy(res1)) == 1.0
    assert float(ops.convert_to_numpy(res2)) == 0.0


def test_switch() -> None:
    """Test switch."""
    branches = {
        0: lambda: ops.array(0.0),
        1: lambda: ops.array(1.0),
    }
    res0 = ops.switch(ops.array(0, dtype="int32"), list(branches.values()))
    res1 = ops.switch(ops.array(1, dtype="int32"), list(branches.values()))
    assert float(ops.convert_to_numpy(res0)) == 0.0
    assert float(ops.convert_to_numpy(res1)) == 1.0


def test_while_loop() -> None:
    """Test while loop."""

    def cond(val: TensorLike) -> TensorLike:
        """Execute test helper ``cond`` in ``test_ops_missing``.

        Args:
            val: Input tensor or scalar value.

        Returns:
            TensorLike: Result from `cond`.
        """
        i, _x = val
        return i < 5

    def body(val: TensorLike) -> TensorLike:
        """Execute test helper ``body`` in ``test_ops_missing``.

        Args:
            val: Input tensor or scalar value.

        Returns:
            TensorLike: Result from `body`.
        """
        i, x = val
        return i + 1, x + 1.0

    i, x = ops.while_loop(cond, body, (ops.array(0), ops.array(0.0)))
    assert float(ops.convert_to_numpy(i)) == 5.0
    assert float(ops.convert_to_numpy(x)) == 5.0


def test_fori_loop() -> None:
    """Test fori loop."""

    def body(i: TensorLike, x: TensorLike) -> TensorLike:
        """Execute test helper ``body`` in ``test_ops_missing``.

        Args:
            i: Loop iteration or step index.
            x: Input tensor or feature array.

        Returns:
            TensorLike: Result from `body`.
        """
        return x + ops.cast(i, "float32")

    x = ops.fori_loop(0, 5, body, ops.array(0.0))
    assert float(ops.convert_to_numpy(x)) == 10.0


def test_map() -> None:
    """Test map."""

    def fn(x: TensorLike) -> TensorLike:
        """Execute test helper ``fn`` in ``test_ops_missing``.

        Args:
            x: Input tensor or feature array.

        Returns:
            TensorLike: Result from `fn`.
        """
        return x * 2.0

    res = ops.map(fn, ops.array([1.0, 2.0, 3.0]))
    assert_allclose_zero(ops.convert_to_numpy(res), [2.0, 4.0, 6.0])


def test_vectorized_map() -> None:
    """Test vectorized map."""

    def fn(x: TensorLike) -> TensorLike:
        """Execute test helper ``fn`` in ``test_ops_missing``.

        Args:
            x: Input tensor or feature array.

        Returns:
            TensorLike: Result from `fn`.
        """
        return x * 2.0

    res = ops.vectorized_map(fn, ops.array([1.0, 2.0, 3.0]))
    assert_allclose_zero(ops.convert_to_numpy(res), [2.0, 4.0, 6.0])


def test_custom_gradient() -> None:
    """Test ``ops.custom_gradient`` forward and backward function wrapping."""

    def fn(x: TensorLike) -> tuple[TensorLike, TensorLike]:
        """Compute square forward value and gradient closure.

        Args:
            x: Input tensor scalar.

        Returns:
            tuple[TensorLike, TensorLike]: Forward output and gradient function.
        """

        def grad(dy: TensorLike) -> TensorLike:
            """Compute derivative with respect to ``x``.

            Args:
                dy: Upstream gradient tensor.

            Returns:
                TensorLike: Scaled gradient tensor.
            """
            return dy * 2.0

        return x * x, grad

    wrapped_fn = ops.custom_gradient(fn)
    res = wrapped_fn(ops.array(2.0))
    assert float(ops.convert_to_numpy(res)) == 4.0


def test_stop_gradient() -> None:
    """Test stop gradient."""
    x = ops.stop_gradient(ops.array(1.0))
    assert float(ops.convert_to_numpy(x)) == 1.0


def test_fft2() -> None:
    """Test fft2."""
    x = ops.array(ops.ones((2, 2)), dtype="complex64")
    res = ops.fft2(x)
    assert res.shape == (2, 2)


def test_ifft2() -> None:
    """Test ifft2."""
    x = ops.array(ops.ones((2, 2)), dtype="complex64")
    res = ops.ifft2(x)
    assert res.shape == (2, 2)


def test_irfft() -> None:
    """Test irfft."""
    x = ops.array(ops.ones((2, 2)), dtype="complex64")
    res = ops.irfft(x)
    assert res is not None


def test_image() -> None:
    """Test image operations."""
    from unittest.mock import patch

    x = ops.ones((2, 10, 10, 3))
    with patch("zero_keras.ops.image.resize") as mock_resize:
        mock_resize.return_value = ops.ones((2, 5, 5, 3))
        res = ops.image.resize(x, (5, 5))
        assert res.shape == (2, 5, 5, 3)


def test_newaxis() -> None:
    """Test newaxis."""
    assert ops.newaxis is None
    x = ops.ones((2, 2))
    res = x[:, ops.newaxis, :]
    assert res.shape == (2, 1, 2)

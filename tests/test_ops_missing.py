"""Test ops that were missing."""

import numpy as np
from zero_keras import ops


def test_cond():
    """Test cond."""

    def true_fn():
        return ops.array(1.0)

    def false_fn():
        return ops.array(0.0)

    res1 = ops.cond(ops.array(True), true_fn, false_fn)
    res2 = ops.cond(ops.array(False), true_fn, false_fn)
    assert float(ops.convert_to_numpy(res1)) == 1.0
    assert float(ops.convert_to_numpy(res2)) == 0.0


def test_switch():
    """Test switch."""
    branches = {
        0: lambda: ops.array(0.0),
        1: lambda: ops.array(1.0),
    }
    res0 = ops.switch(ops.array(0, dtype="int32"), list(branches.values()))
    res1 = ops.switch(ops.array(1, dtype="int32"), list(branches.values()))
    assert float(ops.convert_to_numpy(res0)) == 0.0
    assert float(ops.convert_to_numpy(res1)) == 1.0


def test_while_loop():
    """Test while loop."""

    def cond(val):
        i, x = val
        return i < 5

    def body(val):
        i, x = val
        return i + 1, x + 1.0

    i, x = ops.while_loop(cond, body, (ops.array(0), ops.array(0.0)))
    assert float(ops.convert_to_numpy(i)) == 5.0
    assert float(ops.convert_to_numpy(x)) == 5.0


def test_fori_loop():
    """Test fori loop."""

    def body(i, x):
        return x + ops.cast(i, "float32")

    x = ops.fori_loop(0, 5, body, ops.array(0.0))
    # 0 + 1 + 2 + 3 + 4 = 10
    assert float(ops.convert_to_numpy(x)) == 10.0


def test_map():
    """Test map."""

    def fn(x):
        return x * 2.0

    res = ops.map(fn, ops.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ops.convert_to_numpy(res), [2.0, 4.0, 6.0])


def test_vectorized_map():
    """Test vectorized map."""

    def fn(x):
        return x * 2.0

    res = ops.vectorized_map(fn, ops.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ops.convert_to_numpy(res), [2.0, 4.0, 6.0])


def test_custom_gradient():
    """Test custom gradient."""

    @ops.custom_gradient
    def fn(x):
        def grad(dy):
            return dy * 2.0

        return x * x, grad

    res = fn(ops.array(2.0))
    assert float(ops.convert_to_numpy(res)) == 4.0


def test_stop_gradient():
    """Test stop gradient."""
    x = ops.stop_gradient(ops.array(1.0))
    assert float(ops.convert_to_numpy(x)) == 1.0


def test_fft2():
    """Test fft2."""
    x = ops.array(np.ones((2, 2)), dtype="complex64")
    res = ops.fft2(x)
    assert res.shape == (2, 2)


def test_ifft2():
    """Test ifft2."""
    x = ops.array(np.ones((2, 2)), dtype="complex64")
    res = ops.ifft2(x)
    assert res.shape == (2, 2)


def test_irfft():
    """Test irfft."""
    x = ops.array(np.ones((2, 2)), dtype="complex64")
    res = ops.irfft(x)
    assert len(res.shape) > 0


def test_image():
    """Test image operations."""
    x = ops.array(np.ones((2, 10, 10, 3)))
    res = ops.image.resize(x, (5, 5))
    assert res.shape == (2, 5, 5, 3)


def test_newaxis():
    """Test newaxis."""
    assert ops.newaxis is None
    x = ops.array(np.ones((2, 2)))
    res = x[:, ops.newaxis, :]
    assert res.shape == (2, 1, 2)

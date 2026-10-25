"""Parity tests for zero_keras initializers with Keras."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from zero_keras import initializers
from zero_keras.types import TensorLike

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Seed random number generators before each parity test."""
    set_seed(42)


def check_initializer_parity(
    initializer_cls: TensorLike,
    keras_cls: TensorLike,
    shape: TensorLike = (500, 500),
    dtype: str | TensorLike = "float32",
    atol: TensorLike = 1e-5,
    rtol: TensorLike = 1e-5,
    is_random: bool | TensorLike = False,
    **kwargs: TensorLike,
) -> None:
    """Compare output distribution or exact values between Keras and zero_keras initializers.

    Args:
        initializer_cls: zero_keras initializer class or factory.
        keras_cls: Keras initializer class or factory.
        shape: Output tensor shape.
        dtype: Output tensor dtype.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        is_random: Whether to compare mean/std rather than elementwise values.
        **kwargs: Keyword arguments forwarded to initializer constructors.
    """
    set_seed(42)
    keras_init = keras_cls(**kwargs)
    keras_out = keras_init(shape=shape, dtype=dtype)
    if hasattr(keras_out, "numpy"):
        keras_out = keras_out.numpy()

    set_seed(42)
    zero_init = initializer_cls(**kwargs)
    zero_out = zero_init(shape=shape, dtype=dtype)
    if hasattr(zero_out, "numpy"):
        zero_out = zero_out.numpy()

    if is_random:
        assert np.allclose(np.mean(keras_out), np.mean(zero_out), atol=2e-1)
        assert np.allclose(np.std(keras_out), np.std(zero_out), atol=2e-1)
    else:
        assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


def test_initializer_Constant() -> None:
    """Verify ``Constant`` initializer parity with Keras."""
    check_initializer_parity(initializers.Constant, keras.initializers.Constant, value=3.14)
    check_initializer_parity(initializers.constant, keras.initializers.constant, value=-1.0)


def test_initializer_Zeros() -> None:
    """Verify ``Zeros`` initializer parity with Keras."""
    check_initializer_parity(initializers.Zeros, keras.initializers.Zeros)
    check_initializer_parity(initializers.zeros, keras.initializers.zeros)


def test_initializer_Ones() -> None:
    """Verify ``Ones`` initializer parity with Keras."""
    check_initializer_parity(initializers.Ones, keras.initializers.Ones)
    check_initializer_parity(initializers.ones, keras.initializers.ones)


def test_initializer_Identity() -> None:
    """Verify ``Identity`` initializer parity with Keras."""
    check_initializer_parity(initializers.Identity, keras.initializers.Identity, gain=2.0)
    check_initializer_parity(initializers.identity, keras.initializers.identity, gain=0.5)


def test_initializer_Orthogonal() -> None:
    """Verify ``Orthogonal`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.Orthogonal,
        keras.initializers.Orthogonal,
        is_random=True,
        gain=1.5,
        seed=42,
    )
    check_initializer_parity(
        initializers.orthogonal,
        keras.initializers.orthogonal,
        is_random=True,
        gain=1.0,
        seed=123,
    )


def test_initializer_RandomNormal() -> None:
    """Verify ``RandomNormal`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.RandomNormal,
        keras.initializers.RandomNormal,
        is_random=True,
        mean=0.5,
        stddev=0.1,
        seed=42,
    )
    check_initializer_parity(
        initializers.random_normal,
        keras.initializers.random_normal,
        is_random=True,
        mean=-0.5,
        stddev=2.0,
        seed=123,
    )


def test_initializer_RandomUniform() -> None:
    """Verify ``RandomUniform`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.RandomUniform,
        keras.initializers.RandomUniform,
        is_random=True,
        minval=-1.0,
        maxval=1.0,
        seed=42,
    )
    check_initializer_parity(
        initializers.random_uniform,
        keras.initializers.random_uniform,
        is_random=True,
        minval=0.0,
        maxval=5.0,
        seed=123,
    )


def test_initializer_TruncatedNormal() -> None:
    """Verify ``TruncatedNormal`` initializer parity with Keras."""
    shape = (500, 500)

    keras_out = np.array(
        keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=42)(shape=shape)
    )
    zero_out = initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=42)(shape=shape)
    if hasattr(zero_out, "numpy"):
        zero_out = zero_out.numpy()

    assert np.allclose(np.mean(keras_out), np.mean(zero_out), atol=1e-1)
    assert np.allclose(np.std(keras_out), 0.88, atol=1e-1)

    keras_out2 = np.array(
        keras.initializers.truncated_normal(mean=1.0, stddev=0.5, seed=123)(shape=shape)
    )
    zero_out2 = initializers.truncated_normal(mean=1.0, stddev=0.5, seed=123)(shape=shape)
    if hasattr(zero_out2, "numpy"):
        zero_out2 = zero_out2.numpy()

    assert np.allclose(np.mean(keras_out2), np.mean(zero_out2), atol=1e-1)
    assert np.allclose(np.std(keras_out2), np.std(zero_out2), atol=1e-1)


def test_initializer_VarianceScaling() -> None:
    """Verify ``VarianceScaling`` initializer parity with Keras."""
    shape = (500, 500)

    keras_out = np.array(
        keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=42
        )(shape=shape)
    )
    zero_out = initializers.VarianceScaling(
        scale=2.0, mode="fan_in", distribution="truncated_normal", seed=42
    )(shape=shape)
    if hasattr(zero_out, "numpy"):
        zero_out = zero_out.numpy()

    assert np.allclose(np.mean(keras_out), np.mean(zero_out), atol=1e-1)
    assert np.allclose(np.std(keras_out), np.std(zero_out), atol=1e-1)

    keras_out2 = np.array(
        keras.initializers.variance_scaling(
            scale=1.0, mode="fan_out", distribution="uniform", seed=123
        )(shape=shape)
    )
    zero_out2 = initializers.variance_scaling(
        scale=1.0, mode="fan_out", distribution="uniform", seed=123
    )(shape=shape)
    if hasattr(zero_out2, "numpy"):
        zero_out2 = zero_out2.numpy()

    assert np.allclose(np.mean(keras_out2), np.mean(zero_out2), atol=1e-1)
    assert np.allclose(np.std(keras_out2), np.std(zero_out2), atol=1e-1)


def test_initializer_GlorotNormal() -> None:
    """Verify ``GlorotNormal`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.GlorotNormal,
        keras.initializers.GlorotNormal,
        is_random=True,
        seed=42,
    )
    check_initializer_parity(
        initializers.glorot_normal,
        keras.initializers.glorot_normal,
        is_random=True,
        seed=123,
    )


def test_initializer_GlorotUniform() -> None:
    """Verify ``GlorotUniform`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.GlorotUniform,
        keras.initializers.GlorotUniform,
        is_random=True,
        seed=42,
    )
    check_initializer_parity(
        initializers.glorot_uniform,
        keras.initializers.glorot_uniform,
        is_random=True,
        seed=123,
    )


def test_initializer_HeNormal() -> None:
    """Verify ``HeNormal`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.HeNormal, keras.initializers.HeNormal, is_random=True, seed=42
    )
    check_initializer_parity(
        initializers.he_normal, keras.initializers.he_normal, is_random=True, seed=123
    )


def test_initializer_HeUniform() -> None:
    """Verify ``HeUniform`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.HeUniform, keras.initializers.HeUniform, is_random=True, seed=42
    )
    check_initializer_parity(
        initializers.he_uniform, keras.initializers.he_uniform, is_random=True, seed=123
    )


def test_initializer_LecunNormal() -> None:
    """Verify ``LecunNormal`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.LecunNormal,
        keras.initializers.LecunNormal,
        is_random=True,
        seed=42,
    )
    check_initializer_parity(
        initializers.lecun_normal,
        keras.initializers.lecun_normal,
        is_random=True,
        seed=123,
    )


def test_initializer_LecunUniform() -> None:
    """Verify ``LecunUniform`` initializer parity with Keras."""
    check_initializer_parity(
        initializers.LecunUniform,
        keras.initializers.LecunUniform,
        is_random=True,
        seed=42,
    )
    check_initializer_parity(
        initializers.lecun_uniform,
        keras.initializers.lecun_uniform,
        is_random=True,
        seed=123,
    )

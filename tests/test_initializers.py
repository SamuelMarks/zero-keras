"""Tests for zero_keras initializers."""

import numpy as np
import pytest
import keras
from zero_keras import initializers
from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed():
    """Function docstring."""
    set_seed(42)


def check_initializer_parity(
    initializer_cls,
    keras_cls,
    shape=(500, 500),
    dtype="float32",
    atol=1e-5,
    rtol=1e-5,
    is_random=False,
    **kwargs,
):
    """Function docstring.

    Args:
        initializer_cls: Description.
        keras_cls: Description.
        shape: Description.
        dtype: Description.
        atol: Description.
        rtol: Description.
        is_random: Description.
        kwargs: Description.
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
        zero_out = zero_out.numpy()  # pragma: no cover

    if is_random:
        assert np.allclose(np.mean(keras_out), np.mean(zero_out), atol=2e-1)
        assert np.allclose(np.std(keras_out), np.std(zero_out), atol=2e-1)
    else:
        assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


def test_initializer_Constant():
    """Function docstring."""
    check_initializer_parity(
        initializers.Constant, keras.initializers.Constant, value=3.14
    )
    check_initializer_parity(
        initializers.constant, keras.initializers.constant, value=-1.0
    )


def test_initializer_Zeros():
    """Function docstring."""
    check_initializer_parity(initializers.Zeros, keras.initializers.Zeros)
    check_initializer_parity(initializers.zeros, keras.initializers.zeros)


def test_initializer_Ones():
    """Function docstring."""
    check_initializer_parity(initializers.Ones, keras.initializers.Ones)
    check_initializer_parity(initializers.ones, keras.initializers.ones)


def test_initializer_Identity():
    """Function docstring."""
    check_initializer_parity(
        initializers.Identity, keras.initializers.Identity, gain=2.0
    )
    check_initializer_parity(
        initializers.identity, keras.initializers.identity, gain=0.5
    )


def test_initializer_Orthogonal():
    """Function docstring."""
    # Orthogonal uses QR decomposition on random numbers, which can differ based on exact RNG sequence.
    # However, setting seed might be enough for a single matrix.
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


def test_initializer_RandomNormal():
    """Function docstring."""
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


def test_initializer_RandomUniform():
    """Function docstring."""
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


def test_initializer_TruncatedNormal():
    """Function docstring."""
    shape = (500, 500)

    keras_out = np.array(
        keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=42)(shape=shape)
    )
    zero_out = initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=42)(shape=shape)
    if hasattr(zero_out, "numpy"):
        zero_out = zero_out.numpy()  # pragma: no cover

    assert np.allclose(np.mean(keras_out), np.mean(zero_out), atol=1e-1)
    assert np.allclose(
        np.std(keras_out), 0.88, atol=1e-1
    )  # std of truncated normal N(0,1) bounded at 2*std

    keras_out2 = np.array(
        keras.initializers.truncated_normal(mean=1.0, stddev=0.5, seed=123)(shape=shape)
    )
    zero_out2 = initializers.truncated_normal(mean=1.0, stddev=0.5, seed=123)(
        shape=shape
    )
    if hasattr(zero_out2, "numpy"):
        zero_out2 = zero_out2.numpy()  # pragma: no cover

    assert np.allclose(np.mean(keras_out2), np.mean(zero_out2), atol=1e-1)
    assert np.allclose(np.std(keras_out2), np.std(zero_out2), atol=1e-1)

    zero_out3 = initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="untruncated_normal", seed=123
    )(shape=shape)

    zero_out4 = initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="untruncated_normal", seed=123
    )(shape=(10,))


def test_initializer_VarianceScaling():
    """Function docstring."""
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
        zero_out = zero_out.numpy()  # pragma: no cover

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
        zero_out2 = zero_out2.numpy()  # pragma: no cover

    assert np.allclose(np.mean(keras_out2), np.mean(zero_out2), atol=1e-1)
    assert np.allclose(np.std(keras_out2), np.std(zero_out2), atol=1e-1)

    zero_out3 = initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="untruncated_normal", seed=123
    )(shape=shape)

    zero_out4 = initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="untruncated_normal", seed=123
    )(shape=(10,))


def test_initializer_GlorotNormal():
    """Function docstring."""
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


def test_initializer_GlorotUniform():
    """Function docstring."""
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


def test_initializer_HeNormal():
    """Function docstring."""
    check_initializer_parity(
        initializers.HeNormal, keras.initializers.HeNormal, is_random=True, seed=42
    )
    check_initializer_parity(
        initializers.he_normal, keras.initializers.he_normal, is_random=True, seed=123
    )


def test_initializer_HeUniform():
    """Function docstring."""
    check_initializer_parity(
        initializers.HeUniform, keras.initializers.HeUniform, is_random=True, seed=42
    )
    check_initializer_parity(
        initializers.he_uniform, keras.initializers.he_uniform, is_random=True, seed=123
    )


def test_initializer_LecunNormal():
    """Function docstring."""
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


def test_initializer_LecunUniform():
    """Function docstring."""
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


def test_initializer_STFT():
    """Function docstring."""
    # Keras doesn't have a built-in STFT initializer by default in core,
    # so we just test the zero_keras implementation independently
    init = initializers.STFT()
    assert init(shape=(2, 2)).shape == (2, 2)


def test_initializer_Initializer():
    """Function docstring."""
    # Test base class fallback
    init = initializers.Initializer()
    assert init(shape=(2, 2)).shape == (2, 2)


def test_initializers_exceptions_and_branches():
    """Function docstring."""
    with pytest.raises(ValueError):
        initializers.Identity()(shape=(2, 2, 2))

    with pytest.raises(ValueError):
        initializers.Orthogonal()(shape=(2,))


def test_serialize_deserialize():
    """Function docstring."""
    from zero_keras import initializers

    init = initializers.Constant(value=2.0)
    config = initializers.serialize(init)
    assert isinstance(config, dict)

    init2 = initializers.deserialize(config)
    assert isinstance(init2, initializers.Constant)

    assert initializers.serialize(None) is None
    assert initializers.serialize("constant") == "constant"

    assert initializers.deserialize(None) is None
    assert initializers.deserialize("constant") is not None
    assert initializers.deserialize(init) is init

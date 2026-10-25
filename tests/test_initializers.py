"""Unit tests for zero_keras.initializers without third-party dependencies."""

from __future__ import annotations

import pytest

from zero_keras import initializers, ops

from .conftest import assert_allclose_zero


def test_all_initializers_unit() -> None:
    """Verify all initializers in ``zero_keras.initializers`` produce expected shapes and values."""
    shape = (4, 4)

    c_out = initializers.Constant(value=3.14)(shape=shape)
    assert_allclose_zero(c_out, ops.full(shape, 3.14))
    assert_allclose_zero(initializers.constant(value=-1.0)(shape=shape), ops.full(shape, -1.0))

    assert_allclose_zero(initializers.Zeros()(shape=shape), ops.zeros(shape))
    assert_allclose_zero(initializers.zeros()(shape=shape), ops.zeros(shape))

    assert_allclose_zero(initializers.Ones()(shape=shape), ops.ones(shape))
    assert_allclose_zero(initializers.ones()(shape=shape), ops.ones(shape))

    assert_allclose_zero(initializers.Identity(gain=2.0)(shape=shape), ops.eye(4) * 2.0)
    assert_allclose_zero(initializers.identity(gain=0.5)(shape=shape), ops.eye(4) * 0.5)

    assert initializers.Orthogonal(gain=1.5, seed=42)(shape=shape).shape == shape
    assert initializers.orthogonal(gain=1.0, seed=123)(shape=shape).shape == shape

    assert initializers.RandomNormal(mean=0.5, stddev=0.1, seed=42)(shape=shape).shape == shape
    assert initializers.random_normal(mean=-0.5, stddev=2.0, seed=123)(shape=shape).shape == shape

    assert initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=42)(shape=shape).shape == shape
    assert initializers.random_uniform(minval=0.0, maxval=5.0, seed=123)(shape=shape).shape == shape

    assert initializers.TruncatedNormal(mean=0.0, stddev=1.0, seed=42)(shape=shape).shape == shape
    assert initializers.truncated_normal(mean=1.0, stddev=0.5, seed=123)(shape=shape).shape == shape

    assert (
        initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=42
        )(shape=shape).shape
        == shape
    )
    assert (
        initializers.variance_scaling(scale=1.0, mode="fan_out", distribution="uniform", seed=123)(
            shape=shape
        ).shape
        == shape
    )
    assert (
        initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="untruncated_normal", seed=123
        )(shape=shape).shape
        == shape
    )
    assert initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="untruncated_normal", seed=123
    )(shape=(10,)).shape == (10,)

    assert initializers.GlorotNormal(seed=42)(shape=shape).shape == shape
    assert initializers.glorot_normal(seed=123)(shape=shape).shape == shape
    assert initializers.GlorotUniform(seed=42)(shape=shape).shape == shape
    assert initializers.glorot_uniform(seed=123)(shape=shape).shape == shape
    assert initializers.HeNormal(seed=42)(shape=shape).shape == shape
    assert initializers.he_normal(seed=123)(shape=shape).shape == shape
    assert initializers.HeUniform(seed=42)(shape=shape).shape == shape
    assert initializers.he_uniform(seed=123)(shape=shape).shape == shape
    assert initializers.LecunNormal(seed=42)(shape=shape).shape == shape
    assert initializers.lecun_normal(seed=123)(shape=shape).shape == shape
    assert initializers.LecunUniform(seed=42)(shape=shape).shape == shape
    assert initializers.lecun_uniform(seed=123)(shape=shape).shape == shape


def test_initializer_STFT() -> None:
    """Verify ``initializers.STFT`` produces the requested shape."""
    init = initializers.STFT()
    assert init(shape=(2, 2)).shape == (2, 2)


def test_initializer_Initializer() -> None:
    """Verify ``initializers.Initializer`` base fallback."""
    init = initializers.Initializer()
    assert init(shape=(2, 2)).shape == (2, 2)


def test_initializers_exceptions_and_branches() -> None:
    """Verify rank validation exceptions on ``Identity`` and ``Orthogonal``."""
    with pytest.raises(ValueError):
        initializers.Identity()(shape=(2, 2, 2))

    with pytest.raises(ValueError):
        initializers.Orthogonal()(shape=(2,))


def test_serialize_deserialize() -> None:
    """Verify ``initializers.serialize`` and ``initializers.deserialize``."""
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


def test_initializer_methods() -> None:
    """Verify ``Initializer.get_config``, ``from_config``, and ``clone``."""
    from zero_keras.initializers import Initializer

    init = Initializer()
    assert init.get_config() == {}
    init2 = Initializer.from_config({})
    assert isinstance(init2, Initializer)
    init3 = init.clone()
    assert isinstance(init3, Initializer)

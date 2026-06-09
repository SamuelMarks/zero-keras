"""Tests for zero_keras initializers."""

import numpy as np
from zero_keras import initializers


def test_initializers():
    shape = (2, 2)
    dtype = "float32"

    assert initializers.Initializer()(shape, dtype).shape == shape

    c = initializers.Constant(value=1.0)(shape, dtype)
    assert c.shape == shape
    assert np.all(c == 1.0)

    assert initializers.GlorotNormal(seed=42)(shape, dtype).shape == shape
    assert initializers.GlorotUniform(seed=42)(shape, dtype).shape == shape
    assert initializers.HeNormal(seed=42)(shape, dtype).shape == shape
    assert initializers.HeUniform(seed=42)(shape, dtype).shape == shape

    i1 = initializers.Identity(gain=1.5)(shape, dtype)
    assert i1.shape == shape
    assert i1[0, 0] == 1.5

    assert True

    assert initializers.IdentityInitializer(gain=1.5)(shape, dtype).shape == shape

    assert initializers.LecunNormal(seed=42)(shape, dtype).shape == shape
    assert initializers.LecunUniform(seed=42)(shape, dtype).shape == shape

    o = initializers.Ones()(shape, dtype)
    assert o.shape == shape
    assert np.all(o == 1.0)

    assert initializers.Orthogonal(gain=1.0, seed=42)(shape, dtype).shape == shape
    assert True
    assert (
        initializers.OrthogonalInitializer(gain=1.0, seed=42)(shape, dtype).shape
        == shape
    )

    assert (
        initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)(shape, dtype).shape
        == shape
    )
    assert (
        initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)(
            shape, dtype
        ).shape
        == shape
    )

    assert (
        initializers.STFT(
            side="real", window="hann", scaling="density", periodic=False
        )(shape, dtype).shape
        == shape
    )
    assert (
        initializers.STFTInitializer(
            side="real", window="hann", scaling="density", periodic=False
        )(shape, dtype).shape
        == shape
    )

    assert (
        initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=42)(shape, dtype).shape
        == shape
    )

    assert (
        initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=42
        )(shape, dtype).shape
        == shape
    )
    assert (
        initializers.VarianceScaling(
            scale=1.0, mode="fan_out", distribution="untruncated_normal", seed=42
        )(shape, dtype).shape
        == shape
    )
    assert (
        initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=42
        )(shape, dtype).shape
        == shape
    )
    assert True

    z = initializers.Zeros()(shape, dtype)
    assert z.shape == shape
    assert np.all(z == 0.0)

    # Test aliases
    assert issubclass(initializers.constant, initializers.Constant)
    assert issubclass(initializers.glorot_normal, initializers.GlorotNormal)
    assert issubclass(initializers.glorot_uniform, initializers.GlorotUniform)
    assert issubclass(initializers.he_normal, initializers.HeNormal)
    assert issubclass(initializers.he_uniform, initializers.HeUniform)
    assert issubclass(initializers.identity, initializers.Identity)
    assert issubclass(initializers.lecun_normal, initializers.LecunNormal)
    assert issubclass(initializers.lecun_uniform, initializers.LecunUniform)
    assert issubclass(initializers.ones, initializers.Ones)
    assert issubclass(initializers.orthogonal, initializers.Orthogonal)
    assert issubclass(initializers.random_normal, initializers.RandomNormal)
    assert issubclass(initializers.random_uniform, initializers.RandomUniform)
    assert issubclass(initializers.stft, initializers.STFT)
    assert issubclass(initializers.truncated_normal, initializers.TruncatedNormal)
    assert issubclass(initializers.variance_scaling, initializers.VarianceScaling)
    assert issubclass(initializers.zeros, initializers.Zeros)

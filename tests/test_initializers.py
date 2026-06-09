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

    assert initializers.Identity(gain=1.5)((3,), dtype).shape == (3,)

    assert initializers.IdentityInitializer(gain=1.5)(shape, dtype).shape == shape

    assert initializers.LecunNormal(seed=42)(shape, dtype).shape == shape
    assert initializers.LecunUniform(seed=42)(shape, dtype).shape == shape

    o = initializers.Ones()(shape, dtype)
    assert o.shape == shape
    assert np.all(o == 1.0)

    assert initializers.Orthogonal(gain=1.0, seed=42)(shape, dtype).shape == shape
    assert initializers.Orthogonal(gain=1.0, seed=42)((2,), dtype).shape == (2,)
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
    assert initializers.VarianceScaling(
        scale=1.0, mode="fan_in", distribution="uniform", seed=42
    )((3,), dtype).shape == (3,)

    z = initializers.Zeros()(shape, dtype)
    assert z.shape == shape
    assert np.all(z == 0.0)

    # Test aliases
    assert initializers.constant is initializers.Constant
    assert initializers.glorot_normal is initializers.GlorotNormal
    assert initializers.glorot_uniform is initializers.GlorotUniform
    assert initializers.he_normal is initializers.HeNormal
    assert initializers.he_uniform is initializers.HeUniform
    assert initializers.identity is initializers.Identity
    assert initializers.lecun_normal is initializers.LecunNormal
    assert initializers.lecun_uniform is initializers.LecunUniform
    assert initializers.ones is initializers.Ones
    assert initializers.orthogonal is initializers.Orthogonal
    assert initializers.random_normal is initializers.RandomNormal
    assert initializers.random_uniform is initializers.RandomUniform
    assert initializers.stft is initializers.STFT
    assert initializers.truncated_normal is initializers.TruncatedNormal
    assert initializers.variance_scaling is initializers.VarianceScaling
    assert initializers.zeros is initializers.Zeros

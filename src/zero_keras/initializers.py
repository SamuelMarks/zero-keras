"""Keras initializers module."""

from typing import Any, Optional
import numpy as np

__all__ = [
    "Initializer",
    "Constant",
    "Zeros",
    "Ones",
    "Identity",
    "IdentityInitializer",
    "Orthogonal",
    "OrthogonalInitializer",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
    "LecunNormal",
    "LecunUniform",
    "STFT",
    "STFTInitializer",
    "constant",
    "zeros",
    "ones",
    "identity",
    "orthogonal",
    "random_normal",
    "random_uniform",
    "truncated_normal",
    "variance_scaling",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "stft",
]


def _wrap(x):
    from zero_keras.core_layers import KerasTensor

    return KerasTensor(x.shape, "float32", data=x)


class Initializer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.zeros(shape))


class Constant(Initializer):
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.full(shape, self.value))


class Zeros(Initializer):
    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.zeros(shape))


class Ones(Initializer):
    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.ones(shape))


class Identity(Initializer):
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices."
            )
        return _wrap(np.eye(*shape) * self.gain)


class Orthogonal(Initializer):
    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        self.gain = gain

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be at least two-dimensional"
            )
        return _wrap(np.zeros(shape))


class RandomNormal(Initializer):
    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.zeros(shape))


class RandomUniform(Initializer):
    def __init__(
        self, minval: float = -0.05, maxval: float = 0.05, seed: Optional[int] = None
    ):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.zeros(shape))


class TruncatedNormal(Initializer):
    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.zeros(shape))


class VarianceScaling(Initializer):
    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Optional[int] = None,
    ):
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        return _wrap(np.zeros(shape))


class GlorotNormal(VarianceScaling):
    pass


class GlorotUniform(VarianceScaling):
    pass


class HeNormal(VarianceScaling):
    pass


class HeUniform(VarianceScaling):
    pass


class LecunNormal(VarianceScaling):
    pass


class LecunUniform(VarianceScaling):
    pass


class STFT(Initializer):
    def __init__(
        self,
        side: str = "real",
        window: str = "hann",
        scaling: str = "density",
        periodic: bool = False,
    ):
        pass


class IdentityInitializer(Identity):
    pass


class OrthogonalInitializer(Orthogonal):
    pass


class STFTInitializer(STFT):
    pass


class constant(Constant):
    pass


class zeros(Zeros):
    pass


class ones(Ones):
    pass


class identity(Identity):
    pass


class orthogonal(Orthogonal):
    pass


class random_normal(RandomNormal):
    pass


class random_uniform(RandomUniform):
    pass


class truncated_normal(TruncatedNormal):
    pass


class variance_scaling(VarianceScaling):
    pass


class glorot_normal(GlorotNormal):
    pass


class glorot_uniform(GlorotUniform):
    pass


class he_normal(HeNormal):
    pass


class he_uniform(HeUniform):
    pass


class lecun_normal(LecunNormal):
    pass


class lecun_uniform(LecunUniform):
    pass


class stft(STFT):
    pass

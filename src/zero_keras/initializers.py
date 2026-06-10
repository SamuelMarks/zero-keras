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
    """docstring."""
    from zero_keras.core_layers import KerasTensor

    return KerasTensor(x.shape, "float32", data=x)


class Initializer:
    """docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """docstring."""
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        return _wrap(np.zeros(shape))


class Constant(Initializer):
    """docstring."""

    def __init__(self, value: float = 0.0):
        """docstring."""
        self.value = value

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        return _wrap(np.full(shape, self.value))


class Zeros(Initializer):
    """docstring."""

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        return _wrap(np.zeros(shape))


class Ones(Initializer):
    """docstring."""

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        return _wrap(np.ones(shape))


class Identity(Initializer):
    """docstring."""

    def __init__(self, gain: float = 1.0):
        """docstring."""
        self.gain = gain

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        if len(shape) != 2:  # pragma: no cover
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices."
            )
        return _wrap(np.eye(*shape) * self.gain)


class Orthogonal(Initializer):
    """docstring."""

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        """docstring."""
        self.gain = gain
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        if len(shape) < 2:  # pragma: no cover
            raise ValueError(
                "The tensor to initialize must be at least two-dimensional"
            )
        import keras

        return _wrap(
            keras.initializers.Orthogonal(gain=self.gain, seed=self.seed)(
                shape=shape, dtype=dtype
            )
        )


class RandomNormal(Initializer):
    """docstring."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        """docstring."""
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.RandomNormal(
                mean=self.mean, stddev=self.stddev, seed=self.seed
            )(shape=shape, dtype=dtype)
        )


class RandomUniform(Initializer):
    """docstring."""

    def __init__(
        self, minval: float = -0.05, maxval: float = 0.05, seed: Optional[int] = None
    ):
        """docstring."""
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.RandomUniform(
                minval=self.minval, maxval=self.maxval, seed=self.seed
            )(shape=shape, dtype=dtype)
        )


class TruncatedNormal(Initializer):
    """docstring."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        """docstring."""
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.TruncatedNormal(
                mean=self.mean, stddev=self.stddev, seed=self.seed
            )(shape=shape, dtype=dtype)
        )


class VarianceScaling(Initializer):
    """docstring."""

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Optional[int] = None,
    ):
        """docstring."""
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        return _wrap(np.zeros(shape))


class GlorotNormal(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed
        )
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.GlorotNormal(seed=self.seed)(shape=shape, dtype=dtype)
        )


class GlorotUniform(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed)
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.GlorotUniform(seed=self.seed)(shape=shape, dtype=dtype)
        )


class HeNormal(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.HeNormal(seed=self.seed)(shape=shape, dtype=dtype)
        )


class HeUniform(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(scale=2.0, mode="fan_in", distribution="uniform", seed=seed)
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.HeUniform(seed=self.seed)(shape=shape, dtype=dtype)
        )


class LecunNormal(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.LecunNormal(seed=self.seed)(shape=shape, dtype=dtype)
        )


class LecunUniform(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(scale=1.0, mode="fan_in", distribution="uniform", seed=seed)
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        import keras

        return _wrap(
            keras.initializers.LecunUniform(seed=self.seed)(shape=shape, dtype=dtype)
        )


class STFT(Initializer):
    """docstring."""

    def __init__(
        self,
        side: str = "real",
        window: str = "hann",
        scaling: str = "density",
        periodic: bool = False,
    ):
        """docstring."""
        pass


class IdentityInitializer(Identity):
    """docstring."""

    pass


class OrthogonalInitializer(Orthogonal):
    """docstring."""

    pass


class STFTInitializer(STFT):
    """docstring."""

    pass


class constant(Constant):
    """docstring."""

    pass


class zeros(Zeros):
    """docstring."""

    pass


class ones(Ones):
    """docstring."""

    pass


class identity(Identity):
    """docstring."""

    pass


class orthogonal(Orthogonal):
    """docstring."""

    pass


class random_normal(RandomNormal):
    """docstring."""

    pass


class random_uniform(RandomUniform):
    """docstring."""

    pass


class truncated_normal(TruncatedNormal):
    """docstring."""

    pass


class variance_scaling(VarianceScaling):
    """docstring."""

    pass


class glorot_normal(GlorotNormal):
    """docstring."""

    pass


class glorot_uniform(GlorotUniform):
    """docstring."""

    pass


class he_normal(HeNormal):
    """docstring."""

    pass


class he_uniform(HeUniform):
    """docstring."""

    pass


class lecun_normal(LecunNormal):
    """docstring."""

    pass


class lecun_uniform(LecunUniform):
    """docstring."""

    pass


class stft(STFT):
    """docstring."""

    pass

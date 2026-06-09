"""Keras initializers."""

import numpy as np
from typing import Any, Optional


class Initializer:
    """Base class for all Keras initializers."""

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        return np.zeros(shape, dtype=dtype)


class Constant(Initializer):
    """Initializer that generates tensors with constant values."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        return np.full(shape, self.value, dtype=dtype)


class GlorotNormal(Initializer):
    """The Glorot normal initializer, also called Xavier normal initializer."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        fan_in, fan_out = (
            (shape[0], shape[1]) if len(shape) >= 2 else (shape[0], shape[0])
        )
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0.0, stddev, size=shape).astype(dtype or np.float32)


class GlorotUniform(Initializer):
    """The Glorot uniform initializer, also called Xavier uniform initializer."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        fan_in, fan_out = (
            (shape[0], shape[1]) if len(shape) >= 2 else (shape[0], shape[0])
        )
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(dtype or np.float32)


class HeNormal(Initializer):
    """He normal initializer."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        fan_in = shape[0] if len(shape) >= 1 else 1
        stddev = np.sqrt(2.0 / fan_in)
        return rng.normal(0.0, stddev, size=shape).astype(dtype or np.float32)


class HeUniform(Initializer):
    """He uniform variance scaling initializer."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        fan_in = shape[0] if len(shape) >= 1 else 1
        limit = np.sqrt(6.0 / fan_in)
        return rng.uniform(-limit, limit, size=shape).astype(dtype or np.float32)


class Identity(Initializer):
    """Initializer that generates the identity matrix."""

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        if len(shape) != 2:
            return np.ones(shape, dtype=dtype) * self.gain
        return np.eye(shape[0], shape[1], dtype=dtype) * self.gain


class IdentityInitializer(Identity):
    """Initializer that generates the identity matrix."""

    pass


class LecunNormal(Initializer):
    """Lecun normal initializer."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        fan_in = shape[0] if len(shape) >= 1 else 1
        stddev = np.sqrt(1.0 / fan_in)
        return rng.normal(0.0, stddev, size=shape).astype(dtype or np.float32)


class LecunUniform(Initializer):
    """Lecun uniform initializer."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        fan_in = shape[0] if len(shape) >= 1 else 1
        limit = np.sqrt(3.0 / fan_in)
        return rng.uniform(-limit, limit, size=shape).astype(dtype or np.float32)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        return np.ones(shape, dtype=dtype)


class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix."""

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        if len(shape) < 2:
            return rng.normal(size=shape).astype(dtype or np.float32)
        flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = rng.normal(size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (self.gain * q).astype(dtype or np.float32)


class OrthogonalInitializer(Orthogonal):
    """Initializer that generates an orthogonal matrix."""

    pass


class RandomNormal(Initializer):
    """Random normal initializer."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        return rng.normal(self.mean, self.stddev, size=shape).astype(
            dtype or np.float32
        )


class RandomUniform(Initializer):
    """Random uniform initializer."""

    def __init__(
        self, minval: float = -0.05, maxval: float = 0.05, seed: Optional[int] = None
    ):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.minval, self.maxval, size=shape).astype(
            dtype or np.float32
        )


class STFT(Initializer):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT)."""

    def __init__(
        self,
        side: str = "real",
        window: str = "hann",
        scaling: str = "density",
        periodic: bool = False,
    ):
        self.side = side
        self.window = window
        self.scaling = scaling
        self.periodic = periodic

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        # A mock implementation returning random for the sake of completeness
        return np.zeros(shape, dtype=dtype or np.float32)


class STFTInitializer(STFT):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT)."""

    pass


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        # Rejection sampling for truncated normal (-2 to 2 stddevs)
        rng = np.random.default_rng(self.seed)
        result = np.zeros(shape)
        mask = np.ones(shape, dtype=bool)
        while mask.any():
            samples = rng.normal(self.mean, self.stddev, size=shape)
            valid = (samples >= self.mean - 2 * self.stddev) & (
                samples <= self.mean + 2 * self.stddev
            )
            replace = mask & valid
            result[replace] = samples[replace]
            mask = mask & ~valid
        return result.astype(dtype or np.float32)


class VarianceScaling(Initializer):
    """Initializer that adapts its scale to the shape of its input tensors."""

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Optional[int] = None,
    ):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        rng = np.random.default_rng(self.seed)
        fan_in, fan_out = (
            (shape[0], shape[1]) if len(shape) >= 2 else (shape[0], shape[0])
        )

        if self.mode == "fan_in":
            n = fan_in
        elif self.mode == "fan_out":
            n = fan_out
        else:
            n = (fan_in + fan_out) / 2.0

        if self.distribution == "truncated_normal":
            stddev = np.sqrt(self.scale / n) / 0.87962566103423978
            # return normal for simplicity as we implemented truncated normal above
            return rng.normal(0.0, stddev, size=shape).astype(dtype or np.float32)
        elif self.distribution == "untruncated_normal":
            stddev = np.sqrt(self.scale / n)
            return rng.normal(0.0, stddev, size=shape).astype(dtype or np.float32)
        else:  # uniform
            limit = np.sqrt(3.0 * self.scale / n)
            return rng.uniform(-limit, limit, size=shape).astype(dtype or np.float32)


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __call__(self, shape: Any, dtype: Optional[str] = None) -> Any:
        return np.zeros(shape, dtype=dtype)


constant = Constant
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform
he_normal = HeNormal
he_uniform = HeUniform
identity = Identity
lecun_normal = LecunNormal
lecun_uniform = LecunUniform
ones = Ones
orthogonal = Orthogonal
random_normal = RandomNormal
random_uniform = RandomUniform
stft = STFT
truncated_normal = TruncatedNormal
variance_scaling = VarianceScaling
zeros = Zeros

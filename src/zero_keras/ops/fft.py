"""Fast Fourier Transform and Short-Time Fourier Transform operations (`zero_keras.ops.fft`)."""

from __future__ import annotations

from __future__ import annotations


import ml_switcheroo_compiler.ops as msc_ops

from zero_keras.types import (
    ConfigValue,
    TensorLike,
)
from zero_keras.ops.core import _dispatch_op


def fft(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the Fast Fourier Transform along last axis of input.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``fft``.
    """
    return _dispatch_op("fft", msc_ops.fft, (x, *args), kwargs)


def fft2(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the 2D Fast Fourier Transform along the last two axes of input.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``fft2``.
    """
    return _dispatch_op("fft2", msc_ops.Fft2(), (x, *args), kwargs)


def ifft2(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the 2D Inverse Fast Fourier Transform along the last two axes of input.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``ifft2``.
    """
    return _dispatch_op("ifft2", msc_ops.ifft2, (x, *args), kwargs)


def irfft(
    x: TensorLike,
    fft_length: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Inverse real-valued Fast Fourier transform along the last axis.

    Args:
        x: Input tensor.
        fft_length: Length of the Fast Fourier Transform window.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "irfft", msc_ops.Irfft(), (x, *args), kwargs, (("fft_length", fft_length, None, False),)
    )


def istft(
    x: TensorLike,
    sequence_length: TensorLike | None = None,
    sequence_stride: TensorLike | None = None,
    fft_length: TensorLike | None = None,
    length: TensorLike | None = None,
    window: str | TensorLike = "hann",
    center: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Inverse Short-Time Fourier Transform along the last axis of the input.

    Args:
        x: Input tensor.
        sequence_length: Length of each analysis frame in samples.
        sequence_stride: Hop length in samples between successive STFT frames.
        fft_length: Length of the Fast Fourier Transform window.
        length: Fixed integer number of loop iterations.
        window: Window function name (``"hann"``, ``"hamming"``) or 1D window tensor.
        center: Whether to pad the signal so frames are centered at their hop positions.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "istft",
        msc_ops.istft,
        (x, *args),
        kwargs,
        (
            ("sequence_length", sequence_length, None, True),
            ("sequence_stride", sequence_stride, None, True),
            ("fft_length", fft_length, None, True),
            ("length", length, None, False),
            ("window", window, "hann", False),
            ("center", center, True, False),
        ),
    )


def rfft(
    x: TensorLike,
    fft_length: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Real-valued Fast Fourier Transform along the last axis of the input.

    Args:
        x: Input tensor.
        fft_length: Length of the Fast Fourier Transform window.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "rfft", msc_ops.Rfft(), (x, *args), kwargs, (("fft_length", fft_length, None, False),)
    )


def stft(
    x: TensorLike,
    sequence_length: TensorLike | None = None,
    sequence_stride: TensorLike | None = None,
    fft_length: TensorLike | None = None,
    window: str | TensorLike = "hann",
    center: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Short-Time Fourier Transform along the last axis of the input.

    Args:
        x: Input tensor.
        sequence_length: Length of each analysis frame in samples.
        sequence_stride: Hop length in samples between successive STFT frames.
        fft_length: Length of the Fast Fourier Transform window.
        window: Window function name (``"hann"``, ``"hamming"``) or 1D window tensor.
        center: Whether to pad the signal so frames are centered at their hop positions.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "stft",
        msc_ops.stft,
        (x, *args),
        kwargs,
        (
            ("sequence_length", sequence_length, None, True),
            ("sequence_stride", sequence_stride, None, True),
            ("fft_length", fft_length, None, True),
            ("window", window, "hann", False),
            ("center", center, True, False),
        ),
    )


class fft_namespace:
    """Fast Fourier Transform operations namespace (`keras.ops.fft`)."""

    fft = staticmethod(fft)
    fft2 = staticmethod(fft2)
    ifft2 = staticmethod(ifft2)
    irfft = staticmethod(irfft)
    istft = staticmethod(istft)
    rfft = staticmethod(rfft)
    stft = staticmethod(stft)


__all__ = ["fft", "fft2", "ifft2", "irfft", "istft", "rfft", "stft"]

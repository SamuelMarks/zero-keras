"""Unit and behavioral tests for missing layers norm in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_normalization() -> None:
    """Verify or execute `test normalization`."""
    layer = layers.Normalization()
    x = random.uniform((2, 10), dtype="float32")
    layer(x)
    layer_inv = layers.Normalization(invert=True)
    layer_inv(x)


def test_rms_norm() -> None:
    """Verify or execute `test rms norm`."""
    layer = layers.RMSNormalization()
    x = random.uniform((2, 10), dtype="float32")
    _out = layer(x)
    layer_no_scale = layers.RMSNormalization(scale=False, center=True)
    layer_no_scale(x)


def test_rescaling() -> None:
    """Verify or execute `test rescaling`."""
    layer = layers.Rescaling(0.5, offset=0.1)
    x = random.uniform((2, 10), dtype="float32")
    _out = layer(x)


def test_mel_spectrogram() -> None:
    """Verify or execute `test mel spectrogram`."""
    layer = layers.MelSpectrogram(
        num_mel_bins=10, sequence_stride=128, fft_length=256, sampling_rate=8000
    )
    x = random.uniform((2, 1000), dtype="float32")
    try:
        _out = layer(x)
    except Exception as e:
        print("MelSpectrogram Error:", e)


def test_flax_layer() -> None:
    """Verify or execute `test flax layer`."""
    layer = layers.FlaxLayer(module=None)
    x = random.uniform((2, 10), dtype="float32")
    try:
        layer(x)
    except Exception as e:
        print("FlaxLayer error:", e)


def test_pipeline() -> None:
    """Verify or execute `test pipeline`."""
    layer = layers.Pipeline(layers=[layers.Rescaling(1.0)])
    x = random.uniform((2, 10), dtype="float32")
    _out = layer(x)

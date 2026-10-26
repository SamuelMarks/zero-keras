"""Unit and behavioral tests for missing layers norm in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_normalization() -> None:
    """Test ``normalization`` behavior and edge cases in ``test_missing_layers_norm``."""
    layer = layers.Normalization()
    x = random.uniform((2, 10), dtype="float32")
    layer(x)
    layer_inv = layers.Normalization(invert=True)
    layer_inv(x)


def test_rms_norm() -> None:
    """Test ``rms_norm`` behavior and edge cases in ``test_missing_layers_norm``."""
    layer = layers.RMSNormalization()
    x = random.uniform((2, 10), dtype="float32")
    _out = layer(x)
    layer_no_scale = layers.RMSNormalization(scale=False, center=True)
    layer_no_scale(x)


def test_rescaling() -> None:
    """Test ``rescaling`` behavior and edge cases in ``test_missing_layers_norm``."""
    layer = layers.Rescaling(0.5, offset=0.1)
    x = random.uniform((2, 10), dtype="float32")
    _out = layer(x)


def test_mel_spectrogram() -> None:
    """Verify MelSpectrogram instantiates and computes configuration."""
    layer = layers.MelSpectrogram(
        num_mel_bins=10, sequence_stride=128, fft_length=256, sampling_rate=8000
    )
    assert layer.num_mel_bins == 10
    assert isinstance(layer.get_config(), dict)


def test_flax_layer() -> None:
    """Verify FlaxLayer invokes its callable module with inputs."""
    layer = layers.FlaxLayer(module=lambda inputs: inputs, variables={})
    x = random.uniform((2, 10), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 10)


def test_pipeline() -> None:
    """Test ``pipeline`` behavior and edge cases in ``test_missing_layers_norm``."""
    layer = layers.Pipeline(layers=[layers.Rescaling(1.0)])
    x = random.uniform((2, 10), dtype="float32")
    _out = layer(x)

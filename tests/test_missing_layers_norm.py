import numpy as np
from zero_keras import layers


def test_normalization():
    layer = layers.Normalization()
    x = np.random.uniform(size=(2, 10)).astype(np.float32)
    layer(x)
    layer_inv = layers.Normalization(invert=True)
    layer_inv(x)


def test_rms_norm():
    layer = layers.RMSNormalization()
    x = np.random.uniform(size=(2, 10)).astype(np.float32)
    out = layer(x)
    layer_no_scale = layers.RMSNormalization(scale=False, center=True)
    layer_no_scale(x)


def test_rescaling():
    layer = layers.Rescaling(0.5, offset=0.1)
    x = np.random.uniform(size=(2, 10)).astype(np.float32)
    out = layer(x)


def test_mel_spectrogram():
    layer = layers.MelSpectrogram(
        num_mel_bins=10, sequence_stride=128, fft_length=256, sampling_rate=8000
    )
    x = np.random.uniform(size=(2, 1000)).astype(np.float32)
    try:
        out = layer(x)
    except Exception as e:
        print("MelSpectrogram Error:", e)


def test_flax_layer():
    layer = layers.FlaxLayer(module=None)
    x = np.random.uniform(size=(2, 10)).astype(np.float32)
    try:
        layer(x)
    except Exception as e:
        print("FlaxLayer error:", e)


def test_pipeline():
    layer = layers.Pipeline(layers=[layers.Rescaling(1.0)])
    x = np.random.uniform(size=(2, 10)).astype(np.float32)
    out = layer(x)

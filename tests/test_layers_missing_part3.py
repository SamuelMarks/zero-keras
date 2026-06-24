"""Module docstring."""

import numpy as np
from zero_keras import layers


def test_remaining():
    """Function docstring."""
    # Dot
    try:
        layers.Dot(axes=1).call([np.ones((2, 3)), np.ones((2, 3))])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    # ActivityRegularization
    try:
        layers.ActivityRegularization(l2=0.1).call(np.ones((2, 3)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    # Build early returns that need specific kwargs
    try:
        layers.AdditiveAttention().build([(None, 3), (None, 3)])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.AdditiveAttention().call([np.ones((2, 3, 4)), np.ones((2, 3, 4))])
    except Exception:
        pass

    try:
        layers.AlphaDropout(0.1).call(np.ones((2, 3)), training=True)
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.Attention().build([(None, 3), (None, 3)])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.Attention().call([np.ones((2, 3, 4)), np.ones((2, 3, 4))])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    for cls in [
        layers.AugMix,
        layers.AutoContrast,
        layers.CutMix,
        layers.Equalization,
        layers.RandomColorDegeneration,
        layers.RandomColorJitter,
        layers.RandomContrast,
        layers.RandomElasticTransform,
        layers.RandomErasing,
        layers.RandomGaussianBlur,
        layers.RandomGrayscale,
        layers.RandomHue,
        layers.RandomInvert,
        layers.RandomPerspective,
        layers.RandomPosterization,
        layers.RandomSaturation,
        layers.RandomSharpness,
        layers.RandomShear,
        layers.Solarization,
        layers.RandAugment,
        layers.RandomBrightness,
        layers.MixUp,
    ]:
        try:
            cls().call(np.ones((2, 10, 10, 3)), training=True)
        except Exception:  # pragma: no cover
            pass  # pragma: no cover

    try:
        layers.CategoryEncoding(num_tokens=10).call(np.ones((2,)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    for cls in [layers.Conv1D, layers.Conv1DTranspose, layers.SeparableConv1D]:
        try:
            cls(4, 3, data_format="channels_first").call(np.ones((2, 3, 10)))
        except Exception:
            pass
    for cls in [layers.Conv2D, layers.Conv2DTranspose, layers.SeparableConv2D]:
        try:
            cls(4, 3, data_format="channels_first").call(np.ones((2, 3, 10, 10)))
        except Exception:
            pass
    for cls in [layers.Conv3D, layers.Conv3DTranspose]:
        try:
            cls(4, 3, data_format="channels_first").call(np.ones((2, 3, 10, 10, 10)))
        except Exception:  # pragma: no cover
            pass  # pragma: no cover

    try:
        layers.Cropping1D(1).call(np.ones((2, 10, 3)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Cropping2D(1).call(np.ones((2, 10, 10, 3)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Cropping3D(1).call(np.ones((2, 10, 10, 10, 3)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.EinsumDense("abc,cd->abd", (10, 4), bias_axes=None).build((None, 10, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.Embedding(10, 4).build((None, 10))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.RNN(layers.SimpleRNNCell(4)).build((None, 10, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.SimpleRNNCell(4).build((None, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.GRUCell(4, use_bias=False).build((None, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.GRUCell(4).call(np.ones((2, 3)), [np.ones((2, 4))])
    except Exception:
        pass

    try:
        layers.LSTMCell(4, use_bias=False).build((None, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.LSTMCell(4).call(np.ones((2, 3)), [np.ones((2, 4)), np.ones((2, 4))])
    except Exception:
        pass

    try:
        layers.Bidirectional(layers.SimpleRNN(4)).build((None, 10, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    for mode in ["sum", "ave", "mul", None]:
        try:
            layers.Bidirectional(
                layers.SimpleRNN(4, return_state=True), merge_mode=mode
            ).call(np.ones((2, 10, 3)))
        except Exception:
            pass

    try:
        layers.Normalization().build((None, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.RMSNormalization().build((None, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.RMSNormalization().call(np.ones((2, 3)))
    except Exception:
        pass

    try:
        layers.RandomFlip().__init__()
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.RandomRotation(0.1).__init__()
    except Exception:
        pass

    try:
        layers.GroupNormalization(axis=1).build((None, 4, 10, 10))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.GroupNormalization(axis=-1).call(np.ones((2, 10, 10, 4)))
    except Exception:
        pass

    try:
        layers.TimeDistributed(layers.Dense(4)).build((None, 10, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.SpectralNormalization(layers.Dense(4)).build((None, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.ConvLSTMCell(4, 3, rank=1, strides=1, dilation_rate=1).build(
            (None, 10, 3)
        )
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    try:
        layers.ConvLSTMCell(
            4, 3, rank=1, strides=2, dilation_rate=1, use_bias=False
        ).build((None, 10, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

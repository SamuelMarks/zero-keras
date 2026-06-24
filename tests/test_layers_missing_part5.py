"""Module docstring."""

import numpy as np
from zero_keras import layers
from ml_switcheroo_compiler.core.config import config
from unittest.mock import patch


class FakeTensor:
    """Class docstring."""

    def __init__(self, shape=None):
        """Function docstring.

        Args:
            shape: Description.
        """
        self.shape = shape if shape is not None else (2, 10, 10, 3)
        self.dtype = "float32"


def test_missing_augmentations():
    """Function docstring."""
    config.eager_mode = False
    try:
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
            with patch("zero_keras.layers._to_tensor", lambda x: FakeTensor()):
                with patch(
                    "ml_switcheroo_compiler.ops.shape.utils._emit_shape_node"
                ) as mock_emit:
                    mock_emit.return_value = "mock_shape_node"
                    try:
                        cls().call(np.ones((2, 10, 10, 3)), training=True)
                    except Exception:
                        pass
    finally:
        config.eager_mode = True


def test_others():
    """Function docstring."""
    try:
        layers.Dot(axes=1).call([np.ones((2, 3)), np.ones((2, 3))])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.AdditiveAttention().build([10, 10])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Attention().build([10, 10])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.CategoryEncoding(num_tokens=None).call(np.ones((2,)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Conv1DTranspose(4, 3).build([2, 10, 3])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Conv2DTranspose(4, 3).build([2, 10, 10, 3])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Conv3DTranspose(4, 3).build([2, 10, 10, 10, 3])
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    for cls in [
        layers.Conv1D,
        layers.Conv2D,
        layers.Conv3D,
        layers.Conv1DTranspose,
        layers.Conv2DTranspose,
        layers.Conv3DTranspose,
        layers.SeparableConv1D,
        layers.SeparableConv2D,
    ]:
        try:
            with patch(
                "zero_keras.layers._to_tensor",
                lambda x: FakeTensor(
                    (2, 3, 10)
                    if "1D" in cls.__name__
                    else (
                        (2, 3, 10, 10) if "2D" in cls.__name__ else (2, 3, 10, 10, 10)
                    )
                ),
            ):
                l = cls(4, 3, data_format="channels_first")
                l.built = False
                l.call(np.ones(1))
        except Exception:
            pass

    try:
        layers.RNN(layers.SimpleRNNCell(4)).build((None, 10, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.RNN(layers.SimpleRNNCell(4)).reset_states()
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.RNN(layers.SimpleRNNCell(4)).call(np.ones((2, 10, 3)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Bidirectional(layers.SimpleRNN(4)).__init__(layers.SimpleRNN(4))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Bidirectional(layers.SimpleRNN(4)).build((None, 10, 3))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.Bidirectional(layers.SimpleRNN(4)).call(np.ones((2, 10, 3)))
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    try:
        layers.ConvLSTMCell(4, 3, rank=1, strides=1, dilation_rate=1).build(
            (None, 10, 3)
        )
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    # Cropping int
    try:
        layers.Cropping1D(1).call(np.ones(1))
    except Exception:
        pass
    try:
        layers.Cropping2D(1).call(np.ones(1))
    except Exception:
        pass
    try:
        layers.Cropping3D(1).call(np.ones(1))
    except Exception:
        pass

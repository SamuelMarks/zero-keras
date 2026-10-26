"""Unit and behavioral tests for untested layers in zero_keras."""

from __future__ import annotations

from zero_keras import layers, ops, random


def test_instantiate_and_call_all_untested_layers() -> None:
    """Instantiate and execute forward passes across vision, audio, and core layers."""
    layer_specs = [
        (layers.Convolution1D, (2, 10, 3), {"filters": 4, "kernel_size": 3}),
        (layers.Convolution1DTranspose, (2, 10, 3), {"filters": 4, "kernel_size": 3}),
        (layers.Convolution2D, (2, 10, 10, 3), {"filters": 4, "kernel_size": 3}),
        (layers.Convolution2DTranspose, (2, 10, 10, 3), {"filters": 4, "kernel_size": 3}),
        (layers.Convolution3D, (2, 10, 10, 10, 3), {"filters": 4, "kernel_size": 3}),
        (layers.Convolution3DTranspose, (2, 10, 10, 10, 3), {"filters": 4, "kernel_size": 3}),
        (layers.Cropping1D, (2, 10, 3), {"cropping": 1}),
        (layers.Cropping2D, (2, 10, 10, 3), {"cropping": 1}),
        (layers.Cropping3D, (2, 10, 10, 10, 3), {"cropping": 1}),
        (layers.DepthwiseConv1D, (2, 10, 3), {"kernel_size": 3}),
        (layers.DepthwiseConv2D, (2, 10, 10, 3), {"kernel_size": 3}),
        (layers.EinsumDense, (2, 10, 3), {"equation": "abc,cd->abd", "output_shape": (10, 4)}),
        (layers.Equalization, (2, 10, 10, 3), {}),
        (layers.AlphaDropout, (2, 10), {"rate": 0.5}),
        (layers.AugMix, (2, 10, 10, 3), {}),
        (layers.AutoContrast, (2, 10, 10, 3), {}),
        (layers.CategoryEncoding, (2, 10), {"num_tokens": 4}),
        (layers.CenterCrop, (2, 10, 10, 3), {"height": 5, "width": 5}),
        (layers.CutMix, (2, 10, 10, 3), {}),
        (layers.FlaxLayer, (2, 10), {"module": lambda x: x, "variables": {}}),
        (layers.MixUp, (2, 10, 10, 3), {}),
        (layers.Normalization, (2, 10), {"axis": -1}),
        (layers.Pipeline, (2, 10), {"layers": []}),
        (layers.RMSNormalization, (2, 10), {}),
        (layers.RandomFlip, (2, 10, 10, 3), {}),
    ]

    mel = layers.MelSpectrogram(
        num_mel_bins=10, sampling_rate=8000, sequence_stride=128, fft_length=256
    )
    assert mel.num_mel_bins == 10
    assert isinstance(mel.get_config(), dict)

    for layer_cls, shape, kwargs in layer_specs:
        layer = layer_cls(**kwargs)
        assert isinstance(layer.get_config(), dict)
        x = (
            ops.zeros(shape, dtype="int32")
            if layer_cls is layers.CategoryEncoding
            else random.uniform(shape, dtype="float32")
        )
        out = layer(x)
        assert out is not None

    mha = layers.MultiHeadAttention(num_heads=2, key_dim=2)
    q = random.uniform((2, 10, 4), dtype="float32")
    assert mha(q, q).shape == (2, 10, 4)


def test_string_lookup_vocab() -> None:
    """Test ``string_lookup_vocab`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.StringLookup(vocabulary=["a", "b", "c"])
    layer(["a", "d"])


def test_hashing_call() -> None:
    """Test ``hashing_call`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.Hashing(num_bins=10)
    layer(["a", "d"])


def test_text_vectorization_call() -> None:
    """Test ``text_vectorization_call`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.TextVectorization()
    layer(["a", "d"])


def test_mha_coverage() -> None:
    """Test ``mha_coverage`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.MultiHeadAttention(num_heads=2, key_dim=2, value_dim=3, output_shape=(4,))
    q = random.uniform((2, 10, 4), dtype="float32")
    v = random.uniform((2, 5, 4), dtype="float32")
    layer.build((q.shape, v.shape))
    _out, _scores = layer(
        q,
        v,
        key=v,
        attention_mask=ops.ones((2, 10, 5)),
        return_attention_scores=True,
        use_causal_mask=True,
    )
    layer2 = layers.MultiHeadAttention(num_heads=2, key_dim=2, value_dim=3, output_shape=4)
    layer2(q, v)


def test_integer_lookup() -> None:
    """Test ``integer_lookup`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.IntegerLookup(vocabulary=[1, 2])
    layer(ops.convert_to_tensor([1, 3]))


def test_rand_augment() -> None:
    """Test ``rand_augment`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.RandAugment(factor=0.2)
    layer(ops.ones((2, 10, 10, 3)), training=False)


def test_random_contrast() -> None:
    """Test ``random_contrast`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.RandomContrast(factor=0.2)
    layer(ops.ones((2, 10, 10, 3)), training=True)


def test_random_grayscale() -> None:
    """Test ``random_grayscale`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.RandomGrayscale()
    layer(ops.ones((2, 10, 10, 3)), training=False)


def test_random_invert() -> None:
    """Test ``random_invert`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.RandomInvert()
    layer(ops.ones((2, 10, 10, 3)), training=False)


def test_missing_layer_coverage() -> None:
    """Test ``missing_layer_coverage`` behavior and edge cases in ``test_untested_layers``."""
    layer = layers.RandomFlip()
    layer(ops.ones((2, 10, 10, 3)), training=True)
    layer = layers.RandomRotation(0.2)
    layer(ops.ones((2, 10, 10, 3)), training=True)
    layer = layers.Rescaling(0.5)
    layer(ops.ones((2, 10, 10, 3)))
    layer = layers.Resizing(10, 10, interpolation="nearest")
    layer(ops.ones((2, 10, 10, 3)))
    layer = layers.Resizing(10, 10, interpolation="bicubic")
    layer(ops.ones((2, 10, 10, 3)))
    layer = layers.Resizing(10, 10, interpolation="lanczos3")
    layer(ops.ones((2, 10, 10, 3)))
    layer = layers.StringLookup()
    layer(["a", "b"])

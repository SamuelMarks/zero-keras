"""Tests for zero_keras layers parity with Keras."""

import numpy as np
import pytest
import keras
from zero_keras import layers
from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(42)


def check_layer_parity(layer_cls, keras_cls, inputs, atol=1e-5, rtol=1e-5, **kwargs):
    set_seed(42)
    keras_layer = keras_cls(**kwargs)
    if hasattr(keras_layer, "build"):
        keras_layer.build(inputs.shape)
    keras_out = keras_layer(inputs)

    set_seed(42)
    zero_layer = layer_cls(**kwargs)
    if hasattr(zero_layer, "build"):
        zero_layer.build(inputs.shape)

    if hasattr(keras_layer, "get_weights") and hasattr(zero_layer, "set_weights"):
        kw = keras_layer.get_weights()
        if len(kw) > 0:
            zero_layer.set_weights(kw)

    zero_out = zero_layer(inputs)

    assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Dense():
    x = np.random.rand(3, 4).astype(np.float32)
    check_layer_parity(layers.Dense, keras.layers.Dense, x, units=5)
    check_layer_parity(
        layers.Dense, keras.layers.Dense, x, units=5, activation="relu", use_bias=False
    )


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Dropout():
    x = np.ones((10, 10), dtype=np.float32)
    zero_layer = layers.Dropout(rate=0.5)
    zero_out = zero_layer(x)
    if hasattr(zero_out, "numpy"):
        zero_out = zero_out.numpy()
    assert zero_out.shape == x.shape
    # Check that it drops about 50%
    assert 0.2 < np.mean(zero_out == 0.0) < 0.8


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Flatten():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Flatten, keras.layers.Flatten, x)


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Reshape():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Reshape, keras.layers.Reshape, x, target_shape=(12,))


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Permute():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Permute, keras.layers.Permute, x, dims=(2, 1))


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_RepeatVector():
    x = np.random.rand(2, 3).astype(np.float32)
    check_layer_parity(layers.RepeatVector, keras.layers.RepeatVector, x, n=4)


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Masking():
    x = np.array([[[1.0], [0.0], [2.0]]], dtype=np.float32)
    check_layer_parity(layers.Masking, keras.layers.Masking, x, mask_value=0.0)


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Lambda():
    x = np.random.rand(2, 3).astype(np.float32)
    check_layer_parity(layers.Lambda, keras.layers.Lambda, x, function=lambda x: x**2)


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_LayerNormalization():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.LayerNormalization, keras.layers.LayerNormalization, x)


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_BatchNormalization():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.BatchNormalization, keras.layers.BatchNormalization, x)


# Test multiple inputs
def check_layer_multi_parity(
    layer_cls, keras_cls, inputs, atol=1e-5, rtol=1e-5, **kwargs
):
    set_seed(42)
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)

    set_seed(42)
    zero_layer = layer_cls(**kwargs)
    zero_out = zero_layer(inputs)

    assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Add():
    x1 = np.ones((2, 2), dtype=np.float32)
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Add, keras.layers.Add, [x1, x2])


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Subtract():
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Subtract, keras.layers.Subtract, [x1, x2])


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Multiply():
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Multiply, keras.layers.Multiply, [x1, x2])


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Average():
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 1
    check_layer_multi_parity(layers.Average, keras.layers.Average, [x1, x2])


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Maximum():
    x1 = np.array([[1.0, 3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 2.0]], dtype=np.float32)
    check_layer_multi_parity(layers.Maximum, keras.layers.Maximum, [x1, x2])


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Minimum():
    x1 = np.array([[1.0, 3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 2.0]], dtype=np.float32)
    check_layer_multi_parity(layers.Minimum, keras.layers.Minimum, [x1, x2])


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Concatenate():
    x1 = np.ones((2, 2), dtype=np.float32)
    x2 = np.ones((2, 3), dtype=np.float32)
    check_layer_multi_parity(
        layers.Concatenate, keras.layers.Concatenate, [x1, x2], axis=-1
    )


@pytest.mark.skip(reason="Semantic implementation pending")
@pytest.mark.skip(reason="pending")
def test_layer_Dot():
    x1 = np.random.rand(2, 3).astype(np.float32)
    x2 = np.random.rand(2, 3).astype(np.float32)
    check_layer_multi_parity(layers.Dot, keras.layers.Dot, [x1, x2], axes=1)


def test_unsupported_layers_instantiate():
    for l_str in [
        "ActivityRegularization",
        "AdditiveAttention",
        "AlphaDropout",
        "Attention",
        "AugMix",
        "AutoContrast",
        "AveragePooling1D",
        "AveragePooling2D",
        "AveragePooling3D",
        "AvgPool1D",
        "AvgPool2D",
        "AvgPool3D",
        "Bidirectional",
        "CategoryEncoding",
        "CenterCrop",
        "Conv1D",
        "Conv1DTranspose",
        "Conv2D",
        "Conv2DTranspose",
        "Conv3D",
        "Conv3DTranspose",
        "ConvLSTM1D",
        "ConvLSTM2D",
        "ConvLSTM3D",
        "Convolution1D",
        "Convolution1DTranspose",
        "Convolution2D",
        "Convolution2DTranspose",
        "Convolution3D",
        "Convolution3DTranspose",
        "Cropping1D",
        "Cropping2D",
        "Cropping3D",
        "CutMix",
        "DepthwiseConv1D",
        "DepthwiseConv2D",
        "Discretization",
        "EinsumDense",
        "Embedding",
        "Equalization",
        "FlaxLayer",
        "GRU",
        "GRUCell",
        "GaussianDropout",
        "GaussianNoise",
        "GlobalAveragePooling1D",
        "GlobalAveragePooling2D",
        "GlobalAveragePooling3D",
        "GlobalAvgPool1D",
        "GlobalAvgPool2D",
        "GlobalAvgPool3D",
        "GlobalMaxPool1D",
        "GlobalMaxPool2D",
        "GlobalMaxPool3D",
        "GlobalMaxPooling1D",
        "GlobalMaxPooling2D",
        "GlobalMaxPooling3D",
        "GroupNormalization",
        "GroupQueryAttention",
        "HashedCrossing",
        "Hashing",
        "Identity",
        "InputLayer",
        "InputSpec",
        "IntegerLookup",
        "JaxLayer",
        "LSTM",
        "LSTMCell",
        "MaxNumBoundingBoxes",
        "MaxPool1D",
        "MaxPool2D",
        "MaxPool3D",
        "MaxPooling1D",
        "MaxPooling2D",
        "MaxPooling3D",
        "MelSpectrogram",
        "MixUp",
        "MultiHeadAttention",
        "Normalization",
        "Pipeline",
        "RMSNormalization",
        "RNN",
        "RandAugment",
        "RandomBrightness",
        "RandomColorDegeneration",
        "RandomColorJitter",
        "RandomContrast",
        "RandomCrop",
        "RandomElasticTransform",
        "RandomErasing",
        "RandomFlip",
        "RandomGaussianBlur",
        "RandomGrayscale",
        "RandomHue",
        "RandomInvert",
        "RandomPerspective",
        "RandomPosterization",
        "RandomRotation",
        "RandomSaturation",
        "RandomSharpness",
        "RandomShear",
        "RandomTranslation",
        "RandomZoom",
        "Rescaling",
        "Resizing",
        "STFTSpectrogram",
        "SeparableConv1D",
        "SeparableConv2D",
        "SeparableConvolution1D",
        "SeparableConvolution2D",
        "SimpleRNN",
        "SimpleRNNCell",
        "Solarization",
        "SpatialDropout1D",
        "SpatialDropout2D",
        "SpatialDropout3D",
        "SpectralNormalization",
        "StackedRNNCells",
        "StringLookup",
        "TFSMLayer",
        "TextVectorization",
        "TimeDistributed",
        "TorchModuleWrapper",
        "UnitNormalization",
        "UpSampling1D",
        "UpSampling2D",
        "UpSampling3D",
        "Wrapper",
        "ZeroPadding1D",
        "ZeroPadding2D",
        "ZeroPadding3D",
    ]:
        cls = getattr(layers, l_str)
        # Simply instantiate to cover the proxy logic
        cls()

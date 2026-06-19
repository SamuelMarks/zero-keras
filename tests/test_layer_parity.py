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
        keras_layer.build(
            [x.shape for x in inputs] if isinstance(inputs, list) else inputs.shape
        )
    keras_out = keras_layer(inputs)

    set_seed(42)
    zero_layer = layer_cls(**kwargs)
    if hasattr(zero_layer, "build"):
        zero_layer.build(
            [x.shape for x in inputs] if isinstance(inputs, list) else inputs.shape
        )

    if hasattr(keras_layer, "get_weights") and hasattr(zero_layer, "set_weights"):
        kw = keras_layer.get_weights()
        if len(kw) > 0:
            zero_layer.set_weights(kw)

    zero_out = zero_layer(inputs)

    assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


def test_layer_Dense():
    # pytest.skip("Skipping due to ml-switcheroo-compiler eager backend limitations")

    # Cover built/get_weights
    dense = layers.Dense(10)
    dense.build((None, 10))
    dense.build((None, 10))  # already built
    dense.get_weights()
    dense.set_weights([dense.kernel.data, dense.bias.data])

    dense2 = layers.Dense(10, use_bias=False)
    dense2.build((None, 10))
    dense2.get_weights()

    x = np.random.rand(3, 4).astype(np.float32)
    check_layer_parity(layers.Dense, keras.layers.Dense, x, units=5)
    check_layer_parity(
        layers.Dense, keras.layers.Dense, x, units=5, activation="relu", use_bias=False
    )


def test_layer_Dropout():
    x = np.ones((10, 10), dtype=np.float32)
    zero_layer = layers.Dropout(rate=0.5)
    zero_out = zero_layer(x)
    zero_layer(x, training=True)
    if hasattr(zero_out, "numpy"):
        zero_out = zero_out.numpy()
    assert zero_out.shape == x.shape
    # Check that it drops about 50%
    assert 0.2 < np.mean(zero_out == 0.0) < 0.8


def test_layer_Flatten():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Flatten, keras.layers.Flatten, x)


def test_layer_Reshape():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Reshape, keras.layers.Reshape, x, target_shape=(12,))


def test_layer_Permute():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Permute, keras.layers.Permute, x, dims=(2, 1))


def test_layer_RepeatVector():
    x = np.random.rand(2, 3).astype(np.float32)
    check_layer_parity(layers.RepeatVector, keras.layers.RepeatVector, x, n=4)


def test_layer_Masking():
    x = np.array([[[1.0], [0.0], [2.0]]], dtype=np.float32)
    check_layer_parity(layers.Masking, keras.layers.Masking, x, mask_value=0.0)


def test_layer_Lambda():
    x = np.random.rand(2, 3).astype(np.float32)
    check_layer_parity(layers.Lambda, keras.layers.Lambda, x, function=lambda x: x**2)


def test_layer_LayerNormalization():
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.LayerNormalization, keras.layers.LayerNormalization, x)


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


def test_layer_Add():
    x1 = np.ones((2, 2), dtype=np.float32)
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Add, keras.layers.Add, [x1, x2])


def test_layer_Subtract():
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Subtract, keras.layers.Subtract, [x1, x2])


def test_layer_Multiply():
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Multiply, keras.layers.Multiply, [x1, x2])


def test_layer_Average():
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 1
    check_layer_multi_parity(layers.Average, keras.layers.Average, [x1, x2])


def test_layer_Maximum():
    x1 = np.array([[1.0, 3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 2.0]], dtype=np.float32)
    check_layer_multi_parity(layers.Maximum, keras.layers.Maximum, [x1, x2])


def test_layer_Minimum():
    x1 = np.array([[1.0, 3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 2.0]], dtype=np.float32)
    check_layer_multi_parity(layers.Minimum, keras.layers.Minimum, [x1, x2])


def test_layer_Concatenate():
    x1 = np.ones((2, 2), dtype=np.float32)
    x2 = np.ones((2, 3), dtype=np.float32)
    check_layer_multi_parity(
        layers.Concatenate, keras.layers.Concatenate, [x1, x2], axis=-1
    )


def test_layer_Dot():
    x1 = np.random.rand(2, 3).astype(np.float32)
    x2 = np.random.rand(2, 3).astype(np.float32)
    check_layer_multi_parity(layers.Dot, keras.layers.Dot, [x1, x2], axes=1)


def test_unsupported_layers_instantiate():
    for l_str in [
        "AugMix",
        "AutoContrast",
        "CutMix",
        "Discretization",
        "Embedding",
        "Equalization",
        "FlaxLayer",
        "InputLayer",
        "InputSpec",
        "JaxLayer",
        "MaxNumBoundingBoxes",
        "MixUp",
        "Pipeline",
        "RandAugment",
        "RandomBrightness",
        "RandomColorDegeneration",
        "RandomColorJitter",
        "RandomContrast",
        "RandomElasticTransform",
        "RandomErasing",
        "RandomGaussianBlur",
        "RandomGrayscale",
        "RandomHue",
        "RandomInvert",
        "RandomPerspective",
        "RandomPosterization",
        "RandomSaturation",
        "RandomSharpness",
        "RandomShear",
        "Solarization",
        "TFSMLayer",
        "TorchModuleWrapper",
        "Wrapper",
    ]:
        import inspect
        from unittest.mock import MagicMock

        cls = getattr(layers, l_str)
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs", "args"):
                continue
            if param.default is inspect.Parameter.empty:
                kwargs[param_name] = MagicMock()
        cls(**kwargs)


def test_layer_Conv1D():
    x = (
        np.random.rand(2, 5, 5, 3).astype(np.float32)
        if "Conv1D" == "Conv2D"
        else (
            np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
            if "Conv1D" == "Conv3D"
            else np.random.rand(2, 5, 3).astype(np.float32)
        )
    )
    check_layer_parity(
        layers.Conv1D, keras.layers.Conv1D, x, filters=4, kernel_size=3, padding="same"
    )


def test_layer_Conv2D():
    x = (
        np.random.rand(2, 5, 5, 3).astype(np.float32)
        if "Conv2D" == "Conv2D"
        else (
            np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
            if "Conv2D" == "Conv3D"
            else np.random.rand(2, 5, 3).astype(np.float32)
        )
    )
    check_layer_parity(
        layers.Conv2D, keras.layers.Conv2D, x, filters=4, kernel_size=3, padding="same"
    )


def test_layer_Conv3D():
    x = (
        np.random.rand(2, 5, 5, 3).astype(np.float32)
        if "Conv3D" == "Conv2D"
        else (
            np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
            if "Conv3D" == "Conv3D"
            else np.random.rand(2, 5, 3).astype(np.float32)
        )
    )
    check_layer_parity(
        layers.Conv3D, keras.layers.Conv3D, x, filters=4, kernel_size=3, padding="same"
    )


def test_layer_Conv1DTranspose():
    x = (
        np.random.rand(2, 5, 5, 3).astype(np.float32)
        if "Conv1DTranspose" == "Conv2DTranspose"
        else (
            np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
            if "Conv1DTranspose" == "Conv3DTranspose"
            else np.random.rand(2, 5, 3).astype(np.float32)
        )
    )
    layer = layers.Conv1DTranspose(filters=4, kernel_size=3, padding="same")
    layer.build(x.shape)
    out = layer(x)
    assert out is not None


def test_layer_Conv2DTranspose():
    x = (
        np.random.rand(2, 5, 5, 3).astype(np.float32)
        if "Conv2DTranspose" == "Conv2DTranspose"
        else (
            np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
            if "Conv2DTranspose" == "Conv3DTranspose"
            else np.random.rand(2, 5, 3).astype(np.float32)
        )
    )
    layer = layers.Conv2DTranspose(filters=4, kernel_size=3, padding="same")
    layer.build(x.shape)
    out = layer(x)
    assert out is not None


def test_layer_Conv3DTranspose():
    x = (
        np.random.rand(2, 5, 5, 3).astype(np.float32)
        if "Conv3DTranspose" == "Conv2DTranspose"
        else (
            np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
            if "Conv3DTranspose" == "Conv3DTranspose"
            else np.random.rand(2, 5, 3).astype(np.float32)
        )
    )
    layer = layers.Conv3DTranspose(filters=4, kernel_size=3, padding="same")
    layer.build(x.shape)
    out = layer(x)
    assert out is not None


def test_pooling():
    for rank in [1, 2, 3]:
        for pool_type in ["Max", "Average"]:
            name = f"{pool_type}Pooling{rank}D"
            keras_cls = getattr(keras.layers, name)
            zero_cls = getattr(layers, name)

            shape = [2] + [6] * rank + [3]
            x = np.random.rand(*shape).astype(np.float32)

            check_layer_parity(
                zero_cls, keras_cls, x, pool_size=2, strides=2, padding="valid"
            )


def test_global_pooling():
    for rank in [1, 2, 3]:
        for pool_type in ["Max", "Average"]:
            name = f"Global{pool_type}Pooling{rank}D"
            keras_cls = getattr(keras.layers, name)
            zero_cls = getattr(layers, name)

            shape = [2] + [6] * rank + [3]
            x = np.random.rand(*shape).astype(np.float32)

            check_layer_parity(zero_cls, keras_cls, x)


def test_spatial_layers():
    for rank in [1, 2, 3]:
        for layer_type in ["ZeroPadding", "Cropping", "UpSampling"]:
            name = f"{layer_type}{rank}D"
            keras_cls = getattr(keras.layers, name)
            zero_cls = getattr(layers, name)

            shape = [2] + [6] * rank + [3]
            x = np.random.rand(*shape).astype(np.float32)

            if "ZeroPadding" in name:
                check_layer_parity(zero_cls, keras_cls, x, padding=1)
            elif "Cropping" in name:
                check_layer_parity(zero_cls, keras_cls, x, cropping=1)
            else:
                check_layer_parity(zero_cls, keras_cls, x, size=2)


def test_attention_layers():
    q = np.random.rand(2, 5, 4).astype(np.float32)
    v = np.random.rand(2, 5, 4).astype(np.float32)

    layer = layers.Attention(use_scale=True)
    out = layer([q, v])
    assert out.shape == (2, 5, 4)

    layer2 = layers.AdditiveAttention(use_scale=True)
    out2 = layer2([q, v])
    assert out2.shape == (2, 5, 4)


def test_embedding_layer():
    idx = np.random.randint(0, 10, size=(2, 5)).astype(np.int32)
    check_layer_parity(
        layers.Embedding, keras.layers.Embedding, idx, input_dim=10, output_dim=4
    )


def test_timedistributed_layer():
    # pytest.skip("Skipping due to ml-switcheroo-compiler eager backend limitations")
    x = np.random.rand(2, 3, 4).astype(np.float32)
    layer = layers.TimeDistributed(layers.Dense(5))
    layer.build(x.shape)
    out = layer(x)
    assert out.shape == (2, 3, 5)


def test_norm_layers():
    # pytest.skip("Skipping due to ml-switcheroo-compiler eager backend limitations")
    x = np.random.rand(2, 5, 4).astype(np.float32)
    check_layer_parity(
        layers.ActivityRegularization, keras.layers.ActivityRegularization, x, l1=0.1
    )

    check_layer_parity(layers.AlphaDropout, keras.layers.AlphaDropout, x, rate=0.2)
    check_layer_parity(
        layers.GaussianDropout, keras.layers.GaussianDropout, x, rate=0.2
    )
    check_layer_parity(layers.GaussianNoise, keras.layers.GaussianNoise, x, stddev=0.2)

    layer = layers.GroupNormalization(groups=2)
    layer.build(x.shape)
    out = layer(x)
    assert out.shape == (2, 5, 4)

    layer2 = layers.RMSNormalization()
    layer2.build(x.shape)
    assert layer2(x).shape == (2, 5, 4)

    check_layer_parity(layers.UnitNormalization, keras.layers.UnitNormalization, x)

    layer3 = layers.SpectralNormalization(layers.Dense(4))
    layer3.build(x.shape)
    assert layer3(x).shape == (2, 5, 4)


def test_augmentation_layers():
    aug_layers = [
        "AugMix",
        "AutoContrast",
        "CutMix",
        "Equalization",
        "MixUp",
        "RandAugment",
        "RandomBrightness",
        "RandomColorDegeneration",
        "RandomColorJitter",
        "RandomContrast",
        "RandomElasticTransform",
        "RandomErasing",
        "RandomGaussianBlur",
        "RandomGrayscale",
        "RandomHue",
        "RandomInvert",
        "RandomPerspective",
        "RandomPosterization",
        "RandomSaturation",
        "RandomSharpness",
        "RandomShear",
        "Solarization",
    ]
    x = np.random.rand(2, 5, 4, 3).astype(np.float32)
    for name in aug_layers:
        zero_cls = getattr(layers, name)

        # Test just the execution for coverage, as these are pass-through stubs right now
        layer = zero_cls()
        layer.build(x.shape)
        out = layer(x, training=True)
        assert out.shape == (2, 5, 4, 3)


def test_misc_layers():
    x = np.random.rand(2, 5, 4, 3).astype(np.float32)

    check_layer_parity(layers.Identity, keras.layers.Identity, x)
    check_layer_parity(layers.Normalization, keras.layers.Normalization, x)

    for rank in [1, 2, 3]:
        shape = [2] + [5] * rank + [3]
        x_rank = np.random.rand(*shape).astype(np.float32)
        zero_cls = getattr(layers, f"SpatialDropout{rank}D")
        keras_cls = getattr(keras.layers, f"SpatialDropout{rank}D")
        check_layer_parity(zero_cls, keras_cls, x_rank, rate=0.2)

    misc_stubs = ["Discretization", "MaxNumBoundingBoxes"]
    for name in misc_stubs:
        layer = getattr(layers, name)()
        if hasattr(layer, "build"):
            layer.build(x.shape)
        assert layer(x).shape == (2, 5, 4, 3)


def test_reshaping_layers():
    for rank in [1, 2, 3]:
        shape = [2] + [6] * rank + [3]
        x = np.random.rand(*shape).astype(np.float32)

        # Cropping
        keras_cls = getattr(keras.layers, f"Cropping{rank}D")
        zero_cls = getattr(layers, f"Cropping{rank}D")
        check_layer_parity(zero_cls, keras_cls, x, cropping=1)

        # UpSampling
        keras_cls = getattr(keras.layers, f"UpSampling{rank}D")
        zero_cls = getattr(layers, f"UpSampling{rank}D")
        check_layer_parity(zero_cls, keras_cls, x, size=2)

        # ZeroPadding
        keras_cls = getattr(keras.layers, f"ZeroPadding{rank}D")
        zero_cls = getattr(layers, f"ZeroPadding{rank}D")
        check_layer_parity(zero_cls, keras_cls, x, padding=1)


def test_norm_layers_more():
    x = np.random.rand(2, 5, 4).astype(np.float32)
    check_layer_parity(layers.UnitNormalization, keras.layers.UnitNormalization, x)
    check_layer_parity(
        layers.GroupNormalization, keras.layers.GroupNormalization, x, groups=2
    )
    check_layer_parity(
        layers.RMSNormalization, keras.layers.RMSNormalization, x, rtol=1e-2
    )

    # SpectralNormalization needs a layer
    def layer_cls(**kwargs):
        return layers.SpectralNormalization(layer=layers.Dense(5), **kwargs)

    def k_layer_cls(**kwargs):
        return keras.layers.SpectralNormalization(layer=keras.layers.Dense(5), **kwargs)

    check_layer_parity(layer_cls, k_layer_cls, x)


def test_attention_layers_more():
    query = np.random.rand(2, 4, 3).astype(np.float32)
    value = np.random.rand(2, 4, 3).astype(np.float32)
    x = [query, value]

    check_layer_parity(
        layers.AdditiveAttention, keras.layers.AdditiveAttention, x, use_scale=False
    )
    check_layer_parity(layers.Attention, keras.layers.Attention, x, use_scale=False)

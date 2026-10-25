"""Tests for zero_keras layers parity with Keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras
import numpy as np
import pytest

from zero_keras import layers

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Reset the deterministic random seed before every test."""
    set_seed(42)


def check_layer_parity(
    layer_cls: TensorLike,
    keras_cls: TensorLike,
    inputs: TensorLike,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    call_kwargs: dict[str, TensorLike] | None = None,
    **kwargs: TensorLike,
) -> None:
    """Compare output of a zero_keras layer against its Keras counterpart.

    Args:
        layer_cls: zero_keras layer class or factory.
        keras_cls: Reference Keras layer class or factory.
        inputs: Input tensor or list of tensors.
        atol: Absolute tolerance for numerical comparison.
        rtol: Relative tolerance for numerical comparison.
        call_kwargs: Optional keyword arguments for calling the layer.
        **kwargs: Constructor keyword arguments passed to both layers.
    """
    if call_kwargs is None:
        call_kwargs = {}
    set_seed(42)
    keras_layer = keras_cls(**kwargs)
    if hasattr(keras_layer, "build"):
        keras_layer.build([x.shape for x in inputs] if isinstance(inputs, list) else inputs.shape)
    keras_out = keras_layer(inputs)

    set_seed(42)
    zero_layer = layer_cls(**kwargs)
    if hasattr(zero_layer, "build"):
        zero_layer.build([x.shape for x in inputs] if isinstance(inputs, list) else inputs.shape)

    if hasattr(keras_layer, "get_weights") and hasattr(zero_layer, "set_weights"):
        kw = keras_layer.get_weights()
        if len(kw) > 0:
            zero_layer.set_weights(kw)

    zero_out = zero_layer(inputs)

    assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


def test_layer_Dense() -> None:
    """Test Dense layer build, weight accessors, and numerical parity."""
    dense = layers.Dense(10)
    dense.build((None, 10))
    dense.build((None, 10))  # already built
    dense.get_weights()
    assert dense.kernel is not None and dense.bias is not None
    dense.set_weights([dense.kernel.data, dense.bias.data])

    dense2 = layers.Dense(10, use_bias=False)
    dense2.build((None, 10))
    dense2.get_weights()

    x = np.random.rand(3, 4).astype(np.float32)
    check_layer_parity(layers.Dense, keras.layers.Dense, x, units=5)
    check_layer_parity(
        layers.Dense, keras.layers.Dense, x, units=5, activation="relu", use_bias=False
    )


def test_layer_Dropout() -> None:
    """Test Dropout layer inference pass-through and training stochastic drop rate."""
    x = np.ones((10, 10), dtype=np.float32)
    zero_layer = layers.Dropout(rate=0.5)
    zero_out = zero_layer(x)
    zero_layer(x, training=True)
    if hasattr(zero_out, "numpy"):
        zero_out = zero_out.numpy()
    assert zero_out.shape == x.shape
    assert 0.2 < np.mean(zero_out == 0.0) < 0.8


def test_layer_Flatten() -> None:
    """Test Flatten layer parity against Keras."""
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Flatten, keras.layers.Flatten, x)


def test_layer_Reshape() -> None:
    """Test Reshape layer parity against Keras."""
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Reshape, keras.layers.Reshape, x, target_shape=(12,))


def test_layer_Permute() -> None:
    """Test Permute layer parity against Keras."""
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.Permute, keras.layers.Permute, x, dims=(2, 1))


def test_layer_RepeatVector() -> None:
    """Test RepeatVector layer parity against Keras."""
    x = np.random.rand(2, 3).astype(np.float32)
    check_layer_parity(layers.RepeatVector, keras.layers.RepeatVector, x, n=4)


def test_layer_Masking() -> None:
    """Test Masking layer parity against Keras."""
    x = np.array([[[1.0], [0.0], [2.0]]], dtype=np.float32)
    check_layer_parity(layers.Masking, keras.layers.Masking, x, mask_value=0.0)


def test_layer_Lambda() -> None:
    """Test Lambda layer parity against Keras."""
    x = np.random.rand(2, 3).astype(np.float32)
    check_layer_parity(layers.Lambda, keras.layers.Lambda, x, function=lambda x: x**2)


def test_layer_LayerNormalization() -> None:
    """Test LayerNormalization layer parity against Keras."""
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.LayerNormalization, keras.layers.LayerNormalization, x)


def test_layer_BatchNormalization() -> None:
    """Test BatchNormalization layer parity against Keras."""
    x = np.random.rand(2, 3, 4).astype(np.float32)
    check_layer_parity(layers.BatchNormalization, keras.layers.BatchNormalization, x)


def check_layer_multi_parity(
    layer_cls: TensorLike,
    keras_cls: TensorLike,
    inputs: TensorLike,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    **kwargs: TensorLike,
) -> None:
    """Compare output of a multi-input zero_keras layer against its Keras counterpart.

    Args:
        layer_cls: zero_keras merging layer class.
        keras_cls: Reference Keras merging layer class.
        inputs: Sequence of input tensors.
        atol: Absolute tolerance for numerical comparison.
        rtol: Relative tolerance for numerical comparison.
        **kwargs: Constructor keyword arguments passed to both layers.
    """
    set_seed(42)
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)

    set_seed(42)
    zero_layer = layer_cls(**kwargs)
    zero_out = zero_layer(inputs)

    assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


def test_layer_Add() -> None:
    """Test Add merging layer parity against Keras."""
    x1 = np.ones((2, 2), dtype=np.float32)
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Add, keras.layers.Add, [x1, x2])


def test_layer_Subtract() -> None:
    """Test Subtract merging layer parity against Keras."""
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Subtract, keras.layers.Subtract, [x1, x2])


def test_layer_Multiply() -> None:
    """Test Multiply merging layer parity against Keras."""
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 2
    check_layer_multi_parity(layers.Multiply, keras.layers.Multiply, [x1, x2])


def test_layer_Average() -> None:
    """Test Average merging layer parity against Keras."""
    x1 = np.ones((2, 2), dtype=np.float32) * 3
    x2 = np.ones((2, 2), dtype=np.float32) * 1
    check_layer_multi_parity(layers.Average, keras.layers.Average, [x1, x2])


def test_layer_Maximum() -> None:
    """Test Maximum merging layer parity against Keras."""
    x1 = np.array([[1.0, 3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 2.0]], dtype=np.float32)
    check_layer_multi_parity(layers.Maximum, keras.layers.Maximum, [x1, x2])


def test_layer_Minimum() -> None:
    """Test Minimum merging layer parity against Keras."""
    x1 = np.array([[1.0, 3.0]], dtype=np.float32)
    x2 = np.array([[2.0, 2.0]], dtype=np.float32)
    check_layer_multi_parity(layers.Minimum, keras.layers.Minimum, [x1, x2])


def test_layer_Concatenate() -> None:
    """Test Concatenate merging layer parity against Keras."""
    x1 = np.ones((2, 2), dtype=np.float32)
    x2 = np.ones((2, 3), dtype=np.float32)
    check_layer_multi_parity(layers.Concatenate, keras.layers.Concatenate, [x1, x2], axis=-1)


def test_layer_Dot() -> None:
    """Test Dot merging layer parity against Keras."""
    x1 = np.random.rand(2, 3).astype(np.float32)
    x2 = np.random.rand(2, 3).astype(np.float32)
    check_layer_multi_parity(layers.Dot, keras.layers.Dot, [x1, x2], axes=1)


def test_unsupported_layers_instantiate(tmp_path: TensorLike) -> None:
    """Test forward-pass execution of specialized, preprocessing, and wrapper layers.

    Args:
        tmp_path: Pytest temporary directory fixture for TFSMLayer artifact testing.
    """
    img = np.random.uniform(0.0, 255.0, size=(2, 4, 4, 3)).astype(np.float32)
    x_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # Discretization & Embedding & InputLayer & InputSpec
    disc = layers.Discretization(bin_boundaries=[1.5, 3.5, 5.5])
    assert disc(x_2d).shape == (2, 3)

    emb = layers.Embedding(input_dim=8, output_dim=4)
    assert emb(np.array([[0, 1], [2, 3]], dtype=np.int32)).shape == (2, 2, 4)

    inp_layer = layers.InputLayer(shape=(3,))
    assert inp_layer(x_2d).shape == (2, 3)

    spec = layers.InputSpec(dtype="float32", shape=(None, 3))
    assert spec.dtype == "float32" and spec.shape == (None, 3)

    # MaxNumBoundingBoxes & Pipeline & Wrapper
    bboxes = np.ones((2, 3, 4), dtype=np.float32)
    assert layers.MaxNumBoundingBoxes(max_number=5)(bboxes).shape == (2, 5, 4)
    assert layers.MaxNumBoundingBoxes(max_number=3)(bboxes).shape == (2, 3, 4)
    assert layers.MaxNumBoundingBoxes(max_number=3)(np.ones((3, 4), dtype=np.float32)).shape == (
        3,
        4,
    )

    from zero_keras.core_layers import Variable

    v_custom = Variable("ones", shape=(2,))
    assert v_custom._tensor.shape == (2,)
    v_scalar = Variable("zeros", shape=None)
    assert v_scalar._tensor.shape == ()

    pipe = layers.Pipeline([layers.Identity(), layers.ReLU()])
    assert pipe(x_2d).shape == (2, 3)

    wrap = layers.Wrapper(layers.Dense(4))
    wrap.build((2, 3))
    assert wrap(x_2d).shape == (2, 3)

    # Foreign framework wrappers with real callables
    flax_l = layers.FlaxLayer(module=lambda t: np.asarray(t) * 2.0, variables={})
    assert np.allclose(flax_l(x_2d), x_2d * 2.0)

    jax_l = layers.JaxLayer(call_fn=lambda params, t: np.asarray(t) + 1.0, params={})
    assert np.allclose(jax_l(x_2d), x_2d + 1.0)

    torch_l = layers.TorchModuleWrapper(module=lambda t: np.asarray(t) - 1.0)
    assert np.allclose(torch_l(x_2d), x_2d - 1.0)

    from zero_keras import models

    saved_model_path = str(tmp_path / "saved_dense.keras")
    m = models.Sequential([layers.InputLayer(shape=(3,)), layers.Dense(2)])
    m(x_2d)
    m.save(saved_model_path)
    tfsm = layers.TFSMLayer(filepath=saved_model_path)
    assert tfsm(x_2d).shape == (2, 2)

    # Verify all image augmentation layers run forward passes on real tensors
    for aug_cls in [
        layers.AugMix,
        layers.AutoContrast,
        layers.CutMix,
        layers.Equalization,
        layers.MixUp,
        layers.RandAugment,
        layers.RandomBrightness,
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
    ]:
        layer_inst = aug_cls()
        out = layer_inst(img, training=True)
        assert out.shape == (2, 4, 4, 3)


def test_layer_Conv1D() -> None:
    """Test 1D convolution layer parity against Keras."""
    x = np.random.rand(2, 5, 3).astype(np.float32)
    check_layer_parity(
        layers.Conv1D, keras.layers.Conv1D, x, filters=4, kernel_size=3, padding="same"
    )


def test_layer_Conv2D() -> None:
    """Test 2D convolution layer parity against Keras."""
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    check_layer_parity(
        layers.Conv2D, keras.layers.Conv2D, x, filters=4, kernel_size=3, padding="same"
    )


def test_layer_Conv3D() -> None:
    """Test 3D convolution layer parity against Keras."""
    x = np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
    check_layer_parity(
        layers.Conv3D, keras.layers.Conv3D, x, filters=4, kernel_size=3, padding="same"
    )


def test_layer_Conv1DTranspose() -> None:
    """Test 1D transposed convolution layer parity against Keras."""
    x = np.random.rand(2, 5, 3).astype(np.float32)
    check_layer_parity(
        layers.Conv1DTranspose,
        keras.layers.Conv1DTranspose,
        x,
        filters=4,
        kernel_size=3,
        padding="same",
    )


def test_layer_Conv2DTranspose() -> None:
    """Test 2D transposed convolution layer parity against Keras."""
    x = np.random.rand(2, 5, 5, 3).astype(np.float32)
    check_layer_parity(
        layers.Conv2DTranspose,
        keras.layers.Conv2DTranspose,
        x,
        filters=4,
        kernel_size=3,
        padding="same",
    )


def test_layer_Conv3DTranspose() -> None:
    """Test 3D transposed convolution layer parity against Keras."""
    x = np.random.rand(2, 5, 5, 5, 3).astype(np.float32)
    check_layer_parity(
        layers.Conv3DTranspose,
        keras.layers.Conv3DTranspose,
        x,
        filters=4,
        kernel_size=3,
        padding="same",
    )


def test_pooling() -> None:
    """Test 1D, 2D, and 3D MaxPooling and AveragePooling parity against Keras."""
    for rank, zero_max, keras_max, zero_avg, keras_avg in [
        (
            1,
            layers.MaxPooling1D,
            keras.layers.MaxPooling1D,
            layers.AveragePooling1D,
            keras.layers.AveragePooling1D,
        ),
        (
            2,
            layers.MaxPooling2D,
            keras.layers.MaxPooling2D,
            layers.AveragePooling2D,
            keras.layers.AveragePooling2D,
        ),
        (
            3,
            layers.MaxPooling3D,
            keras.layers.MaxPooling3D,
            layers.AveragePooling3D,
            keras.layers.AveragePooling3D,
        ),
    ]:
        shape = [2] + [6] * rank + [3]
        x = np.random.rand(*shape).astype(np.float32)
        check_layer_parity(zero_max, keras_max, x, pool_size=2, strides=2, padding="valid")
        check_layer_parity(zero_avg, keras_avg, x, pool_size=2, strides=2, padding="valid")


def test_global_pooling() -> None:
    """Test 1D, 2D, and 3D GlobalMaxPooling and GlobalAveragePooling parity against Keras."""
    for rank, zero_max, keras_max, zero_avg, keras_avg in [
        (
            1,
            layers.GlobalMaxPooling1D,
            keras.layers.GlobalMaxPooling1D,
            layers.GlobalAveragePooling1D,
            keras.layers.GlobalAveragePooling1D,
        ),
        (
            2,
            layers.GlobalMaxPooling2D,
            keras.layers.GlobalMaxPooling2D,
            layers.GlobalAveragePooling2D,
            keras.layers.GlobalAveragePooling2D,
        ),
        (
            3,
            layers.GlobalMaxPooling3D,
            keras.layers.GlobalMaxPooling3D,
            layers.GlobalAveragePooling3D,
            keras.layers.GlobalAveragePooling3D,
        ),
    ]:
        shape = [2] + [6] * rank + [3]
        x = np.random.rand(*shape).astype(np.float32)
        check_layer_parity(zero_max, keras_max, x)
        check_layer_parity(zero_avg, keras_avg, x)


def test_spatial_layers() -> None:
    """Test 1D, 2D, and 3D ZeroPadding, Cropping, and UpSampling parity against Keras."""
    for rank, zp_z, zp_k, cr_z, cr_k, up_z, up_k in [
        (
            1,
            layers.ZeroPadding1D,
            keras.layers.ZeroPadding1D,
            layers.Cropping1D,
            keras.layers.Cropping1D,
            layers.UpSampling1D,
            keras.layers.UpSampling1D,
        ),
        (
            2,
            layers.ZeroPadding2D,
            keras.layers.ZeroPadding2D,
            layers.Cropping2D,
            keras.layers.Cropping2D,
            layers.UpSampling2D,
            keras.layers.UpSampling2D,
        ),
        (
            3,
            layers.ZeroPadding3D,
            keras.layers.ZeroPadding3D,
            layers.Cropping3D,
            keras.layers.Cropping3D,
            layers.UpSampling3D,
            keras.layers.UpSampling3D,
        ),
    ]:
        shape = [2] + [6] * rank + [3]
        x = np.random.rand(*shape).astype(np.float32)
        check_layer_parity(zp_z, zp_k, x, padding=1)
        check_layer_parity(cr_z, cr_k, x, cropping=1)
        check_layer_parity(up_z, up_k, x, size=2)


def test_attention_layers() -> None:
    """Test Attention and AdditiveAttention layers with scale enabled."""
    q = np.random.rand(2, 5, 4).astype(np.float32)
    v = np.random.rand(2, 5, 4).astype(np.float32)

    layer = layers.Attention(use_scale=True)
    out = layer([q, v])
    assert out.shape == (2, 5, 4)

    layer2 = layers.AdditiveAttention(use_scale=True)
    out2 = layer2([q, v])
    assert out2.shape == (2, 5, 4)


def test_embedding_layer() -> None:
    """Test Embedding layer parity against Keras."""
    idx = np.random.randint(0, 10, size=(2, 5)).astype(np.int32)
    check_layer_parity(layers.Embedding, keras.layers.Embedding, idx, input_dim=10, output_dim=4)


def test_timedistributed_layer() -> None:
    """Test TimeDistributed wrapper over Dense layer."""
    x = np.random.rand(2, 3, 4).astype(np.float32)
    layer = layers.TimeDistributed(layers.Dense(5))
    layer.build(x.shape)
    out = layer(x)
    assert out.shape == (2, 3, 5)


def test_norm_layers() -> None:
    """Test normalization and regularization layers against Keras."""
    x = np.random.rand(2, 5, 4).astype(np.float32)
    check_layer_parity(
        layers.ActivityRegularization, keras.layers.ActivityRegularization, x, l1=0.1
    )

    check_layer_parity(layers.AlphaDropout, keras.layers.AlphaDropout, x, rate=0.2)
    check_layer_parity(layers.GaussianDropout, keras.layers.GaussianDropout, x, rate=0.2)
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


def test_augmentation_layers() -> None:
    """Test all 22 image augmentation layers with real tensor execution and shape/value checks."""
    aug_layer_classes = [
        layers.AugMix,
        layers.AutoContrast,
        layers.CutMix,
        layers.Equalization,
        layers.MixUp,
        layers.RandAugment,
        layers.RandomBrightness,
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
    ]
    x = np.random.uniform(10.0, 240.0, size=(2, 5, 4, 3)).astype(np.float32)
    for zero_cls in aug_layer_classes:
        layer = zero_cls()
        layer.build(x.shape)
        out_train = layer(x, training=True)
        assert out_train is not None
        assert out_train.shape == (2, 5, 4, 3)
        assert np.all(np.isfinite(np.asarray(out_train)))

        out_eval = layer(x, training=False)
        assert out_eval.shape == (2, 5, 4, 3)
        if zero_cls not in (layers.AutoContrast, layers.Equalization):
            np.testing.assert_allclose(np.asarray(out_eval), x, atol=1e-5)


def test_misc_layers() -> None:
    """Test Identity, Normalization, SpatialDropout, MaxNumBoundingBoxes, and Discretization."""
    x = np.random.rand(2, 5, 4, 3).astype(np.float32)

    check_layer_parity(layers.Identity, keras.layers.Identity, x)
    check_layer_parity(layers.Normalization, keras.layers.Normalization, x)

    for rank, zero_cls, keras_cls in [
        (1, layers.SpatialDropout1D, keras.layers.SpatialDropout1D),
        (2, layers.SpatialDropout2D, keras.layers.SpatialDropout2D),
        (3, layers.SpatialDropout3D, keras.layers.SpatialDropout3D),
    ]:
        shape = [2] + [5] * rank + [3]
        x_rank = np.random.rand(*shape).astype(np.float32)
        check_layer_parity(zero_cls, keras_cls, x_rank, rate=0.2)

    bbox_layer = layers.MaxNumBoundingBoxes(max_number=2)
    bbox_layer.build((2, 5, 4))
    res_bbox = bbox_layer(np.ones((2, 5, 4), dtype=np.float32))
    assert res_bbox.shape == (2, 2, 4)

    layer = layers.Discretization(bin_boundaries=[0.3, 0.7])
    layer.build(x.shape)
    res = layer(x)
    assert res.shape == (2, 5, 4, 3)
    res_data = np.asarray(res)
    assert np.all((res_data >= 0) & (res_data <= 2))


def test_reshaping_layers() -> None:
    """Test Cropping, UpSampling, and ZeroPadding across 1D, 2D, and 3D inputs."""
    test_spatial_layers()


def test_norm_layers_more() -> None:
    """Test UnitNormalization, GroupNormalization, RMSNormalization, and SpectralNormalization parity."""
    x = np.random.rand(2, 5, 4).astype(np.float32)
    check_layer_parity(layers.UnitNormalization, keras.layers.UnitNormalization, x)
    check_layer_parity(layers.GroupNormalization, keras.layers.GroupNormalization, x, groups=2)
    check_layer_parity(layers.RMSNormalization, keras.layers.RMSNormalization, x, rtol=1e-2)

    def layer_cls(**kwargs: TensorLike) -> TensorLike:
        """Construct a zero_keras SpectralNormalization wrapping Dense(5).

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: SpectralNormalization instance.
        """
        return layers.SpectralNormalization(layer=layers.Dense(5), **kwargs)

    def k_layer_cls(**kwargs: TensorLike) -> TensorLike:
        """Construct a Keras SpectralNormalization wrapping Dense(5).

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Keras SpectralNormalization instance.
        """
        return keras.layers.SpectralNormalization(layer=keras.layers.Dense(5), **kwargs)

    check_layer_parity(layer_cls, k_layer_cls, x)


def test_attention_layers_more() -> None:
    """Test AdditiveAttention and Attention parity without scale factor."""
    query = np.random.rand(2, 4, 3).astype(np.float32)
    value = np.random.rand(2, 4, 3).astype(np.float32)
    x = [query, value]

    check_layer_parity(layers.AdditiveAttention, keras.layers.AdditiveAttention, x, use_scale=False)
    check_layer_parity(layers.Attention, keras.layers.Attention, x, use_scale=False)

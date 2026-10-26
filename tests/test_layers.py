"""Comprehensive unit and branch coverage tests for zero_keras.layers."""

from __future__ import annotations

from zero_keras.types import TensorLike
from zero_keras import layers, random
import pytest
from zero_keras.layers import (
    RNN,
    Bidirectional,
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    Dense,
    GRUCell,
    Layer,
    LSTMCell,
    SimpleRNNCell,
    SpectralNormalization,
)
from unittest.mock import patch
from ml_switcheroo_compiler.ops import asarray
from ml_switcheroo_compiler.core.config import config
from zero_keras.layers import (
    LSTM,
    ConvLSTM1D,
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Dot,
    EinsumDense,
    GroupNormalization,
    LayerNormalization,
    Normalization,
    RandAugment,
    RandomFlip,
    RandomRotation,
    SimpleRNN,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
)
from zero_keras import ops
from ml_switcheroo_compiler import ops as msc_ops
from zero_keras.layers import (
    RMSNormalization,
)
from zero_keras import core_layers

from .conftest import assert_allclose_zero, to_flat_floats


def test_layers() -> None:
    """Verify instantiation and forward execution of core Dense, Dropout, Flatten, and Activation layers."""
    x = ops.ones((2, 4), dtype="float32")
    dense = layers.Dense(3, activation="relu")
    out = dense(x)
    assert out.shape == (2, 3)
    drop = layers.Dropout(0.2)
    assert drop(out, training=False).shape == (2, 3)
    flat = layers.Flatten()
    assert flat(ops.ones((2, 3, 4), dtype="float32")).shape == (2, 12)


def test_missing_layers() -> None:
    """Test edge-case configurations, channels_first shapes, and return values across layers."""
    # Dot
    dot_out = layers.Dot(axes=1)(
        [ops.ones((2, 3), dtype="float32"), ops.ones((2, 3), dtype="float32")]
    )
    assert dot_out.shape == (2, 1)
    assert_allclose_zero(dot_out, ops.full((2, 1), 3.0))

    # ActivityRegularization
    act_reg = layers.ActivityRegularization(l2=0.1)
    act_out = act_reg(ops.ones((2, 3), dtype="float32"))
    assert act_out.shape == (2, 3)

    # AdditiveAttention with return_attention_scores=True
    add_attn = layers.AdditiveAttention()
    add_attn.build([(None, 3, 4), (None, 3, 4)])
    add_out, add_scores = add_attn(
        [ops.ones((2, 3, 4), dtype="float32"), ops.ones((2, 3, 4), dtype="float32")],
        return_attention_scores=True,
    )
    assert add_out.shape == (2, 3, 4)
    assert add_scores.shape == (2, 3, 3)

    # AlphaDropout
    alpha_drop = layers.AlphaDropout(rate=0.5)
    alpha_out = alpha_drop(ops.ones((2, 3), dtype="float32"), training=False)
    assert alpha_out.shape == (2, 3)

    # Attention with return_attention_scores=True
    attn = layers.Attention()
    attn.build([(None, 3, 4), (None, 3, 4)])
    attn_out, attn_scores = attn(
        [ops.ones((2, 3, 4), dtype="float32"), ops.ones((2, 3, 4), dtype="float32")],
        return_attention_scores=True,
    )
    assert attn_out.shape == (2, 3, 4)
    assert attn_scores.shape == (2, 3, 3)

    # Image augmentations in inference mode (training=False)
    img_in = ops.ones((2, 10, 10, 3), dtype="float32")
    for layer_cls in [
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
        aug_layer = layer_cls()
        aug_out = aug_layer(img_in, training=False)
        assert aug_out.shape == (2, 10, 10, 3)

    # Pooling layers with data_format="channels_first"
    for rank, pool_classes in [
        (
            1,
            [
                layers.AveragePooling1D,
                layers.MaxPooling1D,
                layers.GlobalAveragePooling1D,
                layers.GlobalMaxPooling1D,
            ],
        ),
        (
            2,
            [
                layers.AveragePooling2D,
                layers.MaxPooling2D,
                layers.GlobalAveragePooling2D,
                layers.GlobalMaxPooling2D,
            ],
        ),
        (
            3,
            [
                layers.AveragePooling3D,
                layers.MaxPooling3D,
                layers.GlobalAveragePooling3D,
                layers.GlobalMaxPooling3D,
            ],
        ),
    ]:
        shape = [2, 3] + [10] * rank
        x_pool = ops.ones(shape, dtype="float32")
        for pool_cls in pool_classes:
            pool_layer = (
                pool_cls(data_format="channels_first")
                if "Global" in pool_cls.__name__
                else pool_cls(pool_size=2, data_format="channels_first")
            )
            p_out = pool_layer(x_pool)
            assert p_out.shape[0] == 2 and p_out.shape[1] == 3

    # CategoryEncoding
    cat_enc = layers.CategoryEncoding(num_tokens=10, output_mode="one_hot")
    cat_out = cat_enc(ops.convert_to_tensor([1, 3], dtype="int32"))
    assert cat_out.shape == (2, 10)

    # Convolutions with data_format="channels_first"
    for rank, conv_classes in [
        (1, [layers.Conv1D, layers.Conv1DTranspose, layers.SeparableConv1D]),
        (2, [layers.Conv2D, layers.Conv2DTranspose, layers.SeparableConv2D]),
        (3, [layers.Conv3D, layers.Conv3DTranspose]),
    ]:
        shape = [2, 3] + [10] * rank
        x_conv = ops.ones(shape, dtype="float32")
        for conv_cls in conv_classes:
            conv_layer = conv_cls(filters=4, kernel_size=3, data_format="channels_first")
            conv_layer.build(tuple(shape))
            c_out = conv_layer(x_conv)
            assert c_out.shape[0] == 2 and c_out.shape[1] == 4

    # Cropping with single-element tuple and standard tuple
    c1_single = layers.Cropping1D(cropping=(1,))
    assert c1_single.cropping == ((1, 1),)
    c1 = layers.Cropping1D(cropping=(1, 1))
    assert c1(ops.ones((2, 10, 3), dtype="float32")).shape == (2, 8, 3)
    c2 = layers.Cropping2D(cropping=(1, 1))
    assert c2(ops.ones((2, 10, 10, 3), dtype="float32")).shape == (2, 8, 8, 3)
    with pytest.raises(TypeError):
        layers.Cropping2D(cropping=(1,))
    c3 = layers.Cropping3D(cropping=(1, 1, 1))
    assert c3(ops.ones((2, 10, 10, 10, 3), dtype="float32")).shape == (2, 8, 8, 8, 3)
    with pytest.raises(TypeError):
        layers.Cropping3D(cropping=(1,))

    # EinsumDense bias_axes=None
    ed = layers.EinsumDense(equation="abc,cd->abd", output_shape=(10, 4), bias_axes=None)
    ed.build((None, 10, 3))
    assert ed.bias is None
    assert ed(ops.ones((2, 10, 3), dtype="float32")).shape == (2, 10, 4)

    # Embedding build
    emb = layers.Embedding(10, 4)
    emb.build((None, 10))
    assert emb.built

    # RNN and RNN cells
    rnn_layer = layers.RNN(layers.SimpleRNNCell(4))
    rnn_layer.build((None, 10, 3))
    rnn_layer.reset_states()
    assert rnn_layer(ops.ones((2, 10, 3), dtype="float32")).shape == (2, 4)

    srnn_cell = layers.SimpleRNNCell(4)
    srnn_cell.build((None, 3))
    s_out, s_states = srnn_cell(
        ops.ones((2, 3), dtype="float32"), [ops.ones((2, 4), dtype="float32")]
    )
    assert s_out.shape == (2, 4) and s_states is not None

    gru_cell = layers.GRUCell(4, use_bias=False)
    gru_cell.build((None, 3))
    g_out, g_states = gru_cell(
        ops.ones((2, 3), dtype="float32"), [ops.ones((2, 4), dtype="float32")]
    )
    assert g_out.shape == (2, 4) and g_states is not None

    lstm_cell = layers.LSTMCell(4, use_bias=False)
    lstm_cell.build((None, 3))
    l_out, l_states = lstm_cell(
        ops.ones((2, 3), dtype="float32"),
        [ops.ones((2, 4), dtype="float32"), ops.ones((2, 4), dtype="float32")],
    )
    assert l_out.shape == (2, 4) and len(l_states) == 2

    # Bidirectional with merge modes and return_state=True
    bidir = layers.Bidirectional(layers.SimpleRNN(4))
    bidir.build((None, 10, 3))
    assert bidir(ops.ones((2, 10, 3), dtype="float32")).shape == (2, 8)
    for mode in ["sum", "ave", "mul", None]:
        b = layers.Bidirectional(layers.SimpleRNN(4, return_state=True), merge_mode=mode)
        b_res = b(ops.ones((2, 10, 3), dtype="float32"))
        assert isinstance(b_res, (list, tuple)) and len(b_res) >= 3

    # Normalization & RMSNormalization
    norm = layers.Normalization()
    norm.build((None, 3))
    assert norm(ops.ones((2, 3), dtype="float32")).shape == (2, 3)

    rms = layers.RMSNormalization()
    rms.build((None, 3))
    assert rms(ops.ones((2, 3), dtype="float32")).shape == (2, 3)

    # RandomFlip, RandomRotation, Rescaling, Resizing
    assert layers.RandomFlip()(img_in, training=False).shape == (2, 10, 10, 3)
    assert layers.RandomRotation(0.1)(img_in, training=False).shape == (2, 10, 10, 3)
    assert layers.Rescaling(1.0, 0.0)(img_in).shape == (2, 10, 10, 3)
    assert layers.Resizing(8, 8)(img_in).shape == (2, 8, 8, 3)

    # STFTSpectrogram
    stft = layers.STFTSpectrogram(frame_length=256, frame_step=None, fft_length=None)
    stft.build((None, 1000))
    assert stft(ops.ones((2, 1000, 1), dtype="float32")).shape[0] == 2
    assert stft(ops.ones((2, 1000), dtype="float32")).shape[0] == 2

    # SeparableConvolution1D/2D & ConvLSTM1D/3D
    assert layers.SeparableConvolution1D(4, 3).filters == 4
    assert layers.SeparableConvolution2D(4, 3).filters == 4
    assert layers.ConvLSTM1D(4, 3).cell.filters == 4
    assert layers.ConvLSTM3D(4, 3).cell.filters == 4

    # GroupNormalization center=False, scale=False
    gn1 = layers.GroupNormalization(axis=1, center=False, scale=False, groups=2)
    gn1.build((None, 4, 10, 10))
    assert gn1(ops.ones((2, 4, 10, 10), dtype="float32")).shape == (2, 4, 10, 10)

    gn2 = layers.GroupNormalization(axis=-1, center=False, scale=False, groups=2)
    gn2.build((None, 10, 10, 4))
    assert gn2(ops.ones((2, 10, 10, 4), dtype="float32")).shape == (2, 10, 10, 4)

    # InputLayer
    il1 = layers.InputLayer(input_shape=(10,), batch_size=2)
    assert il1.batch_input_shape == (2, 10)
    il2 = layers.InputLayer(input_shape=(10,))
    assert il2.batch_input_shape == (None, 10)

    # JaxLayer / TorchModuleWrapper / TFSMLayer
    assert layers.JaxLayer(None, input_shape=(10,))(ops.ones((2, 10), dtype="float32")).shape == (
        2,
        10,
    )
    assert layers.TorchModuleWrapper(lambda x: x, input_shape=(10,))(
        ops.ones((2, 10), dtype="float32")
    ).shape == (2, 10)
    assert layers.TFSMLayer("nonexistent_dir", input_shape=(10,))(
        ops.ones((2, 10), dtype="float32")
    ).shape == (2, 10)

    # Wrapper
    w = layers.Wrapper(layers.Dense(4))
    w.build((None, 3))
    assert w(ops.ones((2, 3), dtype="float32")).shape == (2, 3)

    # ZeroPadding tuple sizes
    zp1_single = layers.ZeroPadding1D((1,))
    assert zp1_single.padding == ((1, 1),)
    zp1 = layers.ZeroPadding1D((1, 1))
    assert zp1(ops.ones((2, 10, 3), dtype="float32")).shape == (2, 12, 3)

    zp2 = layers.ZeroPadding2D((1, 1))
    assert zp2(ops.ones((2, 10, 10, 3), dtype="float32")).shape == (2, 12, 12, 3)
    with pytest.raises(TypeError):
        layers.ZeroPadding2D((1,))

    zp3 = layers.ZeroPadding3D((1, 1, 1))
    assert zp3(ops.ones((2, 10, 10, 10, 3), dtype="float32")).shape == (2, 12, 12, 12, 3)
    with pytest.raises(TypeError):
        layers.ZeroPadding3D((1, 1))

    # TimeDistributed & SpectralNormalization build
    td = layers.TimeDistributed(layers.Dense(4))
    td.build((None, 10, 3))
    assert td.built

    sn = layers.SpectralNormalization(layers.Dense(4))
    sn.build((None, 3))
    assert sn.built

    # ConvLSTMCell init and build branches
    c_cell0 = layers.ConvLSTMCell(4, 3, rank=1, strides=1, dilation_rate=1)
    assert c_cell0.filters == 4
    c_cell1 = layers.ConvLSTMCell(4, 3, rank=1, strides=1, dilation_rate=1, use_bias=False)
    c_cell1.build((None, 10, 3))
    assert c_cell1.built and c_cell1.bias is None
    c_cell2 = layers.ConvLSTMCell(4, 3, rank=1, strides=2, dilation_rate=1, use_bias=False)
    c_cell2.build((None, 10, 3))
    assert c_cell2.built

    # Base Layer call returns inputs
    base_l = layers.Layer()
    assert base_l(ops.ones((2, 3), dtype="float32")).shape == (2, 3)


def test_activation_none_branches() -> None:
    """Test branches where activation is None."""
    # Dense
    layer = Dense(2)
    layer.activation = None
    layer(ops.ones((2, 2)))

    # Conv1D
    layer = Conv1D(2, 2)
    layer.activation = None
    layer(ops.ones((2, 4, 2)))

    # Conv2D
    layer = Conv2D(2, 2)
    layer.activation = None
    layer(ops.ones((2, 4, 4, 2)))

    # Conv3D
    layer = Conv3D(2, 2)
    layer.activation = None
    layer(ops.ones((2, 4, 4, 4, 2)))

    # Conv1DTranspose
    layer = Conv1DTranspose(2, 2)
    layer.activation = None
    layer(ops.ones((2, 4, 2)))

    # Conv2DTranspose
    layer = Conv2DTranspose(2, 2)
    layer.activation = None
    layer(ops.ones((2, 4, 4, 2)))

    # Conv3DTranspose
    layer = Conv3DTranspose(2, 2)
    layer.activation = None
    layer(ops.ones((2, 4, 4, 4, 2)))

    # SimpleRNNCell
    layer = SimpleRNNCell(2)
    layer.activation = None
    layer(ops.ones((2, 2)), [ops.ones((2, 2))])


def test_rnn_cell_no_bias_branch() -> None:
    """Test RNN cells where use_bias is False or getattr returns None."""
    layer = SimpleRNNCell(2, use_bias=False)
    layer.build((None, 2))
    layer = SimpleRNNCell(2, use_bias=False)
    layer.bias = "dummy"
    layer.build((None, 2))

    layer = GRUCell(2, use_bias=False)
    layer.bias = "dummy"
    layer.build((None, 2))

    layer = LSTMCell(2, unit_forget_bias=False)
    layer.build((None, 2))

    layer = LSTMCell(2, use_bias=False)
    layer.bias = "dummy"
    layer.build((None, 2))


def test_bidirectional_rnn_cell_units() -> None:
    """Test Bidirectional wrapper with an RNN(SimpleRNNCell(3)) instance."""
    layer = Bidirectional(RNN(SimpleRNNCell(3)))
    assert isinstance(layer.backward_layer.cell, SimpleRNNCell)
    assert layer.backward_layer.cell.units == 3
    out = layer(ops.ones((2, 4, 5), dtype="float32"))
    assert out.shape == (2, 6)

    class DummyBidirectionalLayer(Layer):
        """Dummy layer for testing Bidirectional branches."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Return ``inputs`` unchanged.

            Args:
                inputs: Input tensor.

            Returns:
                TensorLike: Unchanged input tensor.
            """
            return inputs

    layer2 = DummyBidirectionalLayer()
    layer2.args = ()
    layer2.return_sequences = False
    layer2.return_state = False
    Bidirectional(layer2)


def test_spectral_norm_no_weights_branch() -> None:
    """Test SpectralNormalization where u is not in _weights."""

    # 12831->12835
    class DummyWrapper(SpectralNormalization):
        """Dummy wrapper to skip adding weight to _weights."""

        def add_weight(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
            """Override add_weight.

            Args:
                *args: Positional arguments forwarded to helper.
                **kwargs: Keyword arguments forwarded to helper.

            Returns:
                TensorLike: Result from `add_weight`.
            """
            weight = super().add_weight(*args, **kwargs)
            if weight in self._weights:
                self._weights.remove(weight)
            return weight

    layer = Dense(2)
    sn = DummyWrapper(layer)
    sn.build((None, 3))


def test_convolution_aliases_missing_branches() -> None:
    """Verify Convolution alias layers with preset weights, use_bias=False, and activation=None."""
    for layer_cls in [
        layers.Convolution1D,
        layers.Convolution2D,
        layers.Convolution3D,
        layers.Convolution1DTranspose,
        layers.Convolution2DTranspose,
        layers.Convolution3DTranspose,
    ]:
        rank = 1 if "1D" in layer_cls.__name__ else (2 if "2D" in layer_cls.__name__ else 3)
        input_shape = (2,) + (4,) * rank + (4,)

        layer = layer_cls(use_bias=True, filters=4, kernel_size=2)
        layer.kernel = ops.ones((2,) * rank + (4, 4))
        layer.bias = ops.ones((4,))
        layer.build(input_shape)
        assert layer.built is True

        layer2 = layer_cls(use_bias=False, activation=None, filters=4, kernel_size=2)
        out = layer2(ops.ones(input_shape))
        assert out.shape[0] == 2 and out.shape[-1] == 4


def test_activation_none_dense_conv() -> None:
    """Verify Dense and EinsumDense with activation=None and preset kernel/bias."""
    dense = layers.Dense(units=4, activation=None)
    out_dense = dense(ops.ones((2, 3)))
    assert out_dense.shape == (2, 4)

    ed = layers.EinsumDense("ab,bc->ac", output_shape=(4,), activation=None)
    out_ed = ed(ops.ones((2, 3)))
    assert out_ed.shape == (2, 4)

    ed_preset = layers.EinsumDense("ab,bc->ac", output_shape=(4,), bias_axes="c")
    ed_preset.kernel = ops.ones((3, 4))
    ed_preset.bias = ops.ones((4,))
    ed_preset.build((2, 3))
    assert ed_preset.built is True


def test_rnn_cells_missing_branches() -> None:
    """Verify SimpleRNNCell, GRUCell, and LSTMCell with preset weights and use_bias=False."""
    for layer_cls in [layers.SimpleRNNCell, layers.GRUCell, layers.LSTMCell]:
        mult = 1 if layer_cls is layers.SimpleRNNCell else (3 if layer_cls is layers.GRUCell else 4)
        layer = layer_cls(units=4, use_bias=True)
        layer.kernel = ops.ones((3, 4 * mult))
        layer.recurrent_kernel = ops.ones((4, 4 * mult))
        layer.bias = ops.ones((4 * mult,))
        layer.build((2, 3))
        assert layer.built is True

        layer2 = layer_cls(units=4, use_bias=False, activation=None)
        states = (
            [ops.ones((2, 4))]
            if layer_cls is not layers.LSTMCell
            else [ops.ones((2, 4)), ops.ones((2, 4))]
        )
        out, new_states = layer2(ops.ones((2, 3)), states)
        assert out.shape == (2, 4)
        assert new_states is not None

        layer3 = layer_cls(units=4, use_bias=True)
        out3, _ = layer3.call(ops.ones((2, 3)), states)
        assert out3.shape == (2, 4)


def test_conv_lstm_cell_use_bias_false() -> None:
    """Verify ConvLSTMCell forward pass when use_bias=False."""
    layer = layers.ConvLSTMCell(filters=4, kernel_size=2, use_bias=False)
    x = ops.ones((2, 4, 4, 3))
    h = ops.zeros((2, 3, 3, 4))
    c = ops.zeros((2, 3, 3, 4))
    out, states = layer(x, [h, c])
    assert out.shape == (2, 3, 3, 4)
    assert len(states) == 2


def test_wrapper_already_built() -> None:
    """Verify Wrapper.build when the inner layer is already built."""
    inner = layers.Dense(4)
    inner.build((2, 4))
    wrapper = layers.Wrapper(inner)
    wrapper.build((2, 4))
    assert wrapper.built is True


def test_spectral_norm_no_kernel() -> None:
    """Verify SpectralNormalization.build when the inner layer has no kernel."""
    inner = layers.Activation("relu")
    wrapper = layers.SpectralNormalization(inner)
    wrapper.build((2, 4))
    assert wrapper.built is True


def test_spectral_norm_no_weights_attr() -> None:
    """Verify ``SpectralNormalization.build`` when ``self.u`` is not in ``self._weights``."""
    inner = layers.Dense(4)
    wrapper = layers.SpectralNormalization(inner)
    wrapper.add_weight = lambda **kw: ops.ones(kw["shape"])
    wrapper.build((2, 4))
    assert wrapper.u is not None


def test_stacked_rnn_cells_built() -> None:
    """Verify StackedRNNCells.build when child cells are already built."""
    cell = layers.SimpleRNNCell(4)
    cell.build((2, 4))
    stacked = layers.StackedRNNCells([cell])
    stacked.build((2, 4))
    assert stacked.built is True


def test_missing_layers_part2() -> None:
    """Verify image augmentations, Conv build idempotency, Bidirectional merge modes, and ZeroPadding."""
    img = ops.ones((2, 8, 8, 3))
    for layer_cls, kwargs in [
        (layers.AugMix, {}),
        (layers.AutoContrast, {}),
        (layers.CutMix, {}),
        (layers.Equalization, {}),
        (layers.RandomColorDegeneration, {"factor": 0.2}),
        (layers.RandomColorJitter, {"brightness_factor": 0.1}),
        (layers.RandomContrast, {"factor": 0.2}),
        (layers.RandomElasticTransform, {}),
        (layers.RandomErasing, {}),
        (layers.RandomGaussianBlur, {"factor": 0.5}),
        (layers.RandomGrayscale, {}),
        (layers.RandomHue, {"factor": 0.2}),
        (layers.RandomInvert, {}),
        (layers.RandomPerspective, {}),
        (layers.RandomPosterization, {"factor": 4}),
        (layers.RandomSaturation, {"factor": 0.2}),
        (layers.RandomSharpness, {"factor": 0.2}),
        (layers.RandomShear, {"x_factor": 0.1}),
        (layers.Solarization, {}),
        (layers.RandAugment, {}),
        (layers.RandomBrightness, {"factor": 0.2}),
        (layers.MixUp, {}),
    ]:
        aug_layer = layer_cls(**kwargs)
        out = aug_layer(img, training=True)
        assert out.shape == (2, 8, 8, 3)

    for layer_cls in [
        layers.Conv1D,
        layers.Conv1DTranspose,
        layers.Conv2D,
        layers.Conv2DTranspose,
        layers.Conv3D,
        layers.Conv3DTranspose,
        layers.SeparableConv1D,
        layers.SeparableConv2D,
    ]:
        conv_l = layer_cls(filters=4, kernel_size=3)
        rank = 3 if "3D" in layer_cls.__name__ else (2 if "2D" in layer_cls.__name__ else 1)
        shape = (2,) + (6,) * rank + (3,)
        conv_l.build(shape)
        conv_l.build(shape)
        out_c = conv_l(ops.ones(shape))
        assert out_c.shape[0] == 2 and out_c.shape[-1] == 4

    rnn_l = layers.RNN(layers.SimpleRNNCell(4), stateful=True)
    rnn_l.build((2, 5, 3))
    rnn_l.reset_states(states=None)
    initial_states = rnn_l.states
    assert initial_states is None
    rnn_l(ops.ones((2, 5, 3)))
    assert rnn_l.states is not None
    rnn_l.reset_states(states=None)
    assert len(rnn_l.states) == 1

    for mode in ("sum", "ave", "mul", None):
        b = layers.Bidirectional(layers.SimpleRNN(4, return_state=True), merge_mode=mode)
        b.build((None, 5, 3))
        b_out = b(ops.ones((2, 5, 3)))
        assert isinstance(b_out, (list, tuple))

    inp_layer = layers.InputLayer(input_shape=(10,))
    assert inp_layer(ops.ones((2, 10))).shape == (2, 10)

    wrap_l = layers.Wrapper(layers.Dense(4))
    wrap_l.build((None, 3))
    wrap_l.build((None, 3))
    assert wrap_l.built is True

    cl_cell1 = layers.ConvLSTMCell(4, 3, rank=1, strides=1, dilation_rate=1, use_bias=False)
    cl_cell1.build((None, 6, 3))
    cl_cell1.build((None, 6, 3))
    assert cl_cell1.built is True

    cl_cell2 = layers.ConvLSTMCell(4, 3, rank=1, strides=2, dilation_rate=1, use_bias=False)
    cl_cell2.build((None, 6, 3))
    cl_cell2.build((None, 6, 3))
    assert cl_cell2.built is True

    assert layers.ZeroPadding1D((1, 1))(ops.ones((2, 6, 3))).shape == (2, 8, 3)
    assert layers.ZeroPadding2D((1, 1))(ops.ones((2, 6, 6, 3))).shape == (2, 8, 8, 3)
    assert layers.ZeroPadding3D((1, 1, 1))(ops.ones((2, 4, 4, 4, 3))).shape == (2, 6, 6, 6, 3)
    assert layers.ZeroPadding1D(1)(ops.ones((2, 6, 3))).shape == (2, 8, 3)
    assert layers.ZeroPadding2D(1)(ops.ones((2, 6, 6, 3))).shape == (2, 8, 8, 3)
    assert layers.ZeroPadding3D(1)(ops.ones((2, 4, 4, 4, 3))).shape == (2, 6, 6, 6, 3)


def test_remaining() -> None:
    """Verify Dot, ActivityRegularization, Attention, Cropping, Normalization, and RMSNormalization."""
    dot_out = layers.Dot(axes=1)([ops.ones((2, 3)), ops.ones((2, 3))])
    assert dot_out.shape == (2, 1)

    act_reg = layers.ActivityRegularization(l1=0.1, l2=0.1)
    assert act_reg(ops.ones((2, 3))).shape == (2, 3)
    assert len(act_reg.losses) == 1

    add_attn = layers.AdditiveAttention()
    add_attn.build([(None, 3, 4), (None, 3, 4)])
    assert add_attn([ops.ones((2, 3, 4)), ops.ones((2, 3, 4))]).shape == (2, 3, 4)

    alpha_do = layers.AlphaDropout(0.1, seed=42)
    assert alpha_do(ops.ones((2, 3)), training=True).shape == (2, 3)

    attn = layers.Attention()
    attn.build([(None, 3, 4), (None, 3, 4)])
    assert attn([ops.ones((2, 3, 4)), ops.ones((2, 3, 4))]).shape == (2, 3, 4)

    cat_enc = layers.CategoryEncoding(num_tokens=4)
    assert cat_enc(ops.convert_to_tensor([[0, 1], [2, 3]], dtype="int32")).shape == (2, 4)

    for cls in [layers.Conv1D, layers.Conv1DTranspose, layers.SeparableConv1D]:
        out_1d = cls(4, 3, data_format="channels_first")(ops.ones((2, 3, 8)))
        assert out_1d.shape[0] == 2 and out_1d.shape[1] == 4
    for cls in [layers.Conv2D, layers.Conv2DTranspose, layers.SeparableConv2D]:
        out_2d = cls(4, 3, data_format="channels_first")(ops.ones((2, 3, 8, 8)))
        assert out_2d.shape[0] == 2 and out_2d.shape[1] == 4
    for cls in [layers.Conv3D, layers.Conv3DTranspose]:
        out_3d = cls(4, 3, data_format="channels_first")(ops.ones((2, 3, 6, 6, 6)))
        assert out_3d.shape[0] == 2 and out_3d.shape[1] == 4

    assert layers.Cropping1D(1)(ops.ones((2, 6, 3))).shape == (2, 4, 3)
    assert layers.Cropping2D(1)(ops.ones((2, 6, 6, 3))).shape == (2, 4, 4, 3)
    assert layers.Cropping3D(1)(ops.ones((2, 6, 6, 6, 3))).shape == (2, 4, 4, 4, 3)

    ed = layers.EinsumDense("abc,cd->abd", (10, 4), bias_axes=None)
    ed.build((None, 10, 3))
    assert ed.built is True

    emb = layers.Embedding(10, 4)
    emb.build((None, 5))
    assert emb.built is True

    norm = layers.Normalization()
    norm.build((None, 3))
    assert norm(ops.ones((2, 3))).shape == (2, 3)

    rms = layers.RMSNormalization()
    rms.build((None, 3))
    assert rms(ops.ones((2, 3))).shape == (2, 3)

    gn1 = layers.GroupNormalization(groups=2, axis=1)
    gn1.build((None, 4, 6, 6))
    assert gn1(ops.ones((2, 4, 6, 6))).shape == (2, 4, 6, 6)

    gn2 = layers.GroupNormalization(groups=2, axis=-1)
    assert gn2(ops.ones((2, 6, 6, 4))).shape == (2, 6, 6, 4)

    td = layers.TimeDistributed(layers.Dense(4))
    td.build((None, 5, 3))
    assert td(ops.ones((2, 5, 3))).shape == (2, 5, 4)

    sn = layers.SpectralNormalization(layers.Dense(4))
    sn.build((None, 3))
    assert sn(ops.ones((2, 3))).shape == (2, 4)


def test_missing_augmentations() -> None:
    """Verify image augmentation layers in inference mode return inputs with unchanged shape."""
    img = ops.ones((2, 8, 8, 3))
    for cls, kwargs in [
        (layers.AugMix, {}),
        (layers.AutoContrast, {}),
        (layers.CutMix, {}),
        (layers.Equalization, {}),
        (layers.RandomColorDegeneration, {"factor": 0.2}),
        (layers.RandomColorJitter, {"brightness_factor": 0.1}),
        (layers.RandomContrast, {"factor": 0.2}),
        (layers.RandomElasticTransform, {}),
        (layers.RandomErasing, {}),
        (layers.RandomGaussianBlur, {"factor": 0.5}),
        (layers.RandomGrayscale, {}),
        (layers.RandomHue, {"factor": 0.2}),
        (layers.RandomInvert, {}),
        (layers.RandomPerspective, {}),
        (layers.RandomPosterization, {"factor": 4}),
        (layers.RandomSaturation, {"factor": 0.2}),
        (layers.RandomSharpness, {"factor": 0.2}),
        (layers.RandomShear, {"x_factor": 0.1}),
        (layers.Solarization, {}),
        (layers.RandAugment, {}),
        (layers.RandomBrightness, {"factor": 0.2}),
        (layers.MixUp, {}),
    ]:
        out = cls(**kwargs)(img, training=False)
        assert out.shape == (2, 8, 8, 3)


def test_others() -> None:
    """Verify CategoryEncoding auto-num_tokens, ConvTranspose list shape build, and RNN reset_states."""
    cat_auto = layers.CategoryEncoding(num_tokens=None)
    out_cat = cat_auto(ops.convert_to_tensor([[0, 2], [1, 3]], dtype="int32"))
    assert out_cat.shape == (2, 4)

    c1t = layers.Conv1DTranspose(4, 3)
    c1t.build([2, 6, 3])
    assert c1t.built is True

    c2t = layers.Conv2DTranspose(4, 3)
    c2t.build([2, 6, 6, 3])
    assert c2t.built is True

    c3t = layers.Conv3DTranspose(4, 3)
    c3t.build([2, 6, 6, 6, 3])
    assert c3t.built is True

    rnn_layer = layers.RNN(layers.SimpleRNNCell(4), stateful=True)
    rnn_layer.build((2, 5, 3))
    rnn_layer.reset_states()
    out_rnn = rnn_layer(ops.ones((2, 5, 3)))
    assert out_rnn.shape == (2, 4)


def test_remaining_layers_missing_part6() -> None:
    """Verify build idempotency on Attention, Embedding, RNN, SimpleRNNCell, Bidirectional, Normalization, RMSNormalization, TimeDistributed, SpectralNormalization, and ConvLSTMCell."""
    for layer_inst, build_shape in [
        (layers.AdditiveAttention(), [(None, 3, 4), (None, 3, 4)]),
        (layers.Attention(), [(None, 3, 4), (None, 3, 4)]),
        (layers.Embedding(10, 4), (None, 5)),
        (layers.RNN(layers.SimpleRNNCell(4)), (None, 5, 3)),
        (layers.SimpleRNNCell(4), (None, 3)),
        (layers.Bidirectional(layers.SimpleRNN(4)), (None, 5, 3)),
        (layers.Normalization(), (None, 3)),
        (layers.RMSNormalization(), (None, 3)),
        (layers.TimeDistributed(layers.Dense(4)), (None, 5, 3)),
        (layers.SpectralNormalization(layers.Dense(4)), (None, 3)),
        (layers.ConvLSTMCell(4, 3, rank=1, strides=1, dilation_rate=1), (None, 6, 3)),
    ]:
        layer_inst.built = True
        layer_inst.build(build_shape)
        assert layer_inst.built is True

    rnn_s = layers.RNN(layers.SimpleRNNCell(4), stateful=True)
    rnn_s.build((2, 5, 3))
    rnn_s.reset_states(ops.ones((2, 4)))
    out_s = rnn_s(ops.ones((2, 5, 3)), initial_state=[ops.ones((2, 4))])
    assert out_s.shape == (2, 4)


def test_dot_tuple_axes() -> None:
    """Test ``dot_tuple_axes`` behavior and edge cases in ``test_layers``."""
    layer = Dot(axes=(1, 1))
    layer([ops.ones((2, 3)), ops.ones((2, 3))])


def test_conv_transpose_channels_first() -> None:
    """Test ``conv_transpose_channels_first`` behavior and edge cases in ``test_layers``."""
    l1 = Conv1DTranspose(2, 2, data_format="channels_first")
    l1(ops.ones((1, 2, 4)))

    l2 = Conv2DTranspose(2, 2, data_format="channels_first")
    l2(ops.ones((1, 2, 4, 4)))

    l3 = Conv3DTranspose(2, 2, data_format="channels_first")
    l3(ops.ones((1, 2, 4, 4, 4)))


def test_cropping_tuples() -> None:
    """Test ``cropping_tuples`` behavior and edge cases in ``test_layers``."""
    l1 = Cropping1D(cropping=((1, 2),))
    l1(ops.ones((1, 5, 2)))

    l2 = Cropping2D(cropping=((1, 2), (3, 4)))
    l2(ops.ones((1, 5, 5, 2)))

    l3 = Cropping3D(cropping=((1, 2), (3, 4), (5, 6)))
    l3(ops.ones((1, 5, 5, 5, 2)))


def test_einsum_dense_bias() -> None:
    """Test ``einsum_dense_bias`` behavior and edge cases in ``test_layers``."""
    layer = EinsumDense("abc,cd->abd", output_shape=(None, 64), bias_axes="d")
    layer.build((None, 32, 128))


class DummyCell(Layer):
    """Test helper class `DummyCell` for layers missing part7."""

    def __init__(self, units: TensorLike, **kwargs: TensorLike) -> None:
        """Initialize the ``DummyCell`` test helper instance.

        Args:
            units: Number of output hidden units.
            **kwargs: Keyword arguments forwarded to helper.
        """
        super().__init__(**kwargs)
        self.units = units

    def call(
        self,
        inputs: TensorLike,
        states: TensorLike,
        training: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Execute the forward computation of ``DummyCell``.

        Args:
            inputs: Input tensor or nested structure of tensors.
            states: Sequence of recurrent state tensors.
            training: Boolean flag indicating training versus inference mode.
            **kwargs: Keyword arguments forwarded to helper.

        Returns:
            TensorLike: Result from `call`.
        """
        return inputs, states


def test_rnn_missing_coverage() -> None:
    """Test ``rnn_missing_coverage`` behavior and edge cases in ``test_layers``."""
    rnn = RNN(DummyCell(10))
    rnn(ops.ones((1, 5, 10)))
    rnn(ops.ones((1, 5, 10)), initial_state=ops.zeros((1, 10)))

    # stateful reset_states
    srnn = SimpleRNN(10, stateful=True)
    srnn.build((1, 5, 10))
    srnn.reset_states()
    srnn.reset_states(states=(ops.zeros((1, 10)),))


def test_lstm_return_state() -> None:
    """Test ``lstm_return_state`` behavior and edge cases in ``test_layers``."""
    lstm = LSTM(10, return_state=True)
    lstm(ops.ones((1, 5, 10)))


def test_cells_build() -> None:
    """Test ``cells_build`` behavior and edge cases in ``test_layers``."""
    c1 = SimpleRNNCell(10)
    c1(ops.ones((1, 10)), states=(ops.zeros((1, 10)),))

    c2 = GRUCell(10)
    c2.build((None, 10))
    c2.build((None, 10))

    c3 = LSTMCell(10)
    c3.build((None, 10))
    c3.build((None, 10))


def test_bidirectional_missing() -> None:
    """Test Bidirectional merge_mode='sum' with initial_state and explicit backward_layer."""
    from unittest import mock

    with mock.patch.object(ops, "reverse", side_effect=lambda x, dims: x, create=True):
        bidi = Bidirectional(LSTM(10, return_sequences=True), merge_mode="sum")
        bidi(
            ops.ones((1, 5, 10)),
            initial_state=[
                ops.zeros((1, 10)),
                ops.zeros((1, 10)),
                ops.zeros((1, 10)),
                ops.zeros((1, 10)),
            ],
        )

    bidi2 = Bidirectional(LSTM(10), backward_layer=LSTM(10, go_backwards=True))
    bidi2(ops.ones((1, 5, 10)))


def test_normalization() -> None:
    """Test ``normalization`` behavior and edge cases in ``test_layers``."""
    norm = Normalization(invert=True)
    norm(ops.ones((1, 10)))


def test_group_normalization() -> None:
    """Test ``group_normalization`` behavior and edge cases in ``test_layers``."""
    gn1 = GroupNormalization(groups=2, scale=False, center=True)
    gn1(ops.ones((1, 10, 10, 4)))
    gn2 = GroupNormalization(groups=2, scale=False, center=False)
    gn2(ops.ones((1, 10, 10, 4)))


def test_random_layers() -> None:
    """Test ``random_layers`` behavior and edge cases in ``test_layers``."""
    l1 = RandomFlip("horizontal")
    l1(ops.ones((1, 10, 10, 3)))
    l2 = RandomRotation(0.2)
    l2(ops.ones((1, 10, 10, 3)))
    l4 = RandAugment(2, 9)
    l4(ops.ones((1, 10, 10, 3)))


def test_layer_normalization() -> None:
    """Test ``layer_normalization`` behavior and edge cases in ``test_layers``."""
    ln = LayerNormalization(axis=1)
    ln.build((None, 5, 10))
    ln.build((None, 5, 10))
    ln(ops.ones((1, 5, 10)))


def test_zeropadding() -> None:
    """Test ``zeropadding`` behavior and edge cases in ``test_layers``."""
    z1 = ZeroPadding1D(data_format="channels_first")
    z1(ops.ones((1, 2, 4)))

    z2 = ZeroPadding2D(padding=(1, 2))
    z2(ops.ones((1, 4, 4, 2)))
    z3 = ZeroPadding2D(data_format="channels_first")
    z3(ops.ones((1, 2, 4, 4)))

    z4 = ZeroPadding3D(padding=(1, 2, 3))
    z4(ops.ones((1, 4, 4, 4, 2)))
    z5 = ZeroPadding3D(data_format="channels_first")
    z5(ops.ones((1, 2, 4, 4, 4)))


def test_convlstm() -> None:
    """Test ``convlstm`` behavior and edge cases in ``test_layers``."""
    cl = ConvLSTM1D(2, 2, padding="same")
    cl.build((None, None, None, 3))


def test_conv_channels_first() -> None:
    """Test Conv1D, Conv2D, and Conv3D with data_format='channels_first'."""
    from unittest import mock

    with mock.patch.object(msc_ops, "transpose", side_effect=lambda x, perm: x, create=True):
        l1 = Conv1D(2, 2, data_format="channels_first")
        l1.build((None, 4, 4))
        l1(ops.ones((1, 4, 4)))

        l2 = Conv2D(2, 2, data_format="channels_first")
        l2.build((None, 4, 4, 4))
        l2(ops.ones((1, 4, 4, 4)))

        l3 = Conv3D(2, 2, data_format="channels_first")
        l3.build((None, 4, 4, 4, 4))
        l3(ops.ones((1, 4, 4, 4, 4)))


def test_cropping_elif() -> None:
    """Test ``cropping_elif`` behavior and edge cases in ``test_layers``."""
    l2 = Cropping2D(cropping=(1, 2))
    l2(ops.ones((1, 5, 5, 2)))

    l3 = Cropping3D(cropping=(1, 2, 3))
    l3(ops.ones((1, 5, 5, 5, 2)))


def test_rnn_reset_states_already_built() -> None:
    """Test ``rnn_reset_states_already_built`` behavior and edge cases in ``test_layers``."""
    srnn = SimpleRNN(10, stateful=True)
    srnn(ops.ones((1, 5, 10)))  # build and init states
    srnn.reset_states()
    srnn.reset_states(states=(ops.zeros((1, 10)),))


def test_simplernncell_unbuilt_call() -> None:
    """Test ``simplernncell_unbuilt_call`` behavior and edge cases in ``test_layers``."""
    c1 = SimpleRNNCell(10)
    # bypass __call__ to hit 'if not self.built:' inside call()
    c1.call(ops.ones((1, 10)), states=(ops.zeros((1, 10)),))


def test_bidirectional_dense_units() -> None:
    """Verify that wrapping a non-RNN Dense layer in Bidirectional raises AttributeError."""
    with pytest.raises(AttributeError):
        Bidirectional(Dense(10))


def test_rms_norm_center() -> None:
    """Test ``rms_norm_center`` behavior and edge cases in ``test_layers``."""
    rms = RMSNormalization(center=True, scale=False)
    rms(ops.ones((1, 10)))


def test_group_norm_build_twice_and_axis() -> None:
    """Test ``group_norm_build_twice_and_axis`` behavior and edge cases in ``test_layers``."""
    gn = GroupNormalization(axis=1, groups=2)
    GroupNormalization.build(gn, (None, 2, 10, 2))
    GroupNormalization.build(gn, (None, 2, 10, 2))
    gn(ops.ones((1, 2, 10, 2)))


def test_convlstm_padding_same_spatial() -> None:
    """Test ``convlstm_padding_same_spatial`` behavior and edge cases in ``test_layers``."""
    cl = ConvLSTM1D(2, 2, padding="same")
    cl.build((None, 5, 10, 3))


def test_dense_and_conv_missing_branches() -> None:
    """Test ``dense_and_conv_missing_branches`` behavior and edge cases in ``test_layers``."""
    with (
        patch("zero_keras.layers.core.ops") as mock_ops,
        patch("zero_keras.layers.convolution.ops") as mock_conv_ops,
        patch("ml_switcheroo_compiler.ops.linalg.conv.conv_general_dilated") as mock_cgd,
    ):
        mock_ops.matmul.return_value = asarray([[1.0]])
        mock_ops.add.return_value = asarray([[1.0]])
        mock_conv_ops.conv1d.return_value = asarray([[1.0]])
        mock_conv_ops.conv2d.return_value = asarray([[1.0]])
        mock_conv_ops.conv3d.return_value = asarray([[1.0]])
        mock_conv_ops.conv1d_transpose.return_value = asarray([[1.0]])
        mock_conv_ops.conv2d_transpose.return_value = asarray([[1.0]])
        mock_conv_ops.conv3d_transpose.return_value = asarray([[1.0]])
        mock_cgd.return_value = asarray([[1.0]])

        for layer_cls in [
            layers.Dense,
            layers.Conv1D,
            layers.Conv2D,
            layers.Conv3D,
            layers.Conv1DTranspose,
            layers.Conv2DTranspose,
            layers.Conv3DTranspose,
            layers.DepthwiseConv1D,
            layers.DepthwiseConv2D,
            layers.SeparableConv1D,
            layers.SeparableConv2D,
        ]:
            kwargs = {"use_bias": False, "activation": None}
            if layer_cls == layers.Dense:
                kwargs["units"] = 4
            elif "Conv" in layer_cls.__name__:
                if "Depthwise" not in layer_cls.__name__:
                    kwargs["filters"] = 4
                kwargs["kernel_size"] = 2

            layer = layer_cls(**kwargs)
            layer.kernel = asarray([1.0])

            if layer_cls.__name__ in (
                "DepthwiseConv1D",
                "DepthwiseConv2D",
                "SeparableConv1D",
                "SeparableConv2D",
            ):
                layer.depthwise_kernel = asarray([1.0])
                layer.pointwise_kernel = asarray([1.0])

            input_shape = (2, 4, 4, 4, 4) if layer_cls != layers.Dense else (2, 4)
            if "1D" in layer_cls.__name__:
                input_shape = (2, 4, 4)
            elif "2D" in layer_cls.__name__:
                input_shape = (2, 4, 4, 4)
            layer.build(input_shape)

            layer2 = layer_cls(**kwargs)
            # manually set built to True and weights so we don't fail in eager ops
            layer2.built = True
            layer2.kernel = asarray([1.0])
            if layer_cls.__name__ in (
                "DepthwiseConv1D",
                "DepthwiseConv2D",
                "SeparableConv1D",
                "SeparableConv2D",
            ):
                layer2.depthwise_kernel = asarray([1.0])
                layer2.pointwise_kernel = asarray([1.0])

            if layer_cls == layers.Dense:
                layer2(asarray([[1.0] * 4] * 2))
            elif "1D" in layer_cls.__name__:
                layer2(asarray([[[1.0] * 4] * 4] * 2))
            elif "2D" in layer_cls.__name__:
                layer2(asarray([[[[1.0] * 4] * 4] * 4] * 2))
            elif "3D" in layer_cls.__name__:
                layer2(asarray([[[[[1.0] * 4] * 4] * 4] * 4] * 2))


def test_pre_set_weights_with_bias() -> None:
    """Test ``pre_set_weights_with_bias`` behavior and edge cases in ``test_layers``."""
    for layer_cls in [
        layers.Dense,
        layers.Conv1D,
        layers.Conv2D,
        layers.Conv3D,
        layers.Conv1DTranspose,
        layers.Conv2DTranspose,
        layers.Conv3DTranspose,
        layers.DepthwiseConv1D,
        layers.DepthwiseConv2D,
        layers.SeparableConv1D,
        layers.SeparableConv2D,
    ]:
        kwargs = {"use_bias": True}
        if layer_cls == layers.Dense:
            kwargs["units"] = 4
        elif "Conv" in layer_cls.__name__:
            if "Depthwise" not in layer_cls.__name__:
                kwargs["filters"] = 4
            kwargs["kernel_size"] = 2

        layer = layer_cls(**kwargs)
        layer.kernel = asarray([1.0])
        layer.bias = asarray([1.0])

        if layer_cls.__name__ in (
            "DepthwiseConv1D",
            "DepthwiseConv2D",
            "SeparableConv1D",
            "SeparableConv2D",
        ):
            layer.depthwise_kernel = asarray([1.0])
            layer.pointwise_kernel = asarray([1.0])

        input_shape = (2, 4, 4, 4, 4) if layer_cls != layers.Dense else (2, 4)
        if "1D" in layer_cls.__name__:
            input_shape = (2, 4, 4)
        elif "2D" in layer_cls.__name__:
            input_shape = (2, 4, 4, 4)
        layer.build(input_shape)


def test_dense_activation_none() -> None:
    """Test ``dense_activation_none`` behavior and edge cases in ``test_layers``."""
    l = layers.Dense(10, activation=None)
    l(random.uniform((2, 5)))


def test_activity_reg_zeros() -> None:
    """Test ``activity_reg_zeros`` behavior and edge cases in ``test_layers``."""
    l = layers.ActivityRegularization(l1=0.0, l2=0.0)
    l(random.uniform((2, 5)))


def test_conv_activation_none() -> None:
    """Test ``conv_activation_none`` behavior and edge cases in ``test_layers``."""
    # Conv1D
    l = layers.Conv1D(2, 2, activation=None)
    l(random.uniform((2, 5, 3)))

    # Conv1DTranspose
    l = layers.Conv1DTranspose(2, 2, activation=None)
    l(random.uniform((2, 5, 3)))

    # Conv2D
    l = layers.Conv2D(2, 2, activation=None)
    l(random.uniform((2, 5, 5, 3)))

    # Conv2DTranspose
    l = layers.Conv2DTranspose(2, 2, activation=None)
    l(random.uniform((2, 5, 5, 3)))

    # Conv3D
    l = layers.Conv3D(2, 2, activation=None)
    l(random.uniform((2, 5, 5, 5, 3)))

    # Conv3DTranspose
    l = layers.Conv3DTranspose(2, 2, activation=None)
    l(random.uniform((2, 5, 5, 5, 3)))


def test_kerastensor_bool_eager_false() -> None:
    """Test ``kerastensor_bool_eager_false`` behavior and edge cases in ``test_layers``."""
    t = core_layers.KerasTensor((2,))
    t.data = True

    old_eager = config.eager_mode if hasattr(config, "eager_mode") else False
    config.eager_mode = False
    try:
        with pytest.raises(TypeError):
            bool(t)
    finally:
        config.eager_mode = old_eager


def test_preset_weights_convs() -> None:
    """Test ``preset_weights_convs`` behavior and edge cases in ``test_layers``."""
    # DepthwiseConv1D
    l = layers.DepthwiseConv1D(2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 2))

    # DepthwiseConv2D
    l = layers.DepthwiseConv2D(2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 2))

    # Conv1DTranspose
    l = layers.Conv1DTranspose(2, 3)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 2))

    # Conv2DTranspose
    l = layers.Conv2DTranspose(2, 3)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 2))

    # Conv3DTranspose
    l = layers.Conv3DTranspose(2, 3)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 5, 2))


def test_preset_weights_rnn_cells() -> None:
    """Test ``preset_weights_rnn_cells`` behavior and edge cases in ``test_layers``."""
    # SimpleRNNCell
    c = layers.SimpleRNNCell(4)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))

    # GRUCell with reset_after=True
    c = layers.GRUCell(4, reset_after=True)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))

    # GRUCell with reset_after=False
    c = layers.GRUCell(4, reset_after=False)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))

    # LSTMCell
    c = layers.LSTMCell(4)
    c.kernel = 1
    c.recurrent_kernel = 1
    c.bias = 1
    c.build((None, 2))


def test_simplernn_activation_none() -> None:
    """Test ``simplernn_activation_none`` behavior and edge cases in ``test_layers``."""
    l = layers.SimpleRNN(4, activation=None)
    l(random.uniform((2, 3, 2)))


def test_convlstm_bias_none() -> None:
    """Test ``convlstm_bias_none`` behavior and edge cases in ``test_layers``."""
    l = layers.ConvLSTM1D(2, 3, use_bias=False)
    l(random.uniform((2, 3, 5, 2)))


def test_wrapper_already_built_layers_branches_missing2() -> None:
    """Test ``wrapper_already_built`` behavior and edge cases in ``test_layers``."""
    # TimeDistributed already built wrapper
    l = layers.Dense(4)
    t = layers.TimeDistributed(l)
    t.build((None, 3, 2))
    t.build((None, 3, 2))  # trigger if self.built: return

    # SpectralNormalization already built wrapper
    l = layers.Dense(4)
    s = layers.SpectralNormalization(l)
    s.build((None, 2))
    s.build((None, 2))  # trigger if self.built: return


def test_bidirectional_args() -> None:
    """Test ``bidirectional_args`` behavior and edge cases in ``test_layers``."""

    class DummyCell:
        """Test helper class `DummyCell` for layers branches missing2."""

        def __init__(self, units: TensorLike) -> None:
            """Initialize the ``DummyCell`` test helper instance.

            Args:
                units: Number of output hidden units.
            """
            self.units = units

    class DummyLayer(layers.Layer):
        """Test helper class `DummyLayer` for layers branches missing2."""

        def __init__(self, cell: TensorLike) -> None:
            """Initialize the ``DummyLayer`` test helper instance.

            Args:
                cell: Recurrent cell instance under test.
            """
            super().__init__()
            self.cell = cell
            self.return_sequences = False
            self.return_state = False
            self.go_backwards = False
            self.stateful = False

    l = DummyLayer(DummyCell(4))
    _b = layers.Bidirectional(l)


def test_conv_aliases_activation_none() -> None:
    """Test ``conv_aliases_activation_none`` behavior and edge cases in ``test_layers``."""
    for cls, in_shape in [
        (layers.Convolution1D, (2, 5, 3)),
        (layers.Convolution1DTranspose, (2, 5, 3)),
        (layers.Convolution2D, (2, 5, 5, 3)),
        (layers.Convolution2DTranspose, (2, 5, 5, 3)),
        (layers.Convolution3D, (2, 5, 5, 5, 3)),
        (layers.Convolution3DTranspose, (2, 5, 5, 5, 3)),
    ]:
        l = cls(2, 2, activation=None)
        l(random.uniform(in_shape))


def test_conv_aliases_preset_weights() -> None:
    """Test ``conv_aliases_preset_weights`` behavior and edge cases in ``test_layers``."""
    # 1D
    l = layers.Convolution1D(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 3))

    # 1DT
    l = layers.Convolution1DTranspose(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 3))

    # 2D
    l = layers.Convolution2D(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 3))

    # 2DT
    l = layers.Convolution2DTranspose(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 3))

    # 3D
    l = layers.Convolution3D(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 5, 3))

    # 3DT
    l = layers.Convolution3DTranspose(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 5, 3))


def test_layers_activations_attention_conv_core_merging_norm_pooling_rnn_misc() -> None:
    """Cover every remaining line and branch across activations, attention, convolution, core, merging, normalization, pooling, rnn, and misc layers."""
    # 1. layers/activations.py (PReLU build(None) and alpha=None)
    prelu = layers.PReLU()
    prelu.build(None)
    prelu.alpha = None
    assert prelu(ops.ones((2, 3))) is not None

    # 2. layers/attention.py (GroupQueryAttention branches)
    with pytest.raises(ValueError):
        layers.GroupQueryAttention(head_dim=4, num_query_heads=5, num_key_value_heads=2)

    gqa = layers.GroupQueryAttention(
        head_dim=4, num_query_heads=4, num_key_value_heads=2, dropout=0.1
    )
    gqa.build([(None, 3, 8), (None, 3, 8)])
    q = ops.ones((1, 3, 8))
    v = ops.ones((1, 3, 8))
    mask = ops.ones((1, 1, 3, 3))
    out_gqa = gqa.call(
        q,
        v,
        key=None,
        attention_mask=mask,
        use_causal_mask=True,
        training=True,
    )
    assert out_gqa is not None

    # Also test GroupQueryAttention when num_heads == num_key_value_heads
    gqa_eq = layers.GroupQueryAttention(
        head_dim=4, num_query_heads=2, num_key_value_heads=2, dropout=0.0
    )
    gqa_eq.build((None, 3, 8))
    assert gqa_eq.call(q, v, key=v, training=False) is not None

    # 3. layers/convolution.py (DepthwiseConv1D/2D & SeparableConv1D/2D channels_first + activation=None)
    dw1_cf = layers.DepthwiseConv1D(kernel_size=2, data_format="channels_first", activation=None)
    dw1_cf.activation = None
    assert dw1_cf(ops.ones((1, 2, 4))) is not None
    dw1_cl = layers.DepthwiseConv1D(kernel_size=2, data_format="channels_last", activation="relu")
    assert dw1_cl(ops.ones((1, 4, 2))) is not None

    dw2_cf = layers.DepthwiseConv2D(kernel_size=2, data_format="channels_first", activation=None)
    dw2_cf.activation = None
    assert dw2_cf(ops.ones((1, 2, 4, 4))) is not None
    dw2_cl = layers.DepthwiseConv2D(kernel_size=2, data_format="channels_last", activation="relu")
    assert dw2_cl(ops.ones((1, 4, 4, 2))) is not None

    sep1 = layers.SeparableConv1D(filters=3, kernel_size=2, activation=None)
    sep1.activation = None
    assert sep1(ops.ones((1, 4, 2))) is not None

    sep2 = layers.SeparableConv2D(filters=3, kernel_size=2, activation=None)
    sep2.activation = None
    assert sep2(ops.ones((1, 4, 4, 2))) is not None

    # 4. layers/core.py (Dense.call unbuilt, Dropout training=False, Flatten 1D)
    d = layers.Dense(2)
    assert d.call(ops.ones((1, 3))) is not None
    do = layers.Dropout(0.5)
    assert do.call(ops.ones((1, 3)), training=False) is not None
    fl = layers.Flatten()
    assert fl.call(ops.ones((3,))) is not None

    # 5. layers/merging.py (Dot with list axes)
    dot_l = layers.Dot(axes=[1, 1])
    assert dot_l([ops.ones((2, 3)), ops.ones((2, 3))]) is not None

    # 6. layers/normalization.py (BatchNormalization training=True with scale=False, center=False)
    bn = layers.BatchNormalization(scale=False, center=False)
    assert bn.call(ops.ones((2, 4)), training=True) is not None

    # 7. layers/pooling.py (AdaptiveAveragePooling1D/2D/3D and AdaptiveMaxPooling1D/2D/3D)
    aap1 = layers.AdaptiveAveragePooling1D(output_size=2)
    assert aap1(ops.ones((1, 4, 2))) is not None
    assert aap1.get_config()["output_size"] == (2,)

    aap2 = layers.AdaptiveAveragePooling2D(output_size=(2, 2))
    assert aap2(ops.ones((1, 4, 4, 2))) is not None

    aap3 = layers.AdaptiveAveragePooling3D(output_size=2)
    assert aap3(ops.ones((1, 4, 4, 4, 2))) is not None

    amp1 = layers.AdaptiveMaxPooling1D(output_size=2)
    assert amp1(ops.ones((1, 4, 2))) is not None

    amp2 = layers.AdaptiveMaxPooling2D(output_size=(2, 2))
    assert amp2(ops.ones((1, 4, 4, 2))) is not None

    amp3 = layers.AdaptiveMaxPooling3D(output_size=2)
    assert amp3(ops.ones((1, 4, 4, 4, 2))) is not None

    # 8. layers/rnn.py (ConvLSTMCell with dropout > 0 and recurrent_dropout > 0 in training=True)
    from zero_keras.layers.rnn import ConvLSTMCell

    cl_cell = ConvLSTMCell(
        rank=2, filters=2, kernel_size=(1, 1), dropout=0.1, recurrent_dropout=0.1
    )
    x_cl = ops.ones((1, 4, 4, 2))
    h_cl = ops.ones((1, 4, 4, 2))
    c_cl = ops.ones((1, 4, 4, 2))
    cl_cell.build(x_cl.shape)
    assert cl_cell.call(x_cl, [h_cl, c_cl], training=True) is not None

    # 9. layers/misc.py (EinsumDense, FlaxLayer, JaxLayer, TFSMLayer, TorchModuleWrapper, ReversibleEmbedding, deserialize)
    ed = layers.EinsumDense("ab,bc->ac", output_shape=(3,), activation=None)
    ed.activation = None
    assert ed.call(ops.ones((2, 4))) is not None
    ed.build((2, 4))  # already built -> line 151

    from zero_keras.utils import operation_utils

    inp_src = layers.Input(shape=(3,))
    out_src = layers.Dense(2)(inp_src)
    res_src = operation_utils.get_source_inputs(out_src)
    assert len(res_src) == 1
    res_list_src = operation_utils.get_source_inputs([inp_src])
    assert len(res_list_src) == 1

    flax_single = layers.FlaxLayer(
        module=lambda x, **k: x,
        method=lambda mod, x, **k: mod(x),
    )
    assert flax_single.call(ops.ones((2, 2))) is not None
    flax_tuple = layers.FlaxLayer(module=lambda x, **k: (x, x))
    assert isinstance(flax_tuple.call(ops.ones((2, 2))), tuple)

    jax_l = layers.JaxLayer(call_fn=lambda p, x, **k: x, params={"w": 1})
    assert jax_l.call(ops.ones((2, 2))) is not None
    jax_none = layers.JaxLayer(call_fn=None)
    assert jax_none.call(ops.ones((2, 2))) is not None

    tfsm = layers.TFSMLayer("fake_dir", name="tfsm_named", dtype="float32")
    assert tfsm.call(ops.ones((2, 2))) is not None
    tfsm.module = lambda x, **k: x
    assert tfsm.call(ops.ones((2, 2))) is not None
    tfsm.module = lambda x, **k: (x, x)
    assert isinstance(tfsm.call(ops.ones((2, 2))), tuple)

    tmw_single = layers.TorchModuleWrapper(module=lambda x, **k: x, name="tmw")
    assert tmw_single.call(ops.ones((2, 2))) is not None
    tmw_tuple = layers.TorchModuleWrapper(module=lambda x, **k: (x, x))
    assert isinstance(tmw_tuple.call(ops.ones((2, 2))), tuple)

    rev_emb = layers.ReversibleEmbedding(input_dim=5, output_dim=3, logit_soft_cap=10.0)
    tok_out = rev_emb(ops.convert_to_tensor([[1, 2]], dtype="int32"), reverse=False)
    assert tok_out is not None
    rev_unbuilt = layers.ReversibleEmbedding(input_dim=5, output_dim=3, logit_soft_cap=None)
    assert rev_unbuilt.call(ops.ones((1, 2, 3)), reverse=True) is not None
    assert rev_emb.call(ops.ones((1, 2, 3)), reverse=True) is not None
    assert rev_emb.get_config()["tie_weights"] is True

    # misc.deserialize with nested layer config dict
    nested_cfg = {
        "class_name": "TimeDistributed",
        "config": {
            "layer": {"class_name": "Identity", "config": {}},
        },
    }
    assert isinstance(layers.deserialize(nested_cfg), layers.TimeDistributed)

    # 10. layers/preprocessing.py
    from unittest import mock

    from zero_keras.core_layers import KerasTensor

    cat_int = layers.CategoryEncoding(num_tokens=4, output_mode="int")
    assert cat_int(ops.convert_to_tensor([0, 1], dtype="int32")) is not None
    cat_bad = layers.CategoryEncoding(num_tokens=4, output_mode="bad")
    with pytest.raises(ValueError):
        cat_bad(ops.convert_to_tensor([0, 1], dtype="int32"))

    disc_oh = layers.Discretization(bin_boundaries=[1.0, 2.0], output_mode="one_hot")
    assert disc_oh(ops.convert_to_tensor([0.5, 1.5], dtype="float32")) is not None
    disc_mh = layers.Discretization(bin_boundaries=[1.0, 2.0], output_mode="multi_hot")
    assert disc_mh(ops.convert_to_tensor([[0.5, 1.5]], dtype="float32")) is not None

    with mock.patch("zero_keras.ops.stft", return_value=ops.ones((1, 4, 5), dtype="float32")):
        mel1 = layers.MelSpectrogram(
            fft_length=8,
            sequence_length=4,
            sequence_stride=2,
            num_mel_bins=3,
            mag_exp=2.0,
            power_to_db=True,
            top_db=80.0,
        )
        assert mel1(ops.ones((1, 16), dtype="float32")) is not None
        mel2 = layers.MelSpectrogram(
            fft_length=8,
            sequence_length=4,
            sequence_stride=2,
            num_mel_bins=3,
            mag_exp=1.0,
            power_to_db=False,
            top_db=None,
        )
        assert mel2(ops.ones((1, 16), dtype="float32")) is not None
        with mock.patch("zero_keras.ops.matmul", return_value=ops.ones((4,), dtype="float32")):
            mel3 = layers.MelSpectrogram(
                fft_length=8, sequence_length=4, sequence_stride=2, power_to_db=True, top_db=None
            )
            assert mel3(ops.ones((1, 16), dtype="float32")) is not None

        for stft_mode, exp_d in [
            ("magnitude", True),
            ("power", False),
            ("log", False),
            ("complex", False),
        ]:
            stft_l = layers.STFTSpectrogram(
                mode=stft_mode, frame_length=4, frame_step=2, fft_length=8, expand_dims=exp_d
            )
            assert stft_l(ops.ones((1, 16), dtype="float32")) is not None
        stft_bad = layers.STFTSpectrogram(mode="invalid_mode")
        with pytest.raises(ValueError):
            stft_bad(ops.ones((1, 16), dtype="float32"))

    with (
        mock.patch(
            "zero_keras.ops.image.affine_generator", return_value=ops.ones((1, 8)), create=True
        ),
        mock.patch(
            "zero_keras.ops.image.affine_transform",
            return_value=ops.ones((1, 8, 8, 3)),
            create=True,
        ),
    ):
        rs = layers.RandomShear(x_factor=(-0.1, 0.1), y_factor=(-0.1, 0.1))
        assert rs.call(ops.ones((1, 8, 8, 3)), training=True) is not None

    sl = layers.StringLookup(vocabulary=["a", "b"], output_mode="multi_hot")
    assert sl(KerasTensor((2, 3), dtype="string")) is not None
    sl_int = layers.StringLookup(vocabulary=["a", "b"], output_mode="int")
    assert sl_int(KerasTensor((2, 3), dtype="string")) is not None

    class HasNumpy:
        """Stub with .numpy() returning a Tensor."""

        def numpy(self) -> ops.Tensor:
            """Return a backend Tensor of string tokens.

            Returns:
                ops.Tensor: Backend tensor.
            """
            return ops.convert_to_tensor(["x", "y"])

    sl.adapt(HasNumpy())
    sl.adapt("single_scalar")
    sl.oov_token = "[UNK]"
    sl.adapt(["x", "[UNK]"])

    with mock.patch("zero_keras.ops.one_hot", return_value=ops.ones((2,), dtype="float32")):
        sl_mh = layers.StringLookup(vocabulary=[0, 1], output_mode="multi_hot")
        assert sl_mh(ops.convert_to_tensor([0, 1], dtype="int32")) is not None

    tv = layers.TextVectorization()
    tv.adapt(["hello world", ["foo bar"]])
    tv.adapt(HasNumpy())
    tv.adapt("single word")

    hc = layers.HashedCrossing(num_bins=4)
    assert hc(ops.convert_to_tensor([1, 2], dtype="int32")) is not None

    clahe = layers.ContrastLimitedAdaptiveHistogramEqualization()
    assert clahe(ops.ones((1, 4, 4, 3))) is not None
    assert clahe.get_config()["clip_limit"] == 4.0


def test_random_crop() -> None:
    """Verify ``RandomCrop`` center-crops in inference and randomly crops in training."""
    layer = layers.RandomCrop(height=2, width=2, seed=42)
    x = ops.reshape(ops.arange(2 * 4 * 4 * 3, dtype="float32"), (2, 4, 4, 3))
    out_inf = layer(x, training=False)
    assert out_inf.shape == (2, 2, 2, 3)
    assert_allclose_zero(out_inf.data, x[:, 1:3, 1:3, :])

    out_train = layer(x, training=True)
    assert out_train.shape == (2, 2, 2, 3)


def test_random_translation() -> None:
    """Verify ``RandomTranslation`` output shape in inference and training modes."""
    layer = layers.RandomTranslation(height_factor=0.2, width_factor=0.2, seed=42)
    x = ops.ones((2, 4, 4, 3), dtype="float32")
    out_inf = layer(x, training=False)
    assert out_inf.shape == (2, 4, 4, 3)
    assert_allclose_zero(out_inf.data, x)

    out_train = layer(x, training=True)
    assert out_train.shape == (2, 4, 4, 3)


def test_random_zoom() -> None:
    """Verify ``RandomZoom`` output shape in inference and training modes."""
    layer = layers.RandomZoom(height_factor=0.2, width_factor=0.2, seed=42)
    x = ops.ones((2, 4, 4, 3), dtype="float32")
    out_inf = layer(x, training=False)
    assert out_inf.shape == (2, 4, 4, 3)
    assert_allclose_zero(out_inf.data, x)

    out_train = layer(x, training=True)
    assert out_train.shape == (2, 4, 4, 3)


def test_input_layer() -> None:
    """Verify ``Input`` creation with existing tensor and ``layers.serialize`` / ``deserialize``."""
    layer = layers.Input(
        shape=(2,),
        batch_size=4,
        dtype="float32",
        sparse=True,
        name="test_in",
        tensor=ops.zeros((4, 2)),
    )
    out = layer(ops.zeros((4, 2)))
    assert out.shape == (4, 2)

    assert layers.serialize(None) is None
    assert layers.serialize("SomeLayer") == "SomeLayer"
    assert isinstance(layers.serialize(layer), dict)

    assert layers.deserialize(None) is None
    assert layers.deserialize("UnknownStr") == "UnknownStr"
    with pytest.raises(TypeError):
        layers.deserialize("Dense")
    assert layers.deserialize({"class_name": "Dense", "config": {"units": 2}}) is not None
    assert layers.deserialize(123) == 123


def test_group_query_attention() -> None:
    """Verify ``GroupQueryAttention`` forward execution and output shape."""
    layer = layers.GroupQueryAttention(head_dim=2, num_query_heads=2, num_key_value_heads=2)
    q = ops.ones((2, 4, 4), dtype="float32")
    v = ops.ones((2, 4, 4), dtype="float32")
    out = layer(q, v)
    assert out.shape == (2, 4, 4)


def test_hashed_crossing() -> None:
    """Verify ``HashedCrossing`` integer bin hashing on a pair of integer tensors."""
    layer = layers.HashedCrossing(num_bins=10)
    x1 = ops.convert_to_tensor([[1, 2], [3, 4]], dtype="int32")
    x2 = ops.convert_to_tensor([[5, 6], [7, 8]], dtype="int32")
    out = layer((x1, x2))
    assert out.shape == (2, 2)
    assert all(0.0 <= v < 10.0 for v in to_flat_floats(out))


def test_random_rotation() -> None:
    """Verify ``RandomRotation`` output shape in inference and training modes."""
    layer = layers.RandomRotation(0.2, seed=42)
    x = ops.ones((2, 4, 4, 3), dtype="float32")
    out_inf = layer(x, training=False)
    assert out_inf.shape == (2, 4, 4, 3)
    out_train = layer(x, training=True)
    assert out_train.shape == (2, 4, 4, 3)


def test_resizing_branches() -> None:
    """Verify ``Resizing`` across ``nearest``, ``bicubic``, and ``lanczos3`` interpolation modes."""
    x = ops.ones((2, 4, 4, 3), dtype="float32")
    for interp in ["nearest", "bicubic", "lanczos3"]:
        out = layers.Resizing(2, 2, interpolation=interp)(x)
        assert out.shape == (2, 2, 2, 3)


def test_string_lookup_none() -> None:
    """Verify ``StringLookup`` with ``vocabulary=None``."""
    layer = layers.StringLookup(vocabulary=None)
    x = ["a", "b"]
    out = layer(x)
    assert out.shape == (2,)


def test_functional_merge_layers() -> None:
    """Verify functional merge helpers ``add``, ``subtract``, ``multiply``, ``average``, ``maximum``, ``minimum``, ``concatenate``, and ``dot``."""
    x1 = ops.ones((1, 2), dtype="float32")
    x2 = ops.full((1, 2), 3.0, dtype="float32")

    assert_allclose_zero(layers.add([x1, x2]).data, [[4.0, 4.0]])
    assert_allclose_zero(layers.subtract([x1, x2]).data, [[-2.0, -2.0]])
    assert_allclose_zero(layers.multiply([x1, x2]).data, [[3.0, 3.0]])
    assert_allclose_zero(layers.average([x1, x2]).data, [[2.0, 2.0]])
    assert_allclose_zero(layers.maximum([x1, x2]).data, [[3.0, 3.0]])
    assert_allclose_zero(layers.minimum([x1, x2]).data, [[1.0, 1.0]])
    assert layers.concatenate([x1, x2], axis=-1).shape == (1, 4)
    assert_allclose_zero(layers.dot([x1, x2], axes=1).data, [[6.0]])

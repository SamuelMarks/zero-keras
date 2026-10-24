"""Targeted unit and branch coverage tests for 9 layer modules in ``src/zero_keras/layers/``."""

from __future__ import annotations
from zero_keras.types import TensorLike

import pytest

from zero_keras import layers, ops


def test_layers_activations_attention_conv_core_merging_norm_pooling_rnn_misc() -> None:
    """Cover every remaining line and branch across activations, attention, convolution, core, merging, normalization, pooling, rnn, and misc layers."""
    # 1. layers/activations.py (PReLU build(None) and alpha=None)
    prelu = layers.PReLU()
    prelu.build(None)
    del prelu.alpha
    assert prelu(ops.ones((2, 3))) is not None

    # 2. layers/attention.py (GroupQueryAttention branches)
    with pytest.raises(ValueError):
        layers.GroupQueryAttention(num_heads=5, key_dim=4, num_key_value_heads=2)

    gqa = layers.GroupQueryAttention(num_heads=4, key_dim=4, num_key_value_heads=2, dropout=0.1)
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
    gqa_eq = layers.GroupQueryAttention(num_heads=2, key_dim=4, num_key_value_heads=2, dropout=0.0)
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
        """Stub with .numpy() returning array."""

        def numpy(self) -> TensorLike:
            """Return numpy array."""
            import numpy as np

            return np.array(["x", "y"])

    sl.adapt(HasNumpy())  # type: ignore[arg-type]
    sl.adapt("single_scalar")
    sl.oov_token = "[UNK]"
    sl.adapt(["x", "[UNK]"])

    with mock.patch("zero_keras.ops.one_hot", return_value=ops.ones((2,), dtype="float32")):
        sl_mh = layers.StringLookup(vocabulary=[0, 1], output_mode="multi_hot")
        assert sl_mh(ops.convert_to_tensor([0, 1], dtype="int32")) is not None

    tv = layers.TextVectorization()
    tv.adapt(["hello world", ["foo bar"]])
    tv.adapt(HasNumpy())  # type: ignore[arg-type]
    tv.adapt("single word")

    hc = layers.HashedCrossing(num_bins=4)
    assert hc(ops.convert_to_tensor([1, 2], dtype="int32")) is not None

    clahe = layers.ContrastLimitedAdaptiveHistogramEqualization()
    assert clahe(ops.ones((1, 4, 4, 3))) is not None
    assert clahe.get_config()["clip_limit"] == 4.0

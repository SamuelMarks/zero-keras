"""Unit and behavioral tests for missing layers vision in zero_keras."""

from __future__ import annotations

from zero_keras import layers, ops, random


def test_equalization() -> None:
    """Test ``equalization`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.Equalization()
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    layer(x, training=True)
    layer(x, training=False)


def test_augmix() -> None:
    """Test ``augmix`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.AugMix()
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    layer(x, training=True)
    layer(x, training=False)


def test_auto_contrast() -> None:
    """Test ``auto_contrast`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.AutoContrast()
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    layer(x, training=True)
    layer(x, training=False)


def test_center_crop() -> None:
    """Test CenterCrop across 3D and 4D inputs and both data formats."""
    layer = layers.CenterCrop(height=5, width=5)
    out4 = layer(random.uniform((2, 10, 10, 3), dtype="float32"))
    assert out4.shape == (2, 5, 5, 3)

    out3 = layer(random.uniform((10, 10, 3), dtype="float32"))
    assert out3.shape == (5, 5, 3)

    layer_cf = layers.CenterCrop(height=5, width=5, data_format="channels_first")
    out4_cf = layer_cf(random.uniform((2, 3, 10, 10), dtype="float32"))
    assert out4_cf.shape == (2, 3, 5, 5)

    out3_cf = layer_cf(random.uniform((3, 10, 10), dtype="float32"))
    assert out3_cf.shape == (3, 5, 5)

    out2 = layer(ops.ones((4, 4), dtype="float32"))
    assert out2.shape == (4, 4)

    disc = layers.Discretization(num_bins=3)
    disc.adapt(ops.convert_to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float32"))
    assert disc.bin_boundaries is not None
    assert len(disc.bin_boundaries) == 2

    norm_preset = layers.Normalization(mean=[1.0, 2.0], variance=[0.5, 1.5])
    norm_preset.build((None, 2))
    norm_adapt = layers.Normalization()
    norm_adapt.adapt(ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
    assert norm_adapt.built is True


def test_cutmix() -> None:
    """Verify CutMix on raw images, tuple (images, labels), and dict {"images": ..., "labels": ...}."""
    layer = layers.CutMix()
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    y = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    layer(x, training=True)
    layer(x, training=False)

    out_tup_train = layer((x, y), training=True)
    assert isinstance(out_tup_train, tuple) and len(out_tup_train) == 2
    out_tup_eval = layer((x, y), training=False)
    assert isinstance(out_tup_eval, tuple) and len(out_tup_eval) == 2

    out_dict_train = layer({"images": x, "labels": y}, training=True)
    assert isinstance(out_dict_train, dict) and "labels" in out_dict_train
    out_dict_no_lbl_train = layer({"images": x}, training=True)
    assert isinstance(out_dict_no_lbl_train, dict) and "images" in out_dict_no_lbl_train
    out_dict_eval = layer({"images": x, "labels": y}, training=False)
    assert isinstance(out_dict_eval, dict) and "labels" in out_dict_eval
    out_dict_no_lbl_eval = layer({"images": x}, training=False)
    assert isinstance(out_dict_no_lbl_eval, dict) and "images" in out_dict_no_lbl_eval


def test_mixup() -> None:
    """Verify MixUp on raw images, tuple (images, labels), and dict {"images": ..., "labels": ...}."""
    layer = layers.MixUp()
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    y = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    layer(x, training=True)
    layer(x, training=False)

    out_tup_train = layer((x, y), training=True)
    assert isinstance(out_tup_train, tuple) and len(out_tup_train) == 2
    out_tup_eval = layer((x, y), training=False)
    assert isinstance(out_tup_eval, tuple) and len(out_tup_eval) == 2

    out_dict_train = layer({"images": x, "labels": y}, training=True)
    assert isinstance(out_dict_train, dict) and "labels" in out_dict_train
    out_dict_no_lbl_train = layer({"images": x}, training=True)
    assert isinstance(out_dict_no_lbl_train, dict) and "images" in out_dict_no_lbl_train
    out_dict_eval = layer({"images": x, "labels": y}, training=False)
    assert isinstance(out_dict_eval, dict) and "labels" in out_dict_eval
    out_dict_no_lbl_eval = layer({"images": x}, training=False)
    assert isinstance(out_dict_no_lbl_eval, dict) and "images" in out_dict_no_lbl_eval


def test_random_flip() -> None:
    """Test ``random_flip`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.RandomFlip()
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    layer(x, training=True)
    layer(x, training=False)


def test_random_rotation() -> None:
    """Test ``random_rotation`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.RandomRotation(0.2)
    x = random.uniform((2, 10, 10, 3), dtype="float32")
    layer(x, training=True)
    layer(x, training=False)


def test_resizing() -> None:
    """Test ``resizing`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.Resizing(10, 10)
    x = random.uniform((2, 20, 20, 3), dtype="float32")
    layer(x, training=True)
    layer(x, training=False)


def test_category_encoding() -> None:
    """Test ``category_encoding`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
    x = ops.convert_to_tensor([1, 2], dtype="int32")
    _out = layer(x)

    layer = layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    x = ops.convert_to_tensor([[1, 2], [0, 3]], dtype="int32")
    _out = layer(x)

    layer = layers.CategoryEncoding(num_tokens=4, output_mode="count")
    x = ops.convert_to_tensor([[1, 2], [0, 3]], dtype="int32")
    _out = layer(x)

    layer = layers.CategoryEncoding(output_mode="count")
    x = ops.convert_to_tensor([[1, 2], [0, 3]], dtype="int32")
    _out = layer(x)


def test_multi_head_attention() -> None:
    """Test ``multi_head_attention`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.MultiHeadAttention(num_heads=2, key_dim=2)
    q = random.uniform((2, 10, 4), dtype="float32")
    v = random.uniform((2, 5, 4), dtype="float32")
    layer(q, v, use_causal_mask=True, return_attention_scores=True)
    layer(q, v, attention_mask=ops.zeros((2, 2, 10, 5), dtype="float32"))


def test_alpha_dropout() -> None:
    """Test ``alpha_dropout`` behavior and edge cases in ``test_missing_layers_vision``."""
    layer = layers.AlphaDropout(0.5)
    layer(random.uniform((2, 10), dtype="float32"), training=True)

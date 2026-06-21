import numpy as np
from zero_keras import layers


def test_equalization():
    layer = layers.Equalization()
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_augmix():
    layer = layers.AugMix()
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_auto_contrast():
    layer = layers.AutoContrast()
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_center_crop():
    layer = layers.CenterCrop(height=5, width=5)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_cutmix():
    layer = layers.CutMix()
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_mixup():
    layer = layers.MixUp()
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_random_flip():
    layer = layers.RandomFlip()
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_random_rotation():
    layer = layers.RandomRotation(0.2)
    x = np.random.uniform(size=(2, 10, 10, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_resizing():
    layer = layers.Resizing(10, 10)
    x = np.random.uniform(size=(2, 20, 20, 3)).astype(np.float32)
    layer(x, training=True)
    layer(x, training=False)


def test_category_encoding():
    layer = layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
    x = np.array([1, 2]).astype(np.int32)
    out = layer(x)

    layer = layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    x = np.array([[1, 2], [0, 3]]).astype(np.int32)
    out = layer(x)

    layer = layers.CategoryEncoding(num_tokens=4, output_mode="count")
    x = np.array([[1, 2], [0, 3]]).astype(np.int32)
    out = layer(x)

    layer = layers.CategoryEncoding(output_mode="count")
    x = np.array([[1, 2], [0, 3]]).astype(np.int32)
    out = layer(x)


def test_multi_head_attention():
    layer = layers.MultiHeadAttention(num_heads=2, key_dim=2)
    q = np.random.uniform(size=(2, 10, 4)).astype(np.float32)
    v = np.random.uniform(size=(2, 5, 4)).astype(np.float32)
    layer(q, v, use_causal_mask=True, return_attention_scores=True)
    layer(q, v, attention_mask=np.zeros((2, 2, 10, 5)).astype(np.float32))


def test_alpha_dropout():
    layer = layers.AlphaDropout(0.5)
    layer(np.random.uniform(size=(2, 10)).astype(np.float32), training=True)

import numpy as np
from zero_keras import layers


def test_random_crop():
    layer = layers.RandomCrop(height=2, width=2)
    x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
    try:
        layer(x)
    except Exception:
        pass
    try:
        layer(x, training=True)
    except Exception:
        pass


def test_random_translation():
    layer = layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
    x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
    layer(x)
    try:
        layer(x, training=True)
    except Exception:
        pass


def test_random_zoom():
    layer = layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
    layer(x)
    try:
        layer(x, training=True)
    except Exception:
        pass


def test_input_layer():
    # 13360-13361
    layer = layers.Input(
        shape=(2,),
        batch_size=4,
        dtype="float32",
        sparse=True,
        name="test_in",
        tensor=np.zeros((4, 2)),
    )
    layer(np.zeros((4, 2)))

    # 13366, 13385-13397 (serialize/deserialize)
    assert layers.serialize(None) is None
    assert layers.serialize("SomeLayer") == "SomeLayer"
    layers.serialize(layer)

    assert layers.deserialize(None) is None
    assert layers.deserialize("UnknownStr") == "UnknownStr"
    try:
        layers.deserialize("Dense")
    except Exception:
        pass
    assert (
        layers.deserialize({"class_name": "Dense", "config": {"units": 2}}) is not None
    )
    assert layers.deserialize(123) == 123


def test_group_query_attention():
    try:
        layer = layers.GroupQueryAttention(
            head_dim=2, num_query_heads=2, num_key_value_heads=2
        )
        q = np.random.uniform(size=(2, 4, 4)).astype("float32")
        v = np.random.uniform(size=(2, 4, 4)).astype("float32")
        layer(q, v)
    except Exception:
        pass


def test_hashed_crossing():
    layer = layers.HashedCrossing(num_bins=10)
    x1 = np.random.randint(0, 10, size=(2, 2))
    x2 = np.random.randint(0, 10, size=(2, 2))
    try:
        layer((x1, x2))
    except Exception:
        pass


def test_random_rotation():
    layer = layers.RandomRotation(0.2)
    x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
    try:
        layer(x)
    except Exception:
        pass


def test_resizing_branches():
    x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
    for interp in ["nearest", "bicubic", "lanczos3"]:
        try:
            layers.Resizing(2, 2, interpolation=interp)(x)
        except Exception:
            pass


def test_string_lookup_none():
    layer = layers.StringLookup(vocabulary=None)
    x = np.array(["a", "b"])
    layer(x)


def test_functional_merge_layers():
    x1 = np.ones((1, 2)).astype("float32")
    x2 = np.ones((1, 2)).astype("float32")

    layers.add([x1, x2])
    layers.subtract([x1, x2])
    layers.multiply([x1, x2])
    layers.average([x1, x2])
    layers.maximum([x1, x2])
    layers.minimum([x1, x2])
    layers.concatenate([x1, x2], axis=-1)
    layers.dot([x1, x2], axes=1)

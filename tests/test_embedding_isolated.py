import numpy as np

from zero_keras import layers


def test_embedding_isolated():
    layer = layers.Embedding(input_dim=10, output_dim=4)
    x = np.random.randint(0, 10, size=(2, 5))
    out = layer(x)
    assert out.shape == (2, 5, 4)


def test_embedding_built_twice():
    layer = layers.Embedding(input_dim=10, output_dim=4)
    layer.build((2, 5))
    layer.build((2, 5))
    assert layer.built

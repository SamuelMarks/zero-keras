"""Module docstring."""

import numpy as np
import zero_keras.layers as layers


def test_bidirectional_backward_layer():
    """Function docstring."""
    l1 = layers.SimpleRNN(4)
    l2 = layers.SimpleRNN(4, go_backwards=True)
    b = layers.Bidirectional(l1, backward_layer=l2)
    b(np.random.rand(2, 3, 2))

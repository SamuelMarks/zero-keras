"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import numpy as np

from zero_keras import layers


def test_bidirectional_backward_layer() -> TensorLike:
    """Function docstring."""
    l1 = layers.SimpleRNN(4)
    l2 = layers.SimpleRNN(4, go_backwards=True)
    b = layers.Bidirectional(l1, backward_layer=l2)
    b(np.random.rand(2, 3, 2))

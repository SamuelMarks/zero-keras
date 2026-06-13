"""Keras regularizers."""

from typing import Any
import ml_switcheroo_compiler.ops as ops
from zero_keras.activations import _to_tensor, _wrap


class Regularizer:
    def __call__(self, x: Any) -> Any:
        return 0.0


class L2(Regularizer):
    def __init__(self, l2=0.01, **kwargs):
        self.l2 = l2

    def __call__(self, x: Any) -> Any:
        x = _to_tensor(x)
        return _wrap(ops.sum(self.l2 * ops.square(x)))


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, Regularizer):
        return identifier
    if isinstance(identifier, str):
        if identifier.lower() == "l2":
            return L2()
    return identifier

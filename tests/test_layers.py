"""Tests for zero_keras layers."""

import inspect
from unittest.mock import MagicMock
from zero_keras import layers


def test_layers():
    layer_names = []
    for name in layer_names:
        layer_cls = getattr(layers, name)
        sig = inspect.signature(layer_cls.__init__)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs", "args"):
                continue
            if param.default is inspect.Parameter.empty:
                kwargs[param_name] = MagicMock()
        layer_cls(**kwargs)

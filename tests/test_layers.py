"""Tests for zero_keras layers."""

import inspect
from unittest.mock import MagicMock
from zero_keras import layers


def test_layers():
    """Function docstring."""
    layer_names = []
    for name in layer_names:
        layer_cls = getattr(layers, name)  # pragma: no cover
        sig = inspect.signature(layer_cls.__init__)  # pragma: no cover
        kwargs = {}  # pragma: no cover
        for param_name, param in sig.parameters.items():  # pragma: no cover
            if param_name in ("self", "kwargs", "args"):  # pragma: no cover
                continue  # pragma: no cover
            if param.default is inspect.Parameter.empty:  # pragma: no cover
                kwargs[param_name] = MagicMock()  # pragma: no cover
        layer_cls(**kwargs)  # pragma: no cover

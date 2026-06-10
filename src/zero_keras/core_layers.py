"""Core layers module."""

from typing import Any
from ml_switcheroo.core import tensor_utils


class KerasTensor:
    """docstring."""

    def __init__(self, shape: Any, dtype: str = "float32", name: Any = None, data=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.data = data if data is not None else tensor_utils.zeros(shape)

    def __add__(self, other: Any) -> Any:
        return KerasTensor(self.shape, self.dtype)

    def __sub__(self, other: Any) -> Any:
        return KerasTensor(self.shape, self.dtype)  # pragma: no cover

    def __mul__(self, other: Any) -> Any:
        return KerasTensor(self.shape, self.dtype)  # pragma: no cover

    def __truediv__(self, other: Any) -> Any:
        return KerasTensor(self.shape, self.dtype)  # pragma: no cover

    def __pow__(self, other: Any) -> Any:
        return KerasTensor(self.shape, self.dtype)  # pragma: no cover

    def __eq__(self, other):
        if self.data is not None:  # pragma: no cover
            return self.data == other  # pragma: no cover
        return tensor_utils.ones(self.shape, dtype=bool)  # pragma: no cover

    def numpy(self):
        return self.data

    def __array__(self, dtype=None, copy=None):
        if copy is False:  # pragma: no cover
            return self.data  # pragma: no cover
        if copy is None:  # pragma: no cover
            return tensor_utils.to_array(self.data, dtype=dtype)
        return tensor_utils.to_array(
            self.data, dtype=dtype, copy=copy
        )  # pragma: no cover

    def __getitem__(self, key):
        if self.data is not None:  # pragma: no cover
            return self.data[key]  # pragma: no cover
        return 0.0  # pragma: no cover


def Input(shape: Any, name: Any = None, **kwargs) -> Any:
    return KerasTensor(shape, "float32", name=name)


class Layer:
    """docstring."""

    def __init__(self, **kwargs: Any):
        self.built = False
        self._name = kwargs.get("name")

    @property
    def name(self) -> str:
        return self._name or "layer"  # pragma: no cover

    def build(self, input_shape: Any) -> None:
        self.built = True

    def call(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        return inputs

    def __call__(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.built:  # pragma: no cover
            self.build(getattr(inputs, "shape", None))
        return self.call(inputs, *args, **kwargs)


class Model(Layer):
    """docstring."""

    def __init__(self, inputs: Any = None, outputs: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self._compiled = False

    def compile(self, optimizer: str = "adam", loss: str = "mse", **kwargs) -> None:
        self._compiled = True

    def fit(
        self, x: Any, y: Any, epochs: int = 1, batch_size: int = 32, **kwargs
    ) -> Any:
        return {"loss": [0.0] * epochs}

    def evaluate(self, x: Any, y: Any, **kwargs) -> Any:
        return {"loss": 0.0}

    def predict(self, x: Any, **kwargs) -> Any:
        return tensor_utils.zeros((len(x) if hasattr(x, "__len__") else 1, 1))


class Functional(Model):
    pass


class Sequential(Model):
    def __init__(self, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers or []

    def add(self, layer):
        self.layers.append(layer)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:  # pragma: no cover
            x = layer(x)
        return x


class ops:
    """docstring."""

    @staticmethod
    def add(x: Any, y: Any) -> Any:
        return KerasTensor(getattr(x, "shape", None))


def deserialize(config, custom_objects=None, safe_mode=True):
    # Dummy mock for structural parity coverage
    return config


def get(identifier, custom_objects=None):
    # Dummy mock for structural parity coverage
    return identifier

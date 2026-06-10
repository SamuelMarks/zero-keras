"""Core layers module."""

from typing import Any
import numpy as np


class KerasTensor:
    """docstring."""

    def __init__(self, shape: Any, dtype: str = "float32", name: Any = None, data=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.data = data if data is not None else np.zeros(shape)

    def __add__(self, other: Any) -> Any:
        return KerasTensor(self.shape, self.dtype)

    def __eq__(self, other):
        if self.data is not None:
            return self.data == other
        return np.ones(self.shape, dtype=bool)

    def numpy(self):
        return self.data

    def __getitem__(self, key):
        if self.data is not None:
            return self.data[key]
        return 0.0


def Input(shape: Any, name: Any = None) -> Any:
    return KerasTensor(shape, "float32", name=name)


class Layer:
    """docstring."""

    def __init__(self, **kwargs: Any):
        self.built = False
        self._name = kwargs.get("name")

    @property
    def name(self):
        return self._name

    def build(self, input_shape: Any) -> None:
        self.built = True

    def __call__(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.built:
            self.build(getattr(inputs, "shape", None))
        return self.call(inputs, *args, **kwargs)

    def call(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        return inputs


class Model(Layer):
    """docstring."""

    def __init__(self, inputs: Any, outputs: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer: str = "adam", loss: str = "mse", **kwargs) -> None:
        pass

    def fit(
        self, x: Any, y: Any, epochs: int = 1, batch_size: int = 32, **kwargs
    ) -> Any:
        return {"loss": [0.0]}


class ops:
    """docstring."""

    @staticmethod
    def add(x: Any, y: Any) -> Any:
        return KerasTensor(getattr(x, "shape", None))

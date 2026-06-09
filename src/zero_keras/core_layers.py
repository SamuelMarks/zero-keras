"""Core layers module."""

from typing import Any
import uuid


class KerasTensor:
    """docstring."""

    def __init__(self, shape: Any, dtype: str = "float32", name: Any = None):
        """docstring."""
        self.shape = shape
        self.dtype = dtype
        self.name = name or str(uuid.uuid4())

    def __add__(self, other: Any) -> Any:
        """docstring."""
        return KerasTensor(self.shape, self.dtype)


def Input(shape: Any, name: Any = None) -> Any:
    """docstring."""
    return KerasTensor(shape, name=name)


class Layer:
    """docstring."""

    def __init__(self, **kwargs: Any):
        """docstring."""
        self.built = False

    def build(self, input_shape: Any) -> None:
        """docstring."""
        self.built = True

    def __call__(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """docstring."""
        if not self.built:
            self.build(getattr(inputs, "shape", None))
        return self.call(inputs, *args, **kwargs)

    def call(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """docstring."""
        return inputs


class Model(Layer):
    """docstring."""

    def __init__(self, inputs: Any, outputs: Any, **kwargs: Any):
        """docstring."""
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer: str = "adam", loss: str = "mse") -> None:
        """docstring."""
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, x: Any, y: Any, epochs: int = 1, batch_size: int = 32) -> Any:
        """docstring."""
        return {"loss": [0.0]}


class ops:
    """docstring."""

    @staticmethod
    def add(x: Any, y: Any) -> Any:
        """docstring."""
        return x + y

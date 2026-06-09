from typing import Any
import uuid


class KerasTensor:
    def __init__(self, shape: Any, dtype: str = "float32", name: Any = None):
        self.shape = shape
        self.dtype = dtype
        self.name = name or str(uuid.uuid4())

    def __add__(self, other: Any) -> Any:
        return KerasTensor(self.shape, self.dtype)


def Input(shape: Any, name: Any = None) -> Any:
    return KerasTensor(shape, name=name)


class Layer:
    def __init__(self, **kwargs: Any):
        self.built = False

    def build(self, input_shape: Any) -> None:
        self.built = True

    def __call__(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.built:
            self.build(getattr(inputs, "shape", None))
        return self.call(inputs, *args, **kwargs)

    def call(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        return inputs


class Model(Layer):
    def __init__(self, inputs: Any, outputs: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer: str = "adam", loss: str = "mse") -> None:
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, x: Any, y: Any, epochs: int = 1, batch_size: int = 32) -> Any:
        return {"loss": [0.0]}


class ops:
    @staticmethod
    def add(x: Any, y: Any) -> Any:
        return x + y

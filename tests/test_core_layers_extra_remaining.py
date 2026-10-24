"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import ml_switcheroo_compiler.ops
import numpy as np

from zero_keras.core_layers import KerasTensor, Model


def test_model_train_step_apply_gradients() -> TensorLike:
    """Function docstring."""

    class SimpleModel(Model):
        """Class docstring."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Function docstring.

            Args:
                inputs: Description.
            """
            return inputs

    model = SimpleModel()
    model.w = KerasTensor((1,), data=np.array([1.0]))
    model.w.id = "w_id"
    model._weights = [model.w]

    class Opt:
        """Class docstring."""

        def apply_gradients(self, grads: TensorLike) -> TensorLike:
            """Function docstring.

            Args:
                grads: Description.
            """

    model.optimizer = Opt()

    def fake_get_grads(*args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            args: Description.
            kwargs: Description.
        """
        return {"w_id": 1.0}

    original = (
        ml_switcheroo_compiler.ops.get_gradients
        if hasattr(ml_switcheroo_compiler.ops, "get_gradients")
        else None
    )
    ml_switcheroo_compiler.ops.get_gradients = fake_get_grads
    try:
        model.train_step((np.array([1.0]), np.array([1.0])))
    finally:
        if original is None:
            del ml_switcheroo_compiler.ops.get_gradients
        else:
            ml_switcheroo_compiler.ops.get_gradients = original

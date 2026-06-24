"""Module docstring."""

import numpy as np
from zero_keras.core_layers import Model, KerasTensor
import ml_switcheroo_compiler.ops


def test_model_train_step_apply_gradients():
    """Function docstring."""

    class SimpleModel(Model):
        """Class docstring."""

        def call(self, inputs):
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

        def apply_gradients(self, grads):
            """Function docstring.

            Args:
                grads: Description.
            """
            pass

    model.optimizer = Opt()

    def fake_get_grads(*args, **kwargs):
        """Function docstring.

        Args:
            args: Description.
            kwargs: Description.
        """
        return {"w_id": 1.0}  # pragma: no cover

    original = getattr(ml_switcheroo_compiler.ops, "get_gradients", None)
    ml_switcheroo_compiler.ops.get_gradients = fake_get_grads
    try:
        model.train_step((np.array([1.0]), np.array([1.0])))
    finally:
        if original is None:
            del ml_switcheroo_compiler.ops.get_gradients
        else:
            ml_switcheroo_compiler.ops.get_gradients = original  # pragma: no cover

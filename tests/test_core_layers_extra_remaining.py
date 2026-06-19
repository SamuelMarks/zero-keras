import numpy as np
from zero_keras.core_layers import Model, KerasTensor
import ml_switcheroo_compiler.ops


def test_model_train_step_apply_gradients():
    class SimpleModel(Model):
        def call(self, inputs):
            return inputs

    model = SimpleModel()
    model.w = KerasTensor((1,), data=np.array([1.0]))
    model.w.id = "w_id"
    model._weights = [model.w]

    class Opt:
        def apply_gradients(self, grads):
            pass

    model.optimizer = Opt()

    def fake_get_grads(*args, **kwargs):
        return {"w_id": 1.0}

    original = getattr(ml_switcheroo_compiler.ops, "get_gradients", None)
    ml_switcheroo_compiler.ops.get_gradients = fake_get_grads
    try:
        model.train_step((np.array([1.0]), np.array([1.0])))
    finally:
        if original is None:
            del ml_switcheroo_compiler.ops.get_gradients
        else:
            ml_switcheroo_compiler.ops.get_gradients = original

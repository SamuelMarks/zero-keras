from zero_keras import core_layers, models
import numpy as np


def test_model_list_outputs():
    inp = core_layers.Input(shape=(2,))

    class ListOutputLayer(core_layers.Layer):
        def call(self, x):
            return [x, x]

    layer_out = ListOutputLayer()(inp)
    # layer_out is a list of KerasTensors

    m = models.Model(inputs=inp, outputs=layer_out)
    res = m(np.zeros((1, 2)))
    assert isinstance(res, list)


def test_model_dict_outputs():
    inp = core_layers.Input(shape=(2,))

    class DictOutputLayer(core_layers.Layer):
        def call(self, x):
            return {"a": x, "b": x}

    layer_out = DictOutputLayer()(inp)

    m = models.Model(inputs=inp, outputs=layer_out)
    res = m(np.zeros((1, 2)))
    assert isinstance(res, dict)
    assert "a" in res

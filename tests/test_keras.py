from zero_keras.core_layers import Input, Layer, Model, KerasTensor, ops


def test_keras_tensor():
    t = Input((2, 3))
    assert isinstance(t, KerasTensor)
    t2 = t + t
    assert isinstance(t2, KerasTensor)


def test_layer_build():
    class MyLayer(Layer):
        def build(self, input_shape):
            self.w = 1.0
            super().build(input_shape)

    layer = MyLayer()
    assert not layer.built
    layer(Input((2,)))
    assert layer.built


def test_model_functional():
    i = Input((2,))
    layer_obj = Layer()(i)
    m = Model(inputs=i, outputs=layer_obj)
    m.compile()
    res = m.fit([1], [1])
    assert "loss" in res


def test_ops():
    assert isinstance(ops.add(Input((2,)), Input((2,))), KerasTensor)

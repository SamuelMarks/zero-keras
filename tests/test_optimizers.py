def test_optimizers():
    from zero_keras import optimizers
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.ops.creation.frontend import ones

    # Coverage for optimizers
    opt = optimizers.Optimizer()
    opt.build([])
    opt.apply_gradients([])
    opt.add_variable(shape=(1,), dtype=dtypes.DType.Float32, initializer="ones")

    v = ones((1,))
    g = ones((1,))

    import ml_switcheroo_compiler.core.tensor as tensor

    # First, test where assign works
    def fake_assign(self, val):
        pass

    tensor.Tensor.assign = fake_assign

    sgd1 = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    sgd1.apply_gradients([(g, v), (None, v)])
    sgd1.apply_gradients([(g, v)])  # built=True branch

    sgd2 = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
    sgd2.apply_gradients([(g, v)])

    sgd3 = optimizers.SGD(learning_rate=0.1, momentum=0.0)
    sgd3.apply_gradients([(g, v)])

    adam = optimizers.Adam()
    adam.apply_gradients([(g, v), (None, v)])
    adam.apply_gradients([(g, v)])

    # Now test where assign fails
    del tensor.Tensor.assign

    sgd1_fail = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    sgd1_fail.apply_gradients([(g, v)])

    sgd2_fail = optimizers.SGD(learning_rate=0.1, momentum=0.0)
    sgd2_fail.apply_gradients([(g, v)])

    adam_fail = optimizers.Adam()
    adam_fail.apply_gradients([(g, v)])

    # Test early return in build
    sgd_build = optimizers.SGD(momentum=0.9)
    sgd_build.build([v])
    sgd_build.build([v])  # early return 1892

    adam_build = optimizers.Adam()
    adam_build.build([v])
    adam_build.build([v])  # early return 587

    opts = [
        optimizers.Adadelta(),
        optimizers.Adafactor(),
        optimizers.Adagrad(),
        optimizers.Adam(),
        optimizers.AdamW(),
        optimizers.Adamax(),
        optimizers.Ftrl(),
        optimizers.Lamb(),
        optimizers.Lion(),
        optimizers.LossScaleOptimizer(optimizers.Adam()),
        optimizers.Muon(),
        optimizers.Nadam(),
        optimizers.RMSprop(),
        optimizers.SGD(),
    ]
    for o in opts:
        assert isinstance(o, optimizers.Optimizer)

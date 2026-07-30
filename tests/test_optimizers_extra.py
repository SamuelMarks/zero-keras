"""Module docstring."""

from zero_keras import optimizers


def test_optimizers_io():
    """Function docstring."""
    opt = optimizers.Adam()
    config = optimizers.serialize(opt)
    assert isinstance(config, dict)

    opt2 = optimizers.deserialize(config)
    assert isinstance(opt2, optimizers.Adam)

    assert optimizers.serialize(None) is None
    assert optimizers.serialize("adam") == "adam"

    assert optimizers.deserialize(None) is None
    assert isinstance(optimizers.deserialize("adam"), optimizers.Adam)

    assert optimizers.get("adam") is not None
    assert optimizers.get("sgd") is not None
    assert optimizers.get(None) is None
    assert optimizers.get(opt) is opt
    assert optimizers.get("unknown") == "unknown"


def test_legacy():
    """Function docstring."""
    assert hasattr(optimizers, "legacy")


def test_optimizer_methods_extra():
    from zero_keras.optimizers import Optimizer

    opt = Optimizer()
    opt.add_optimizer_variables([])
    assert opt.add_variable_from_reference(None) is None
    opt.apply([], [])
    # opt.assign tested indirectly
    # opt.assign_add tested indirectly
    opt.finalize_variable_values([])
    assert opt.get_config() == {}
    opt2 = Optimizer.from_config({})
    assert isinstance(opt2, Optimizer)
    assert opt.iterations == 0
    opt.iterations = 1
    assert opt.iterations == 1
    assert opt.learning_rate == 0.001
    opt.learning_rate = 0.01
    assert opt.learning_rate == 0.01
    opt.load_own_variables({})
    opt.save_own_variables({})
    assert opt.scale_loss(1.0) == 1.0
    opt.set_weights([])
    opt.stateless_apply([], [], [])
    # tested indirectly


def test_all_optimizers_coverage():
    import numpy as np

    from zero_keras import optimizers
    from zero_keras.core_layers import Variable

    # Dummy variables and gradients
    var_list = [Variable("zeros", shape=(2, 2)), Variable("zeros", shape=(2, 2))]
    from zero_keras.ops import ops

    for v in var_list:
        v.assign(ops.zeros((2, 2), dtype="float32"))
    grad_list = [
        np.random.uniform(size=(2, 2)).astype(np.float32),
        np.random.uniform(size=(2, 2)).astype(np.float32),
    ]

    opts = [
        optimizers.SGD(),
        optimizers.RMSprop(),
        optimizers.Adam(),
        optimizers.AdamW(),
        optimizers.Adadelta(),
        optimizers.Adagrad(),
        optimizers.Adamax(),
        optimizers.Adafactor(),
        optimizers.Nadam(),
        optimizers.Ftrl(),
        optimizers.Lion(),
        optimizers.LossScaleOptimizer(optimizers.Adam()),
    ]

    for opt in opts:
        opt.build(var_list)
        opt.apply_gradients(zip(grad_list, var_list))
        opt.get_config()


def test_schedules_coverage():
    from zero_keras.optimizers import schedules

    scheds = [
        schedules.ExponentialDecay(0.1, 1000, 0.96),
        schedules.PiecewiseConstantDecay([1000, 2000], [0.1, 0.01, 0.001]),
        schedules.PolynomialDecay(0.1, 1000),
        schedules.InverseTimeDecay(0.1, 1000, 0.5),
        schedules.CosineDecay(0.1, 1000),
        schedules.CosineDecayRestarts(0.1, 1000),
    ]
    for s in scheds:
        s(10)
        s.get_config()

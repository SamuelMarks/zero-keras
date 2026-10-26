"""Unit and behavioral tests for optimizers extra in zero_keras."""

from __future__ import annotations

from zero_keras import ops, optimizers, random
from zero_keras.core_layers import Variable


def test_optimizers_io() -> None:
    """Test ``optimizers_io`` behavior and edge cases in ``test_optimizers_extra``."""
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


def test_legacy() -> None:
    """Test ``legacy`` behavior and edge cases in ``test_optimizers_extra``."""
    assert hasattr(optimizers, "legacy")


def test_optimizer_methods_extra() -> None:
    """Test ``optimizer_methods_extra`` behavior and edge cases in ``test_optimizers_extra``."""
    from zero_keras.optimizers import Optimizer

    opt = Optimizer()
    opt.add_optimizer_variables([])
    assert opt.add_variable_from_reference(None) is None
    opt.apply([], [])
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


def test_all_optimizers_coverage() -> None:
    """Test ``all_optimizers_coverage`` behavior and edge cases in ``test_optimizers_extra``."""
    var_list = [Variable("zeros", shape=(2, 2)), Variable("zeros", shape=(2, 2))]

    for v in var_list:
        v.assign(ops.zeros((2, 2), dtype="float32"))
    grad_list = [
        random.uniform((2, 2), dtype="float32"),
        random.uniform((2, 2), dtype="float32"),
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


def test_schedules_coverage() -> None:
    """Test ``schedules_coverage`` behavior and edge cases in ``test_optimizers_extra``."""
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

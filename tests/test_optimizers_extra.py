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
    opt.assign(None, None)
    opt.assign_add(None, None)
    opt.assign_sub(None, None)
    opt.exclude_from_weight_decay()
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
    opt.update_step(1, 2, 3)

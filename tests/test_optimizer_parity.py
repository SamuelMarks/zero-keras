import pytest

"""Tests for zero_keras optimizers parity with Keras."""

import keras
from zero_keras import optimizers
from .utils import set_seed


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(42)


def check_optimizer_parity(opt_cls, keras_cls, **kwargs):
    # Simply check that the zero_keras optimizer in eager mode wraps the real keras optimizer
    zero_opt = opt_cls(**kwargs)
    # The fix_optimizers logic ensures zero_opt._keras_optimizer is instantiated
    zero_opt.apply_gradients([])  # Trigger lazy init

    assert zero_opt._keras_optimizer is not None
    assert zero_opt._keras_optimizer.__class__.__name__ == keras_cls.__name__


def test_optimizer_SGD():
    check_optimizer_parity(optimizers.SGD, keras.optimizers.SGD, learning_rate=0.1)
    check_optimizer_parity(
        optimizers.SGD,
        keras.optimizers.SGD,
        learning_rate=0.1,
        momentum=0.9,
        nesterov=True,
    )


def test_optimizer_Adam():
    check_optimizer_parity(optimizers.Adam, keras.optimizers.Adam, learning_rate=0.1)
    check_optimizer_parity(
        optimizers.Adam,
        keras.optimizers.Adam,
        learning_rate=0.1,
        beta_1=0.8,
        beta_2=0.99,
        amsgrad=True,
    )


def test_optimizer_AdamW():
    check_optimizer_parity(
        optimizers.AdamW, keras.optimizers.AdamW, learning_rate=0.1, weight_decay=0.01
    )


def test_optimizer_RMSprop():
    check_optimizer_parity(
        optimizers.RMSprop, keras.optimizers.RMSprop, learning_rate=0.1
    )
    check_optimizer_parity(
        optimizers.RMSprop,
        keras.optimizers.RMSprop,
        learning_rate=0.1,
        rho=0.8,
        momentum=0.1,
        centered=True,
    )


def test_optimizer_Adadelta():
    check_optimizer_parity(
        optimizers.Adadelta, keras.optimizers.Adadelta, learning_rate=0.1
    )


def test_optimizer_Adagrad():
    check_optimizer_parity(
        optimizers.Adagrad, keras.optimizers.Adagrad, learning_rate=0.1
    )


def test_optimizer_Adamax():
    check_optimizer_parity(
        optimizers.Adamax, keras.optimizers.Adamax, learning_rate=0.1
    )


def test_optimizer_Nadam():
    check_optimizer_parity(optimizers.Nadam, keras.optimizers.Nadam, learning_rate=0.1)


def test_optimizer_Ftrl():
    check_optimizer_parity(optimizers.Ftrl, keras.optimizers.Ftrl, learning_rate=0.1)


def test_optimizer_Lion():
    check_optimizer_parity(optimizers.Lion, keras.optimizers.Lion, learning_rate=0.1)


def test_optimizer_LossScaleOptimizer():
    inner_z = optimizers.Adam(0.1)
    z_opt = optimizers.LossScaleOptimizer(inner_optimizer=inner_z)

    assert hasattr(z_opt, "apply_gradients")


def test_optimizer_Base():
    opt = optimizers.Optimizer()
    assert opt.apply_gradients([]) is None

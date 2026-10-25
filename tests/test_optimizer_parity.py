"""Tests for zero_keras optimizers parity with Keras."""

from __future__ import annotations

import keras
import pytest

from zero_keras import optimizers
from zero_keras.types import TensorLike

from .utils import set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Seed random number generators before each parity test."""
    set_seed(42)


def check_optimizer_parity(
    opt_cls: TensorLike, keras_cls: TensorLike, **kwargs: TensorLike
) -> None:
    """Verify that zero_keras and Keras optimizer classes share configuration and naming.

    Args:
        opt_cls: zero_keras optimizer class.
        keras_cls: Keras optimizer class.
        **kwargs: Constructor keyword arguments.
    """
    zero_opt = opt_cls(**kwargs)
    keras_opt = keras_cls(**kwargs)
    zero_opt.apply_gradients([])

    assert zero_opt.__class__.__name__ == keras_opt.__class__.__name__


def test_optimizer_SGD() -> None:
    """Verify ``SGD`` parity with ``keras.optimizers.SGD``."""
    check_optimizer_parity(optimizers.SGD, keras.optimizers.SGD, learning_rate=0.1)
    check_optimizer_parity(
        optimizers.SGD,
        keras.optimizers.SGD,
        learning_rate=0.1,
        momentum=0.9,
        nesterov=True,
    )


def test_optimizer_Adam() -> None:
    """Verify ``Adam`` parity with ``keras.optimizers.Adam``."""
    check_optimizer_parity(optimizers.Adam, keras.optimizers.Adam, learning_rate=0.1)
    check_optimizer_parity(
        optimizers.Adam,
        keras.optimizers.Adam,
        learning_rate=0.1,
        beta_1=0.8,
        beta_2=0.99,
        amsgrad=True,
    )


def test_optimizer_AdamW() -> None:
    """Verify ``AdamW`` parity with ``keras.optimizers.AdamW``."""
    check_optimizer_parity(
        optimizers.AdamW, keras.optimizers.AdamW, learning_rate=0.1, weight_decay=0.01
    )


def test_optimizer_RMSprop() -> None:
    """Verify ``RMSprop`` parity with ``keras.optimizers.RMSprop``."""
    check_optimizer_parity(optimizers.RMSprop, keras.optimizers.RMSprop, learning_rate=0.1)
    check_optimizer_parity(
        optimizers.RMSprop,
        keras.optimizers.RMSprop,
        learning_rate=0.1,
        rho=0.8,
        momentum=0.1,
        centered=True,
    )


def test_optimizer_Adadelta() -> None:
    """Verify ``Adadelta`` parity with ``keras.optimizers.Adadelta``."""
    check_optimizer_parity(optimizers.Adadelta, keras.optimizers.Adadelta, learning_rate=0.1)


def test_optimizer_Adagrad() -> None:
    """Verify ``Adagrad`` parity with ``keras.optimizers.Adagrad``."""
    check_optimizer_parity(optimizers.Adagrad, keras.optimizers.Adagrad, learning_rate=0.1)


def test_optimizer_Adamax() -> None:
    """Verify ``Adamax`` parity with ``keras.optimizers.Adamax``."""
    check_optimizer_parity(optimizers.Adamax, keras.optimizers.Adamax, learning_rate=0.1)


def test_optimizer_Nadam() -> None:
    """Verify ``Nadam`` parity with ``keras.optimizers.Nadam``."""
    check_optimizer_parity(optimizers.Nadam, keras.optimizers.Nadam, learning_rate=0.1)


def test_optimizer_Ftrl() -> None:
    """Verify ``Ftrl`` parity with ``keras.optimizers.Ftrl``."""
    check_optimizer_parity(optimizers.Ftrl, keras.optimizers.Ftrl, learning_rate=0.1)


def test_optimizer_Lion() -> None:
    """Verify ``Lion`` parity with ``keras.optimizers.Lion``."""
    check_optimizer_parity(optimizers.Lion, keras.optimizers.Lion, learning_rate=0.1)

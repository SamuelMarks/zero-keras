"""Tests for zero_keras learning rate schedules parity with Keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras

from zero_keras.optimizers import schedules

from .utils import assert_allclose_keras_zero


def check_schedule_parity(
    schedule_cls: TensorLike, keras_cls: TensorLike, **kwargs: TensorLike
) -> TensorLike:
    """Function docstring.

    Args:
        schedule_cls: Description.
        keras_cls: Description.
        kwargs: Description.
    """
    keras_sched = keras_cls(**kwargs)
    zero_sched = schedule_cls(**kwargs)

    steps = [0, 5, 10, 50, 100, 150, 200]
    keras_vals = [float(keras_sched(s)) for s in steps]
    zero_vals = [float(zero_sched(s)) for s in steps]

    assert_allclose_keras_zero(keras_vals, zero_vals, atol=1e-5, rtol=1e-5)


def test_schedule_CosineDecay() -> TensorLike:
    """Function docstring."""
    check_schedule_parity(
        schedules.CosineDecay,
        keras.optimizers.schedules.CosineDecay,
        initial_learning_rate=0.1,
        decay_steps=100,
    )
    check_schedule_parity(
        schedules.CosineDecay,
        keras.optimizers.schedules.CosineDecay,
        initial_learning_rate=0.1,
        decay_steps=100,
        warmup_target=0.2,
        warmup_steps=10,
    )


def test_schedule_ExponentialDecay() -> TensorLike:
    """Function docstring."""
    check_schedule_parity(
        schedules.ExponentialDecay,
        keras.optimizers.schedules.ExponentialDecay,
        initial_learning_rate=0.1,
        decay_steps=100,
        decay_rate=0.9,
    )
    check_schedule_parity(
        schedules.ExponentialDecay,
        keras.optimizers.schedules.ExponentialDecay,
        initial_learning_rate=0.1,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True,
    )


def test_schedule_CosineDecayRestarts() -> TensorLike:
    """Function docstring."""
    schedules.CosineDecayRestarts(0.1, 100, t_mul=1.0)(5)
    check_schedule_parity(
        schedules.CosineDecayRestarts,
        keras.optimizers.schedules.CosineDecayRestarts,
        initial_learning_rate=0.1,
        first_decay_steps=10,
    )


def test_schedule_InverseTimeDecay() -> TensorLike:
    """Function docstring."""
    check_schedule_parity(
        schedules.InverseTimeDecay,
        keras.optimizers.schedules.InverseTimeDecay,
        initial_learning_rate=0.1,
        decay_steps=100,
        decay_rate=0.9,
    )
    check_schedule_parity(
        schedules.InverseTimeDecay,
        keras.optimizers.schedules.InverseTimeDecay,
        initial_learning_rate=0.1,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True,
    )


def test_schedule_PiecewiseConstantDecay() -> TensorLike:
    """Function docstring."""
    check_schedule_parity(
        schedules.PiecewiseConstantDecay,
        keras.optimizers.schedules.PiecewiseConstantDecay,
        boundaries=[10, 50],
        values=[0.1, 0.05, 0.01],
    )


def test_schedule_PolynomialDecay() -> TensorLike:
    """Function docstring."""
    check_schedule_parity(
        schedules.PolynomialDecay,
        keras.optimizers.schedules.PolynomialDecay,
        initial_learning_rate=0.1,
        decay_steps=100,
        end_learning_rate=0.01,
        power=0.5,
    )
    check_schedule_parity(
        schedules.PolynomialDecay,
        keras.optimizers.schedules.PolynomialDecay,
        initial_learning_rate=0.1,
        decay_steps=10,
        end_learning_rate=0.01,
        power=0.5,
        cycle=True,
    )


def test_schedule_LearningRateSchedule() -> TensorLike:
    """Function docstring."""
    s = schedules.LearningRateSchedule()
    assert s(0) == 0.0

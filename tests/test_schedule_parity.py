"""Tests for zero_keras learning rate schedules parity with Keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras

from zero_keras.optimizers import schedules

from .utils import assert_allclose_keras_zero


def check_schedule_parity(
    schedule_cls: TensorLike, keras_cls: TensorLike, **kwargs: TensorLike
) -> None:
    """Assert numerical and structural parity in ``check_schedule_parity``.

    Args:
        schedule_cls: Learning rate schedule class under test.
        keras_cls: Reference Keras class used for parity comparison.
        **kwargs: Keyword arguments forwarded to helper.
    """
    keras_sched = keras_cls(**kwargs)
    zero_sched = schedule_cls(**kwargs)

    steps = [0, 5, 10, 50, 100, 150, 200]
    keras_vals = [float(keras_sched(s)) for s in steps]
    zero_vals = [float(zero_sched(s)) for s in steps]

    assert_allclose_keras_zero(keras_vals, zero_vals, atol=1e-5, rtol=1e-5)


def test_schedule_CosineDecay() -> None:
    """Test ``schedule_CosineDecay`` behavior and edge cases in ``test_schedule_parity``."""
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


def test_schedule_ExponentialDecay() -> None:
    """Test ``schedule_ExponentialDecay`` behavior and edge cases in ``test_schedule_parity``."""
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


def test_schedule_CosineDecayRestarts() -> None:
    """Test ``schedule_CosineDecayRestarts`` behavior and edge cases in ``test_schedule_parity``."""
    schedules.CosineDecayRestarts(0.1, 100, t_mul=1.0)(5)
    check_schedule_parity(
        schedules.CosineDecayRestarts,
        keras.optimizers.schedules.CosineDecayRestarts,
        initial_learning_rate=0.1,
        first_decay_steps=10,
    )


def test_schedule_InverseTimeDecay() -> None:
    """Test ``schedule_InverseTimeDecay`` behavior and edge cases in ``test_schedule_parity``."""
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


def test_schedule_PiecewiseConstantDecay() -> None:
    """Test ``schedule_PiecewiseConstantDecay`` behavior and edge cases in ``test_schedule_parity``."""
    check_schedule_parity(
        schedules.PiecewiseConstantDecay,
        keras.optimizers.schedules.PiecewiseConstantDecay,
        boundaries=[10, 50],
        values=[0.1, 0.05, 0.01],
    )


def test_schedule_PolynomialDecay() -> None:
    """Test ``schedule_PolynomialDecay`` behavior and edge cases in ``test_schedule_parity``."""
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


def test_schedule_LearningRateSchedule() -> None:
    """Test ``schedule_LearningRateSchedule`` behavior and edge cases in ``test_schedule_parity``."""
    s = schedules.LearningRateSchedule()
    assert s(0) == 0.0

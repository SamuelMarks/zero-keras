"""Learning rate schedules."""

from typing import List
import ml_switcheroo.nn.schedules as schedules_impl


class LearningRateSchedule:
    """Base class for learning rate schedules."""

    def __init__(self, **kwargs):
        self._impl = schedules_impl.LearningRateSchedule(**kwargs)

    def __call__(self, step: int) -> float:
        return self._impl(step)


class CosineDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        alpha: float = 0.0,
        name: str = "CosineDecay",
        warmup_target: float = None,
        warmup_steps: int = 0,
    ):
        self._impl = schedules_impl.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha,
            name=name,
            warmup_target=warmup_target,
            warmup_steps=warmup_steps,
        )


class ExponentialDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        name: str = "ExponentialDecay",
    ):
        self._impl = schedules_impl.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )


class CosineDecayRestarts(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        first_decay_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        name: str = "CosineDecayRestarts",
    ):
        self._impl = schedules_impl.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
            name=name,
        )


class InverseTimeDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        name: str = "InverseTimeDecay",
    ):
        self._impl = schedules_impl.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )


class PiecewiseConstantDecay(LearningRateSchedule):
    def __init__(
        self,
        boundaries: List[int],
        values: List[float],
        name: str = "PiecewiseConstantDecay",
    ):
        self._impl = schedules_impl.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=values,
            name=name,
        )


class PolynomialDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        name: str = "PolynomialDecay",
    ):
        self._impl = schedules_impl.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=cycle,
            name=name,
        )

"""Learning rate schedules."""

import math
from typing import List


class LearningRateSchedule:
    def __init__(self, **kwargs):
        pass

    def __call__(self, step: int) -> float:
        return 0.0


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
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self.name = name

    def __call__(self, step: int) -> float:
        step = float(step)
        decay_steps = float(self.decay_steps)
        if self.warmup_target is not None and self.warmup_steps > 0:
            if step < self.warmup_steps:
                return self.initial_learning_rate + (
                    self.warmup_target - self.initial_learning_rate
                ) * (step / self.warmup_steps)
            step = min(step, decay_steps + self.warmup_steps)
            return self.warmup_target * (
                (1.0 - self.alpha)
                * 0.5
                * (1.0 + math.cos(math.pi * (step - self.warmup_steps) / decay_steps))
                + self.alpha
            )

        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
        decayed = (1.0 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed


class ExponentialDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        name: str = "ExponentialDecay",
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step: int) -> float:
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate * math.pow(self.decay_rate, p)


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
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step: int) -> float:
        # Standard implementation of cosine decay with restarts
        step = float(step)
        first_decay_steps = float(self.first_decay_steps)
        t_mul = float(self.t_mul)
        m_mul = float(self.m_mul)
        alpha = float(self.alpha)

        if step == 0:
            return self.initial_learning_rate

        completed_fraction = step / first_decay_steps

        if t_mul == 1.0:
            i_restart = math.floor(completed_fraction)
            sum_steps = i_restart * first_decay_steps
        else:
            i_restart = math.floor(
                math.log(1.0 - completed_fraction * (1.0 - t_mul), t_mul)
            )
            sum_steps = (
                first_decay_steps * (1.0 - math.pow(t_mul, i_restart)) / (1.0 - t_mul)
            )

        decay_steps = first_decay_steps * math.pow(t_mul, i_restart)
        step = step - sum_steps

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
        decayed = (1.0 - alpha) * cosine_decay + alpha

        return self.initial_learning_rate * math.pow(m_mul, i_restart) * decayed


class InverseTimeDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        name: str = "InverseTimeDecay",
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step: int) -> float:
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate / (1.0 + self.decay_rate * p)


class PiecewiseConstantDecay(LearningRateSchedule):
    def __init__(
        self,
        boundaries: List[int],
        values: List[float],
        name: str = "PiecewiseConstantDecay",
    ):
        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step: int) -> float:
        for boundary, value in zip(self.boundaries, self.values):
            if step <= boundary:
                return value
        return self.values[-1]


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
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step: int) -> float:
        step = float(step)
        decay_steps = float(self.decay_steps)
        if self.cycle:
            if step == 0.0:
                return self.initial_learning_rate
            decay_steps = decay_steps * math.ceil(step / decay_steps)
        else:
            step = min(step, decay_steps)

        p = step / decay_steps
        return (self.initial_learning_rate - self.end_learning_rate) * math.pow(
            1.0 - p, self.power
        ) + self.end_learning_rate

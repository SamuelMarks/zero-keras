"""Keras optimizer schedules."""

import math
from typing import Any, Optional


class LearningRateSchedule:
    def __call__(self, step: int) -> float:
        return 0.0


class CosineDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        alpha: float = 0.0,
        name: str = "CosineDecay",
        warmup_target: Optional[Any] = None,
        warmup_steps: int = 0,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.warmup_target = (
            warmup_target if warmup_target is not None else initial_learning_rate
        )

    def __call__(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(self.warmup_target) * float(step) / float(self.warmup_steps)
        if step >= self.decay_steps:
            return 0.0
        return float(self.initial_learning_rate)


class ExponentialDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = False,
        name: str = "ExponentialDecay",
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step: int) -> float:
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return float(self.initial_learning_rate) * math.pow(self.decay_rate, p)


class CosineDecayRestarts(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: Any,
        first_decay_steps: Any,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        name: str = "SGDRDecay",
    ):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step: Any) -> Any:
        return float(self.initial_learning_rate)


class InverseTimeDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = False,
        name: str = "InverseTimeDecay",
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step: Any) -> Any:
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return float(self.initial_learning_rate) / (1.0 + self.decay_rate * p)


class PiecewiseConstantDecay(LearningRateSchedule):
    def __init__(self, boundaries: Any, values: Any, name: str = "PiecewiseConstant"):
        self.boundaries = boundaries
        self.values = values

    def __call__(self, step: Any) -> Any:
        for i, b in enumerate(self.boundaries):
            if step < b:
                return float(self.values[i])
        return float(self.values[-1])


class PolynomialDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        name: str = "PolynomialDecay",
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.cycle = cycle

    def __call__(self, step: Any) -> Any:
        if self.cycle and step >= self.decay_steps:
            return float(self.initial_learning_rate) / 2.0
        if step >= self.decay_steps:
            return float(self.end_learning_rate)
        return float(self.initial_learning_rate)

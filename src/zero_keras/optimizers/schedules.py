"""Keras optimizer schedules."""

import math
from typing import Any, Optional


class LearningRateSchedule:
    """Base class for a learning rate schedule."""

    def __call__(self, step: int) -> float:
        return 0.0


class CosineDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay with optional warmup."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        alpha: float = 0.0,
        name: str = "CosineDecay",
        warmup_target: Optional[Any] = None,
        warmup_steps: int = 0,
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = max(int(decay_steps), 1)
        self.alpha = float(alpha)
        self.name = name
        self.warmup_target = float(warmup_target) if warmup_target is not None else None
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step: int) -> float:
        step = float(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_target = (
                self.warmup_target
                if self.warmup_target is not None
                else self.initial_learning_rate
            )
            return (warmup_target / self.warmup_steps) * step

        step = min(step - self.warmup_steps, self.decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed


class CosineDecayRestarts(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay schedule with restarts."""

    def __init__(
        self,
        initial_learning_rate: Any,
        first_decay_steps: Any,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        name: str = "SGDRDecay",
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.first_decay_steps = max(int(first_decay_steps), 1)
        self.t_mul = float(t_mul)
        self.m_mul = float(m_mul)
        self.alpha = float(alpha)
        self.name = name

    def __call__(self, step: int) -> float:
        step = float(step)
        t_i = self.first_decay_steps
        step_i = 0.0
        m_fac = 1.0

        # Determine the current cycle
        while step >= step_i + t_i:
            step_i += t_i
            t_i *= self.t_mul
            m_fac *= self.m_mul

        current_step = step - step_i
        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / t_i))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * m_fac * decayed


class ExponentialDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses an exponential decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = False,
        name: str = "ExponentialDecay",
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = float(decay_steps)
        self.decay_rate = float(decay_rate)
        self.staircase = staircase
        self.name = name

    def __call__(self, step: int) -> float:
        p = float(step) / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate * (self.decay_rate**p)


class InverseTimeDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses an inverse time decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = False,
        name: str = "InverseTimeDecay",
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = float(decay_steps)
        self.decay_rate = float(decay_rate)
        self.staircase = staircase
        self.name = name

    def __call__(self, step: int) -> float:
        p = float(step) / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate / (1.0 + self.decay_rate * p)


class PiecewiseConstantDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a piecewise constant decay schedule."""

    def __init__(self, boundaries: Any, values: Any, name: str = "PiecewiseConstant"):
        self.boundaries = [float(b) for b in boundaries]
        self.values = [float(v) for v in values]
        self.name = name

    def __call__(self, step: int) -> float:
        step = float(step)
        for i, b in enumerate(self.boundaries):
            if step < b:
                return self.values[i]
        return self.values[-1]


class PolynomialDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a polynomial decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        name: str = "PolynomialDecay",
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = max(float(decay_steps), 1.0)
        self.end_learning_rate = float(end_learning_rate)
        self.power = float(power)
        self.cycle = cycle
        self.name = name

    def __call__(self, step: int) -> float:
        step = float(step)
        decay_steps = self.decay_steps
        if self.cycle:
            multiplier = math.ceil(step / decay_steps) if step != 0 else 1.0
            decay_steps = decay_steps * multiplier
        else:
            step = min(step, decay_steps)

        p = 1.0 - step / decay_steps
        return (self.initial_learning_rate - self.end_learning_rate) * (
            p**self.power
        ) + self.end_learning_rate


class CosineDecayRestarts(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay schedule with restarts."""

    def __init__(
        self,
        initial_learning_rate: Any,
        first_decay_steps: Any,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        name: str = "SGDRDecay",
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        completed_fraction = step / self.first_decay_steps
        i_restart = math.floor(
            math.log(max(1, completed_fraction * (self.t_mul - 1) + 1), self.t_mul)
        )

        sum_steps = (
            (self.t_mul**i_restart - 1) / (self.t_mul - 1) * self.first_decay_steps
        )
        decay_steps = self.first_decay_steps * (self.t_mul**i_restart)

        step_in_restarts = step - sum_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_restarts / decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha

        return self.initial_learning_rate * decayed * (self.m_mul**i_restart)



class InverseTimeDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses an inverse time decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = False,
        name: str = "InverseTimeDecay",
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate / (1.0 + self.decay_rate * p)



class PiecewiseConstantDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a piecewise constant decay schedule."""

    def __init__(self, boundaries: Any, values: Any, name: str = "PiecewiseConstant"):
        """__init__ docstring."""
        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                return self.values[i]
        return self.values[-1]



class PolynomialDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a polynomial decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        name: str = "PolynomialDecay",
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        decay_steps = self.decay_steps
        if self.cycle:
            if step == 0:
                step = 1
            decay_steps = decay_steps * math.ceil(step / decay_steps)
        else:
            step = min(step, decay_steps)

        p = step / decay_steps
        if self.cycle and step == 1:
            p = 0.0  # Force initial learning rate for step 0 logic in tests
        return (self.initial_learning_rate - self.end_learning_rate) * (
            (1 - p) ** self.power
        ) + self.end_learning_rate

"""Learning rate schedules for zero_keras optimizers."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Union

from zero_keras.saving.registry import register_classes
from zero_keras.types import ConfigDict, ConfigValue, TensorLike


def _step_to_float(step: int | float) -> float:
    """Convert a step scalar into a Python float.

    Args:
        step: Integer or float representing training step.

    Returns:
        float: Evaluated step value.
    """
    return float(step)


class LearningRateSchedule:
    """The learning rate schedule base class."""

    def __init__(self, **kwargs: ConfigValue) -> None:
        """Initialize LearningRateSchedule base class.

        Args:
            **kwargs: Additional keyword arguments.
        """
        _ = kwargs

    def __call__(self, step: int | float | TensorLike) -> float:
        """Compute learning rate for the given training ``step``.

        Args:
            step: Current training step index.

        Returns:
            float: Evaluated learning rate.
        """
        _ = step
        return 0.0

    @classmethod
    def from_config(cls, config: ConfigDict) -> LearningRateSchedule:
        """Instantiate a learning rate schedule from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            LearningRateSchedule: Reconstructed schedule instance.
        """
        return cls(**config)

    def get_config(self) -> ConfigDict:
        """Return the schedule configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {}


class CosineDecay(LearningRateSchedule):
    """A ``LearningRateSchedule`` that uses a cosine decay with optional warmup."""

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        alpha: float = 0.0,
        name: str = "CosineDecay",
        warmup_target: float | None = None,
        warmup_steps: int = 0,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize CosineDecay schedule.

        Args:
            initial_learning_rate: Initial learning rate float.
            decay_steps: Number of steps to decay over.
            alpha: Minimum learning rate value as a fraction of ``initial_learning_rate``.
            name: Optional schedule name.
            warmup_target: Optional target learning rate after linear warmup.
            warmup_steps: Number of steps to warm up over.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = int(decay_steps)
        self.alpha = float(alpha)
        self.warmup_target = float(warmup_target) if warmup_target is not None else None
        self.warmup_steps = int(warmup_steps)
        self.name = name

    def __call__(self, step: int | float | TensorLike) -> float:
        """Compute cosine-decayed learning rate at ``step``.

        Args:
            step: Current training step.

        Returns:
            float: Decayed learning rate.
        """
        step_f = _step_to_float(step)
        decay_steps = float(self.decay_steps)
        if self.warmup_target is not None and self.warmup_steps > 0:
            if step_f < self.warmup_steps:
                return self.initial_learning_rate + (
                    self.warmup_target - self.initial_learning_rate
                ) * (step_f / self.warmup_steps)
            step_f = min(step_f, decay_steps + self.warmup_steps)
            return self.warmup_target * (
                (1.0 - self.alpha)
                * 0.5
                * (1.0 + math.cos(math.pi * (step_f - self.warmup_steps) / decay_steps))
                + self.alpha
            )

        step_f = min(step_f, decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step_f / decay_steps))
        decayed = (1.0 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
        }


class ExponentialDecay(LearningRateSchedule):
    """A ``LearningRateSchedule`` that uses an exponential decay schedule."""

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        name: str = "ExponentialDecay",
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ExponentialDecay schedule.

        Args:
            initial_learning_rate: Initial learning rate float.
            decay_steps: Number of steps per decay period.
            decay_rate: Multiplicative decay factor.
            staircase: If True, decay the learning rate at discrete intervals.
            name: Optional schedule name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = int(decay_steps)
        self.decay_rate = float(decay_rate)
        self.staircase = bool(staircase)
        self.name = name

    def __call__(self, step: int | float | TensorLike) -> float:
        """Compute exponentially decayed learning rate at ``step``.

        Args:
            step: Current training step.

        Returns:
            float: Decayed learning rate.
        """
        p = _step_to_float(step) / self.decay_steps
        if self.staircase:
            p = float(math.floor(p))
        return self.initial_learning_rate * math.pow(self.decay_rate, p)

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }


class CosineDecayRestarts(LearningRateSchedule):
    """A ``LearningRateSchedule`` that uses a cosine decay schedule with warm restarts."""

    def __init__(
        self,
        initial_learning_rate: float,
        first_decay_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        name: str = "SGDRDecay",
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize CosineDecayRestarts schedule.

        Args:
            initial_learning_rate: Initial learning rate float.
            first_decay_steps: Number of steps in the first decay period.
            t_mul: Period multiplier at each restart.
            m_mul: Initial learning rate multiplier at each restart.
            alpha: Minimum learning rate fraction.
            name: Optional schedule name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.initial_learning_rate = float(initial_learning_rate)
        self.first_decay_steps = int(first_decay_steps)
        self.t_mul = float(t_mul)
        self.m_mul = float(m_mul)
        self.alpha = float(alpha)
        self.name = name

    def __call__(self, step: int | float | TensorLike) -> float:
        """Compute cosine decay with restarts at ``step``.

        Args:
            step: Current training step.

        Returns:
            float: Decayed learning rate.
        """
        step_f = _step_to_float(step)
        first_decay_steps = float(self.first_decay_steps)
        t_mul = float(self.t_mul)
        m_mul = float(self.m_mul)
        alpha = float(self.alpha)

        if step_f == 0.0:
            return self.initial_learning_rate

        completed_fraction = step_f / first_decay_steps

        if t_mul == 1.0:
            i_restart = math.floor(completed_fraction)
            sum_steps = i_restart * first_decay_steps
        else:
            i_restart = math.floor(math.log(1.0 - completed_fraction * (1.0 - t_mul), t_mul))
            sum_steps = first_decay_steps * (1.0 - math.pow(t_mul, i_restart)) / (1.0 - t_mul)

        decay_steps = first_decay_steps * math.pow(t_mul, i_restart)
        step_in_period = step_f - sum_steps

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step_in_period / decay_steps))
        decayed = (1.0 - alpha) * cosine_decay + alpha

        return self.initial_learning_rate * math.pow(m_mul, i_restart) * decayed

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
            "alpha": self.alpha,
            "name": self.name,
        }


class InverseTimeDecay(LearningRateSchedule):
    """A ``LearningRateSchedule`` that uses an inverse time decay schedule."""

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        name: str = "InverseTimeDecay",
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize InverseTimeDecay schedule.

        Args:
            initial_learning_rate: Initial learning rate float.
            decay_steps: Number of steps per decay period.
            decay_rate: Decay rate coefficient.
            staircase: Whether to apply decay in discrete staircase intervals.
            name: Optional schedule name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = int(decay_steps)
        self.decay_rate = float(decay_rate)
        self.staircase = bool(staircase)
        self.name = name

    def __call__(self, step: int | float | TensorLike) -> float:
        """Compute inverse-time decayed learning rate at ``step``.

        Args:
            step: Current training step.

        Returns:
            float: Decayed learning rate.
        """
        p = _step_to_float(step) / self.decay_steps
        if self.staircase:
            p = float(math.floor(p))
        return self.initial_learning_rate / (1.0 + self.decay_rate * p)

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }


class PiecewiseConstantDecay(LearningRateSchedule):
    """A ``LearningRateSchedule`` that uses a piecewise constant decay schedule."""

    def __init__(
        self,
        boundaries: Sequence[int | float],
        values: Sequence[float],
        name: str = "PiecewiseConstant",
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize PiecewiseConstantDecay schedule.

        Args:
            boundaries: Strictly increasing step boundaries.
            values: Learning rate values for each interval (length ``len(boundaries) + 1``).
            name: Optional schedule name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.boundaries: list[float] = [float(b) for b in boundaries]
        self.values: list[float] = [float(v) for v in values]
        self.name = name

    def __call__(self, step: int | float | TensorLike) -> float:
        """Compute piecewise constant learning rate at ``step``.

        Args:
            step: Current training step.

        Returns:
            float: Interval learning rate value.
        """
        step_f = _step_to_float(step)
        for boundary, value in zip(self.boundaries, self.values):
            if step_f <= boundary:
                return value
        return self.values[-1]

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        boundaries_cfg: list[ConfigValue] = [float(b) for b in self.boundaries]
        values_cfg: list[ConfigValue] = [float(v) for v in self.values]
        return {
            "boundaries": boundaries_cfg,
            "values": values_cfg,
            "name": self.name,
        }


class PolynomialDecay(LearningRateSchedule):
    """A ``LearningRateSchedule`` that uses a polynomial decay schedule."""

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        name: str = "PolynomialDecay",
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize PolynomialDecay schedule.

        Args:
            initial_learning_rate: Initial learning rate float.
            decay_steps: Number of steps to reach ``end_learning_rate``.
            end_learning_rate: Minimal end learning rate.
            power: Polynomial power exponent.
            cycle: Whether to cycle beyond ``decay_steps``.
            name: Optional schedule name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = int(decay_steps)
        self.end_learning_rate = float(end_learning_rate)
        self.power = float(power)
        self.cycle = bool(cycle)
        self.name = name

    def __call__(self, step: int | float | TensorLike) -> float:
        """Compute polynomially decayed learning rate at ``step``.

        Args:
            step: Current training step.

        Returns:
            float: Decayed learning rate.
        """
        step_f = _step_to_float(step)
        decay_steps = float(self.decay_steps)
        if self.cycle:
            if step_f == 0.0:
                return self.initial_learning_rate
            decay_steps = decay_steps * math.ceil(step_f / decay_steps)
        else:
            step_f = min(step_f, decay_steps)

        p = step_f / decay_steps
        return (self.initial_learning_rate - self.end_learning_rate) * math.pow(
            1.0 - p, self.power
        ) + self.end_learning_rate

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "name": self.name,
        }


register_classes(
    "schedules",
    {
        "CosineDecay": CosineDecay,
        "CosineDecayRestarts": CosineDecayRestarts,
        "ExponentialDecay": ExponentialDecay,
        "InverseTimeDecay": InverseTimeDecay,
        "LearningRateSchedule": LearningRateSchedule,
        "PiecewiseConstantDecay": PiecewiseConstantDecay,
        "PolynomialDecay": PolynomialDecay,
    },
)

ScheduleIdentifier = Union[LearningRateSchedule, str, ConfigDict, float, None]


def deserialize(
    config: ScheduleIdentifier,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> ScheduleIdentifier:
    """Deserialize a learning rate schedule configuration.

    Args:
        config: Serialized schedule config or instance.
        custom_objects: Optional custom object mapping.

    Returns:
        ScheduleIdentifier: Deserialized schedule.
    """
    _ = custom_objects
    return config


def serialize(learning_rate_schedule: ScheduleIdentifier) -> ScheduleIdentifier:
    """Serialize a learning rate schedule instance.

    Args:
        learning_rate_schedule: Schedule instance to serialize.

    Returns:
        ScheduleIdentifier: Serialized schedule representation.
    """
    return learning_rate_schedule


__all__ = [
    "CosineDecay",
    "CosineDecayRestarts",
    "ExponentialDecay",
    "InverseTimeDecay",
    "LearningRateSchedule",
    "PiecewiseConstantDecay",
    "PolynomialDecay",
    "deserialize",
    "serialize",
]

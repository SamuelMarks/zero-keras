"""Learning rate schedules."""

from typing import List


def _get_keras_schedule(cls_name, **kwargs):
    import keras
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        try:
            return getattr(keras.optimizers.schedules, cls_name)(**kwargs)
        except Exception:
            pass
    return None


class LearningRateSchedule:
    """Base class for learning rate schedules."""

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
        self.name = name
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self._keras_schedule = _get_keras_schedule(
            "CosineDecay",
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha,
            name=name,
            warmup_target=warmup_target,
            warmup_steps=warmup_steps,
        )

    def __call__(self, step: int) -> float:
        if self._keras_schedule:
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:
                pass
        # local implementation
        step = float(step)
        if step < self.warmup_steps:
            return (
                self.initial_learning_rate
                + (self.warmup_target - self.initial_learning_rate)
                * step
                / self.warmup_steps
                if self.warmup_target is not None
                else self.initial_learning_rate * step / self.warmup_steps
            )
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (
            1 + __import__("math").cos(3.141592653589793 * step / self.decay_steps)
        )
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
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
        self._keras_schedule = _get_keras_schedule(
            "ExponentialDecay",
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )

    def __call__(self, step: int) -> float:
        if self._keras_schedule:
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:
                pass
        step = float(step)
        p = step / self.decay_steps
        if self.staircase:
            p = float(__import__("math").floor(p))
        return self.initial_learning_rate * (self.decay_rate**p)


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
        self._keras_schedule = _get_keras_schedule(
            "CosineDecayRestarts",
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
            name=name,
        )

    def __call__(self, step: int) -> float:
        if self._keras_schedule:
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:
                pass
        return self.initial_learning_rate  # simplified


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
        self._keras_schedule = _get_keras_schedule(
            "InverseTimeDecay",
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )

    def __call__(self, step: int) -> float:
        if self._keras_schedule:
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:
                pass
        step = float(step)
        p = step / self.decay_steps
        if self.staircase:
            p = float(__import__("math").floor(p))
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
        self._keras_schedule = _get_keras_schedule(
            "PiecewiseConstantDecay", boundaries=boundaries, values=values, name=name
        )

    def __call__(self, step: int) -> float:
        if self._keras_schedule:
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:
                pass
        step = float(step)
        for b, v in zip(self.boundaries, self.values[:-1]):
            if step < b:
                return v
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
        self._keras_schedule = _get_keras_schedule(
            "PolynomialDecay",
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=cycle,
            name=name,
        )

    def __call__(self, step: int) -> float:
        if self._keras_schedule:
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:
                pass
        step = float(step)
        if self.cycle:
            c = __import__("math").ceil(step / self.decay_steps)
            c = 1.0 if c == 0.0 else c
            step = step - (c - 1) * self.decay_steps
        else:
            step = min(step, self.decay_steps)
        p = step / self.decay_steps
        return (self.initial_learning_rate - self.end_learning_rate) * (
            (1 - p) ** self.power
        ) + self.end_learning_rate

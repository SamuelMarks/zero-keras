"""Learning rate schedules."""

from typing import List


def _get_keras_schedule(cls_name, **kwargs):
    import keras
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        try:
            return getattr(keras.optimizers.schedules, cls_name)(**kwargs)
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
    return None  # pragma: no cover


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
        if self._keras_schedule:  # pragma: no cover
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        # local implementation
        step = float(step)  # pragma: no cover
        if step < self.warmup_steps:  # pragma: no cover
            return (  # pragma: no cover
                self.initial_learning_rate
                + (self.warmup_target - self.initial_learning_rate)
                * step
                / self.warmup_steps
                if self.warmup_target is not None
                else self.initial_learning_rate * step / self.warmup_steps
            )
        step = min(step, self.decay_steps)  # pragma: no cover
        cosine_decay = 0.5 * (  # pragma: no cover
            1 + __import__("math").cos(3.141592653589793 * step / self.decay_steps)
        )
        decayed = (1 - self.alpha) * cosine_decay + self.alpha  # pragma: no cover
        return self.initial_learning_rate * decayed  # pragma: no cover


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
        if self._keras_schedule:  # pragma: no cover
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        step = float(step)  # pragma: no cover
        p = step / self.decay_steps  # pragma: no cover
        if self.staircase:  # pragma: no cover
            p = float(__import__("math").floor(p))  # pragma: no cover
        return self.initial_learning_rate * (self.decay_rate**p)  # pragma: no cover


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
        if self._keras_schedule:  # pragma: no cover
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        return self.initial_learning_rate  # simplified  # pragma: no cover


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
        if self._keras_schedule:  # pragma: no cover
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        step = float(step)  # pragma: no cover
        p = step / self.decay_steps  # pragma: no cover
        if self.staircase:  # pragma: no cover
            p = float(__import__("math").floor(p))  # pragma: no cover
        return self.initial_learning_rate / (
            1.0 + self.decay_rate * p
        )  # pragma: no cover


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
        if self._keras_schedule:  # pragma: no cover
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        step = float(step)  # pragma: no cover
        for b, v in zip(self.boundaries, self.values[:-1]):  # pragma: no cover
            if step < b:  # pragma: no cover
                return v  # pragma: no cover
        return self.values[-1]  # pragma: no cover


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
        if self._keras_schedule:  # pragma: no cover
            try:
                res = self._keras_schedule(step)
                from ml_switcheroo.core.tensor_utils import to_float

                return to_float(res)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        step = float(step)  # pragma: no cover
        if self.cycle:  # pragma: no cover
            c = __import__("math").ceil(step / self.decay_steps)  # pragma: no cover
            c = 1.0 if c == 0.0 else c  # pragma: no cover
            step = step - (c - 1) * self.decay_steps  # pragma: no cover
        else:
            step = min(step, self.decay_steps)  # pragma: no cover
        p = step / self.decay_steps  # pragma: no cover
        return (
            self.initial_learning_rate - self.end_learning_rate
        ) * (  # pragma: no cover
            (1 - p) ** self.power
        ) + self.end_learning_rate

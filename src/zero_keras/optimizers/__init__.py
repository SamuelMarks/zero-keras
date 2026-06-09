"""Keras optimizers."""

from typing import Any, Optional
from zero_keras.optimizers import schedules


class Optimizer:
    """Base optimizer class."""

    def __init__(self, name: str = "Optimizer", **kwargs: Any):
        self.name = name


class Adadelta(Optimizer):
    """Optimizer that implements the Adadelta algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.95,
        epsilon: float = 1e-07,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "adadelta",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Adafactor(Optimizer):
    """Optimizer that implements the Adafactor algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_2_decay: float = -0.8,
        epsilon_1: float = 1e-30,
        epsilon_2: float = 0.001,
        clip_threshold: float = 1.0,
        relative_step: bool = True,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        ema_momentum: float = 0.99,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        name: str = "adafactor",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Adagrad(Optimizer):
    """Optimizer that implements the Adagrad algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-07,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "adagrad",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Adam(Optimizer):
    """Optimizer that implements the Adam algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "adam",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class AdamW(Optimizer):
    """Optimizer that implements the AdamW algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "adamw",
        weight_decay: float = 0.004,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Adamax(Optimizer):
    """Optimizer that implements the Adamax algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "adamax",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Ftrl(Optimizer):
    """Optimizer that implements the FTRL algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        learning_rate_power: float = -0.5,
        initial_accumulator_value: float = 0.1,
        l1_regularization_strength: float = 0.0,
        l2_regularization_strength: float = 0.0,
        l2_shrinkage_regularization_strength: float = 0.0,
        beta: float = 0.0,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        ema_momentum: float = 0.99,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        name: str = "ftrl",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Lamb(Optimizer):
    """Optimizer that implements the Lamb algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "lamb",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Lion(Optimizer):
    """Optimizer that implements the Lion algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "lion",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class LossScaleOptimizer(Optimizer):
    """An optimizer that dynamically scales the loss to prevent underflow."""

    def __init__(
        self,
        inner_optimizer: Any,
        initial_scale: Any = 32768.0,
        dynamic_growth_steps: int = 2000,
        learning_rate: Any = None,
        weight_decay: Optional[float] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name or "LossScaleOptimizer", **kwargs)
        self.inner_optimizer = inner_optimizer


class Muon(Optimizer):
    """Optimizer that implements the Muon algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        adam_beta_1: float = 0.9,
        adam_beta_2: float = 0.999,
        epsilon: float = 1e-07,
        exclude_layers: Optional[Any] = None,
        exclude_embeddings: bool = True,
        muon_a: float = 3.4445,
        muon_b: float = -4.775,
        muon_c: float = 2.0315,
        adam_lr_ratio: float = 0.1,
        momentum: float = 0.95,
        ns_steps: int = 6,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        loss_scale_factor: Optional[float] = None,
        ema_overwrite_frequency: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        name: str = "muon",
        use_ema: bool = False,
        global_clipnorm: Optional[float] = None,
        clipnorm: Optional[float] = None,
        ema_momentum: float = 0.99,
        clipvalue: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class Nadam(Optimizer):
    """Optimizer that implements the Nadam algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "nadam",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class RMSprop(Optimizer):
    """Optimizer that implements the RMSprop algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-07,
        centered: bool = False,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "rmsprop",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


class SGD(Optimizer):
    """Gradient descent (with momentum) optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        use_ema: bool = False,
        ema_overwrite_frequency: Optional[int] = None,
        loss_scale_factor: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        ema_momentum: float = 0.99,
        name: str = "SGD",
        weight_decay: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)


__all__ = ["schedules"]

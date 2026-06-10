"""Keras optimizers."""


def _get_keras_optimizer(cls_name, **kwargs):
    import keras
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        try:
            return getattr(keras.optimizers, cls_name)(**kwargs)
        except Exception:
            pass
    return None


class Optimizer:
    """Base class for all Keras optimizers."""

    def __init__(self, **kwargs):
        self._keras_optimizer = None
        self._keras_class = self.__class__.__name__
        self._kwargs = kwargs

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        if self._keras_optimizer is None:
            ko = _get_keras_optimizer(self._keras_class, **self._kwargs)
            if ko:
                self._keras_optimizer = ko
        if self._keras_optimizer:
            try:
                return self._keras_optimizer.apply_gradients(
                    grads_and_vars, *args, **kwargs
                )
            except Exception:
                pass
        return None

    def build(self, var_list):
        if self._keras_optimizer is None:
            ko = _get_keras_optimizer(self._keras_class, **self._kwargs)
            if ko:
                self._keras_optimizer = ko
        if self._keras_optimizer:
            try:
                self._keras_optimizer.build(var_list)
            except Exception:
                pass


class Adadelta(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Adafactor(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class AdamW(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Adamax(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Ftrl(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Lamb(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Lion(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class LossScaleOptimizer(Optimizer):
    def __init__(self, inner_optimizer=None, **kwargs):
        super().__init__(inner_optimizer=inner_optimizer, **kwargs)


class Muon(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class Nadam(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)


from . import schedules as schedules

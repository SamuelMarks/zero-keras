"""Keras optimizers."""

import ml_switcheroo.nn.optimizers as optimizers_impl


class Optimizer:
    """Base class for all Keras optimizers."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._impl = optimizers_impl.Optimizer(**kwargs)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        return self._impl.apply_gradients(grads_and_vars, *args, **kwargs)

    def build(self, var_list):
        return self._impl.build(var_list)


class Adadelta(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        rho=0.95,
        epsilon=1e-07,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="adadelta",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            rho=rho,
            epsilon=epsilon,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Adadelta(
            learning_rate=learning_rate, rho=rho, epsilon=epsilon, **kwargs
        )


class Adafactor(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_2_decay=-0.8,
        epsilon_1=1e-30,
        epsilon_2=0.001,
        clip_threshold=1.0,
        relative_step=True,
        use_ema=False,
        ema_overwrite_frequency=None,
        ema_momentum=0.99,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        name="adafactor",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_2_decay=beta_2_decay,
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            clip_threshold=clip_threshold,
            relative_step=relative_step,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            ema_momentum=ema_momentum,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Adafactor(
            learning_rate=learning_rate,
            beta_2_decay=beta_2_decay,
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            clip_threshold=clip_threshold,
            relative_step=relative_step,
            **kwargs,
        )


class Adagrad(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="adagrad",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon,
            **kwargs,
        )


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="adam",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            **kwargs,
        )


class AdamW(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="adamw",
        weight_decay=0.004,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.AdamW(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
            **kwargs,
        )


class Adamax(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="adamax",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Adamax(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            **kwargs,
        )


class Ftrl(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        learning_rate_power=-0.5,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0,
        l2_shrinkage_regularization_strength=0.0,
        beta=0.0,
        use_ema=False,
        ema_overwrite_frequency=None,
        ema_momentum=0.99,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        name="ftrl",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            learning_rate_power=learning_rate_power,
            initial_accumulator_value=initial_accumulator_value,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
            beta=beta,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            ema_momentum=ema_momentum,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Ftrl(
            learning_rate=learning_rate,
            learning_rate_power=learning_rate_power,
            initial_accumulator_value=initial_accumulator_value,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
            beta=beta,
            **kwargs,
        )


class Lamb(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="lamb",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Lamb(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            **kwargs,
        )


class Lion(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.99,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="lion",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Lion(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, **kwargs
        )


class LossScaleOptimizer(Optimizer):
    def __init__(
        self,
        inner_optimizer,
        initial_scale=32768.0,
        dynamic_growth_steps=2000,
        **kwargs,
    ):
        super().__init__(
            inner_optimizer=inner_optimizer,
            initial_scale=initial_scale,
            dynamic_growth_steps=dynamic_growth_steps,
            **kwargs,
        )
        self._impl = optimizers_impl.LossScaleOptimizer(
            inner_optimizer=inner_optimizer,
            initial_scale=initial_scale,
            dynamic_growth_steps=dynamic_growth_steps,
            **kwargs,
        )


class Muon(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-07,
        exclude_layers=None,
        exclude_embeddings=True,
        muon_a=3.4445,
        muon_b=-4.775,
        muon_c=2.0315,
        adam_lr_ratio=0.1,
        momentum=0.95,
        ns_steps=6,
        nesterov=True,
        weight_decay=0.1,
        loss_scale_factor=None,
        ema_overwrite_frequency=None,
        gradient_accumulation_steps=None,
        name="muon",
        use_ema=False,
        global_clipnorm=None,
        clipnorm=None,
        ema_momentum=0.99,
        clipvalue=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            epsilon=epsilon,
            exclude_layers=exclude_layers,
            exclude_embeddings=exclude_embeddings,
            muon_a=muon_a,
            muon_b=muon_b,
            muon_c=muon_c,
            adam_lr_ratio=adam_lr_ratio,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
            weight_decay=weight_decay,
            loss_scale_factor=loss_scale_factor,
            ema_overwrite_frequency=ema_overwrite_frequency,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            use_ema=use_ema,
            global_clipnorm=global_clipnorm,
            clipnorm=clipnorm,
            ema_momentum=ema_momentum,
            clipvalue=clipvalue,
            **kwargs,
        )
        self._impl = optimizers_impl.Muon(
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            epsilon=epsilon,
            exclude_layers=exclude_layers,
            exclude_embeddings=exclude_embeddings,
            muon_a=muon_a,
            muon_b=muon_b,
            muon_c=muon_c,
            adam_lr_ratio=adam_lr_ratio,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
            weight_decay=weight_decay,
            **kwargs,
        )


class Nadam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="nadam",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.Nadam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            **kwargs,
        )


class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="rmsprop",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.RMSprop(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
            **kwargs,
        )


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="SGD",
        weight_decay=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            global_clipnorm=global_clipnorm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            ema_momentum=ema_momentum,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self._impl = optimizers_impl.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, **kwargs
        )


from . import schedules as schedules

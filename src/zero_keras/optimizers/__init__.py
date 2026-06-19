"""Keras optimizers."""


class Optimizer:
    """Abstract optimizer base class.

    If you intend to create your own optimization algorithm, please inherit from
    this class and override the following methods:

    - `build`: Create your optimizer-related variables, such as momentum
        variables in the SGD optimizer.
    - `update_step`: Implement your optimizer's variable updating logic.
    - `get_config`: serialization of the optimizer.

    Example:
    ```python
    class SGD(Optimizer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.momentum = 0.9

        def build(self, variables):
            super().build(variables)
            self.momentums = []
            for variable in variables:
                self.momentums.append(
                    self.add_variable_from_reference(
                        reference_variable=variable, name="momentum"
                    )
                )

        def update_step(self, gradient, variable, learning_rate):
            learning_rate = ops.cast(learning_rate, variable.dtype)
            gradient = ops.cast(gradient, variable.dtype)
            m = self.momentums[self._get_variable_index(variable)]
            self.assign(
                m,
                ops.subtract(
                    ops.multiply(m, ops.cast(self.momentum, variable.dtype)),
                    ops.multiply(gradient, learning_rate),
                ),
            )
            self.assign_add(variable, m)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                }
            )
            return config
    ```

    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.variables = []
        self.built = False

        class DummyKerasOpt:
            """DummyKerasOpt class."""

            pass

        DummyKerasOpt.__name__ = self.__class__.__name__
        self._keras_optimizer = DummyKerasOpt()

    def add_variable(self, shape, dtype="float32", initializer="zeros", name=None):
        """add_variable function.

        Args:
        shape: Parameter shape.
        dtype: Parameter dtype.
        initializer: Parameter initializer.
        name: Parameter name.

        Returns:
        Any: Return value.

        """
        import ml_switcheroo_compiler.ops as ops
        import ml_switcheroo_compiler.core.dtype as dtypes

        if isinstance(dtype, str):
            dtype = dtypes.DType(dtype)
        if initializer == "zeros":
            var = ops.zeros(shape, dtype=dtype)
        else:
            var = ops.ones(shape, dtype=dtype)
        self.variables.append(var)
        return var

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """apply_gradients function.

        Args:
        grads_and_vars: Parameter grads_and_vars.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        pass

    def build(self, var_list):
        """Build function.

        Args:
        var_list: Parameter var_list.

        Returns:
        Any: Return value.

        """
        self.built = True


class Adadelta(Optimizer):
    """Optimizer that implements the Adadelta algorithm.

    Adadelta optimization is a stochastic gradient descent method that is based
    on adaptive learning rate per dimension to address two drawbacks:

    - The continual decay of learning rates throughout training.
    - The need for a manually selected global learning rate.

    Adadelta is a more robust extension of Adagrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past gradients. This way, Adadelta continues learning even when many updates
    have been done. Compared to Adagrad, in the original version of Adadelta you
    don't have to set an initial learning rate. In this version, the initial
    learning rate can be set, as in most other Keras optimizers.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`. Note that `Adadelta`
            tends to benefit from higher initial learning rate values compared
            to other optimizers. To match the exact form in the original paper,
            use 1.0.
        rho: A floating point value. The decay rate. Defaults to `0.95`.
        epsilon: Small floating point value for maintaining numerical stability.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    Reference:

    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)

    """

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
        pass


class Adafactor(Optimizer):
    """Optimizer that implements the Adafactor algorithm.

    Adafactor is commonly used in NLP tasks, and has the advantage
    of taking less memory because it only saves partial information of previous
    gradients.

    The default argument setup is based on the original paper (see reference).
    When gradients are of dimension > 2, Adafactor optimizer will delete the
    last 2 dimensions separately in its accumulator variables.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_2_decay: float, defaults to -0.8. The decay rate of `beta_2`.
        epsilon_1: float, defaults to 1e-30. A small offset to keep denominator
            away from 0.
        epsilon_2: float, defaults to 1e-3. A small offset to avoid learning
            rate becoming too small by time.
        clip_threshold: float, defaults to 1.0. Clipping threshold. This is a
            part of Adafactor algorithm, independent from `clipnorm`,
            `clipvalue`, and `global_clipnorm`.
        relative_step: bool, defaults to `True`. If `learning_rate` is a
            constant and `relative_step=True`, learning rate will be adjusted
            based on current iterations. This is a default learning rate decay
            in Adafactor.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    Reference:

    - [Shazeer, Noam et al., 2018](https://arxiv.org/abs/1804.04235).

    """

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
        pass


class Adagrad(Optimizer):
    """Optimizer that implements the Adagrad algorithm.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`. Note that `Adagrad`
            tends to benefit from higher initial learning rate values compared
            to other optimizers. To match the exact form in the original paper,
            use `1.0`.
        initial_accumulator_value: Floating point value. Starting value for the
            accumulators (per-parameter momentum values). Must be non-negative.
        epsilon: Small floating point value for maintaining numerical stability.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    Reference:

    - [Duchi et al., 2011](
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

    """

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
        pass


class Adam(Optimizer):
    """Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates. Defaults to
            `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
            to `1e-7`.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond". Defaults
            to `False`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).

    """

    def __init__(
        self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, **kwargs
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.

        """
        if self.built:
            return
        self.m = []
        self.v = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
            self.v.append(self.add_variable(shape=var.shape, name="v"))
        self.iterations = self.add_variable(shape=(), name="iterations")
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """apply_gradients function.

        Args:
        grads_and_vars: Parameter grads_and_vars.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        import ml_switcheroo_compiler.ops as ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        try:
            self.iterations.assign(self.iterations + 1)
        except AttributeError:
            pass

        t = self.iterations + 1
        lr_t = (
            self.learning_rate
            * ops.sqrt(1.0 - ops.power(self.beta_2, t))
            / (1.0 - ops.power(self.beta_1, t))
        )

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue
            m = self.m[i]
            v = self.v[i]

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * ops.square(g)

            var_new = var - lr_t * m_new / (ops.sqrt(v_new) + self.epsilon)

            try:
                m.assign(m_new)
                v.assign(v_new)
                var.assign(var_new)
            except AttributeError:
                pass


class AdamW(Optimizer):
    """Optimizer that implements the AdamW algorithm.

    AdamW optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments with an added
    method to decay weights per the techniques discussed in the paper,
    'Decoupled Weight Decay Regularization' by
    [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the underlying Adam method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates.
            Defaults to `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just
            before Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to 1e-7.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond".
            Defaults to `False`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    References:
    - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980) for `adam`
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    """

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
        pass


class Adamax(Optimizer):
    """Optimizer that implements the Adamax algorithm.

    Adamax, a variant of Adam based on the infinity norm, is a first-order
    gradient-based optimization method. Due to its capability of adjusting the
    learning rate based on data characteristics, it is suited to learn
    time-variant process, e.g., speech data with dynamically changed noise
    conditions. Default parameters follow those provided in the paper (see
    references below).

    Initialization:

    ```python
    m = 0  # Initialize initial 1st moment vector
    u = 0  # Initialize the exponentially weighted infinity norm
    t = 0  # Initialize timestep
    ```

    The update rule for parameter `w` with gradient `g` is described at the end
    of section 7.1 of the paper (see the reference section):

    ```python
    t += 1
    m = beta1 * m + (1 - beta) * g
    u = max(beta2 * u, abs(g))
    current_lr = learning_rate / (1 - beta1 ** t)
    w = w - current_lr * m / (u + epsilon)
    ```

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay
            rate for the exponentially weighted infinity norm.
        epsilon: A small constant for numerical stability.
            name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    Reference:

    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)

    """

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
        pass


class Ftrl(Optimizer):
    """Optimizer that implements the FTRL algorithm.

    "Follow The Regularized Leader" (FTRL) is an optimization algorithm
    developed at Google for click-through rate prediction in the early 2010s. It
    is most suitable for shallow models with large and sparse feature spaces.
    The algorithm is described by
    [McMahan et al., 2013](https://research.google.com/pubs/archive/41159.pdf).
    The Keras version has support for both online L2 regularization
    (the L2 regularization described in the paper
    above) and shrinkage-type L2 regularization
    (which is the addition of an L2 penalty to the loss function).

    Initialization:

    ```python
    n = 0
    sigma = 0
    z = 0
    ```

    Update rule for one variable `w`:

    ```python
    prev_n = n
    n = n + g ** 2
    sigma = (n ** -lr_power - prev_n ** -lr_power) / lr
    z = z + g - sigma * w
    if abs(z) < lambda_1:
      w = 0
    else:
      w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
    ```

    Notation:

    - `lr` is the learning rate
    - `g` is the gradient for the variable
    - `lambda_1` is the L1 regularization strength
    - `lambda_2` is the L2 regularization strength
    - `lr_power` is the power to scale n.

    Check the documentation for the `l2_shrinkage_regularization_strength`
    parameter for more details when shrinkage is enabled, in which case gradient
    is replaced with a gradient with shrinkage.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        learning_rate_power: A float value, must be less or equal to zero.
            Controls how the learning rate decreases during training. Use zero
            for a fixed learning rate.
        initial_accumulator_value: The starting value for accumulators. Only
            zero or positive values are allowed.
        l1_regularization_strength: A float value, must be greater than or equal
            to zero. Defaults to `0.0`.
        l2_regularization_strength: A float value, must be greater than or equal
            to zero. Defaults to `0.0`.
        l2_shrinkage_regularization_strength: A float value, must be greater
            than or equal to zero. This differs from L2 above in that the L2
            above is a stabilization penalty, whereas this L2 shrinkage is a
            magnitude penalty. When input is sparse shrinkage will only happen
            on the active weights.
        beta: A float value, representing the beta value from the paper.
            Defaults to `0.0`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).

    """

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
        pass


class Lamb(Optimizer):
    """Optimizer that implements the Lamb algorithm.

    Lamb is a stochastic gradient descent method that
    uses layer-wise adaptive moments to adjusts the
    learning rate for each parameter based on the ratio of the
    norm of the weight to the norm of the gradient
    This helps to stabilize the training process and improves convergence
    especially for large batch sizes.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates. Defaults to
            `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability.
            Defaults to `1e-7`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    References:
        - [Yang et al.](https://arxiv.org/pdf/1904.00962)

    """

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
        pass


class Lion(Optimizer):
    """Optimizer that implements the Lion algorithm.

    The Lion optimizer is a stochastic-gradient-descent method that uses the
    sign operator to control the magnitude of the update, unlike other adaptive
    optimizers such as Adam that rely on second-order moments. This makes
    Lion more memory-efficient as it only keeps track of the momentum. According
    to the authors (see reference), its performance gain over Adam grows with
    the batch size. Because the update of Lion is produced through the sign
    operation, resulting in a larger norm, a suitable learning rate for Lion is
    typically 3-10x smaller than that for AdamW. The weight decay for Lion
    should in turn be 3-10x larger than that for AdamW to maintain a
    similar strength (lr * wd).

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            rate to combine the current gradient and the 1st moment estimate.
            Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimate. Defaults to
            `0.99`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    References:
    - [Chen et al., 2023](http://arxiv.org/abs/2302.06675)
    - [Authors' implementation](
        http://github.com/google/automl/tree/master/lion)

    """

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
        pass


class LossScaleOptimizer(Optimizer):
    """An optimizer that dynamically scales the loss to prevent underflow.

    Loss scaling is a technique to prevent numeric underflow in intermediate
    gradients when float16 is used. To prevent underflow, the loss is multiplied
    (or "scaled") by a certain factor called the "loss scale", which causes
    intermediate gradients to be scaled by the loss scale as well. The final
    gradients are divided (or "unscaled") by the loss scale to bring them back
    to their original value.

    `LossScaleOptimizer` wraps another optimizer and applies dynamic loss
    scaling to it. This loss scale is dynamically updated over time as follows:
    - On any train step, if a nonfinite gradient is encountered, the loss scale
      is halved, and the train step is skipped.
    - If `dynamic_growth_steps` have occurred since the last time the loss scale
      was updated, and no nonfinite gradients have occurred, the loss scale
      is doubled.

    Args:
        inner_optimizer: The `keras.optimizers.Optimizer` instance to wrap.
        initial_scale: Float. The initial loss scale. This scale will be updated
            during training. It is recommended for this to be a very high
            number, because a loss scale that is too high gets lowered far more
            quickly than a loss scale that is too low gets raised.
        dynamic_growth_steps: Int. How often to update the scale upwards. After
            every `dynamic_growth_steps` steps with finite gradients, the
            loss scale is doubled.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).

    """

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
        pass


class Muon(Optimizer):
    """Optimizer that implements the Muon algorithm.

    Note that this optimizer should not be used in the following layers:

    1. Embedding layer
    2. Final output fully connected layer
    3. Any {0,1}-D variables

    These should all be optimized using AdamW.

    The Muon optimizer can use both the Muon update step or the
    AdamW update step based on the following:

    - For any variable that isn't 2D, 3D or 4D, the AdamW step
        will be used. This is not configurable.
    - If the argument `exclude_embeddings` (defaults to `True`) is set
    to `True`, the AdamW step will be used.
    - For any variablewith a name that matches an expression
        listed in the argument `exclude_layers` (a list), the
        AdamW step will be used.
    - Any other variable uses the Muon step.

    Typically, you only need to pass the name of your densely-connected
    output layer to `exclude_layers`, e.g.
    `exclude_layers=["output_dense"]`.

    References:
        - [Original implementation](https://github.com/KellerJordan/Muon)
        - [Liu et al, 2025](https://arxiv.org/abs/2502.16982)

    Args:
        learning_rate: A float,
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        adam_beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use.
            The exponential decay rate for the 1st moment estimates. Defaults to
            `0.9`.
        adam_beta_2: A float value or a constant float tensor, ora callable
            that takes no arguments and returns the actual value to use.
            The exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This is
            "epsilon hat" in the Kingma and Ba paper
            (in the formula just before Section 2.1),
            not the epsilon in Algorithm 1 of the paper.
            It be used at Adamw.Defaults to `1e-7`.
        exclude_layers: List of strings, keywords of layer names to exclude.
            All layers with keywords in their path will use adamw.
        exclude_embeddings: Boolean value
            If True, embedding layers will use adamw.
        muon_a: Float, parameter a of the muon algorithm.
            It is recommended to use the default value
        muon_b: Float, parameter b of the muon algorithm.
            It is recommended to use the default value
        muon_c: Float, parameter c of the muon algorithm.
            It is recommended to use the default value
        adam_lr_ratio: Float, the ratio of the learning rate when
                using Adam to the main learning rate.
                it is recommended to set it to 0.1
        momentum: Float, momentum used by internal SGD.
        ns_steps: Integer, number of Newton-Schulz iterations to run.
        nesterov: Boolean, whether to use Nesterov-style momentum
        {{base_optimizer_keyword_args}}

    """

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
        pass


class Nadam(Optimizer):
    """Optimizer that implements the Nadam algorithm.

    Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
    Nesterov momentum.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to `1e-7`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    Reference:

    - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).

    """

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
        pass


class RMSprop(Optimizer):
    """Optimizer that implements the RMSprop algorithm.

    The gist of RMSprop is to:

    - Maintain a moving (discounted) average of the square of gradients
    - Divide the gradient by the root of this average

    This implementation of RMSprop uses plain momentum, not Nesterov momentum.

    The centered version additionally maintains a moving average of the
    gradients, and uses that average to estimate the variance.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        rho: float, defaults to 0.9. Discounting factor for the old gradients.
        momentum: float, defaults to 0.0. If not 0.0., the optimizer tracks the
            momentum value, with a decay rate equals to `1 - momentum`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
            to 1e-7.
        centered: Boolean. If `True`, gradients are normalized by the estimated
            variance of the gradient; if False, by the uncentered second moment.
            Setting this to `True` may help with training, but is slightly more
            expensive in terms of computation and memory. Defaults to `False`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).


    Example:
    >>> opt = keras.optimizers.RMSprop(learning_rate=0.1)
    >>> var1 = keras.backend.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0  # d(loss) / d(var1) = var1
    >>> opt.minimize(loss, [var1])
    >>> var1
    9.683772

    Reference:

    - [Hinton, 2012](
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    """

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
        pass


class SGD(Optimizer):
    """Gradient descent (with momentum) optimizer.

    Update rule for parameter `w` with gradient `g` when `momentum` is 0:

    ```python
    w = w - learning_rate * g
    ```

    Update rule when `momentum` is larger than 0:

    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + velocity
    ```

    When `nesterov=True`, this rule becomes:

    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + momentum * velocity - learning_rate * g
    ```

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.01`.
        momentum: float hyperparameter >= 0 that accelerates gradient descent in
            the relevant direction and dampens oscillations. 0 is vanilla
            gradient descent. Defaults to `0.0`.
        nesterov: boolean. Whether to apply Nesterov momentum.
            Defaults to `False`.
        name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).

    """

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def build(self, var_list):
        """Initialize optimizer variables.

        SGD optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGD variables on.

        """
        if self.built:
            return
        self.momentums = []
        if self.momentum != 0:
            for var in var_list:
                self.momentums.append(
                    self.add_variable(shape=var.shape, name="momentum")
                )
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """apply_gradients function.

        Args:
        grads_and_vars: Parameter grads_and_vars.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        import ml_switcheroo_compiler.ops as ops  # noqa: F401

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        for i, (g, v) in enumerate(grads_and_vars):
            if g is None:
                continue
            if self.momentum != 0:
                m = self.momentums[i]
                m_new = self.momentum * m - self.learning_rate * g
                if self.nesterov:
                    v_new = v + self.momentum * m_new - self.learning_rate * g
                else:
                    v_new = v + m_new

                try:
                    m.assign(m_new)
                    v.assign(v_new)
                except AttributeError:
                    # In eager without proper variables, we can't assign in-place easily
                    # but we mock it
                    pass
            else:
                v_new = v - self.learning_rate * g
                try:
                    v.assign(v_new)
                except AttributeError:
                    pass


from . import schedules as schedules

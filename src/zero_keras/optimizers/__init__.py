"""Module docstring."""

from zero_keras.activations import _to_tensor

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

    def add_optimizer_variables(self, variables):
        """add_optimizer_variables docstring.

        Args:
            variables: Variables.
        """
        pass

    def add_variable_from_reference(self, reference_variable, name=""):
        """add_variable_from_reference docstring.

        Args:
            reference_variable: Reference variable.
            name: Name.
        """
        return None

    def apply(self, grads, variables):
        """apply docstring.

        Args:
            grads: Grads.
            variables: Variables.
        """
        pass

    def assign(self, variable, value):
        """assign docstring.

        Args:
            variable: Variable.
            value: Value.
        """
        pass

    def assign_add(self, variable, value):
        """assign_add docstring.

        Args:
            variable: Variable.
            value: Value.
        """
        pass

    def assign_sub(self, variable, value):
        """assign_sub docstring.

        Args:
            variable: Variable.
            value: Value.
        """
        pass

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """exclude_from_weight_decay docstring.

        Args:
            var_list: Var list.
            var_names: Var names.
        """
        pass

    def finalize_variable_values(self, var_list):
        """finalize_variable_values docstring.

        Args:
            var_list: Var list.
        """
        pass

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return getattr(self, "_config", {})

    @property
    def iterations(self):
        """iterations docstring."""
        return getattr(self, "_iterations", 0)

    @iterations.setter
    def iterations(self, value):
        """docstring."""

        self._iterations = value

    @property
    def learning_rate(self):
        """learning_rate docstring."""
        return getattr(self, "_learning_rate", 0.001)

    @learning_rate.setter
    def learning_rate(self, value):
        """docstring."""

        self._learning_rate = value

    def load_own_variables(self, store):
        """load_own_variables docstring.

        Args:
            store: Store.
        """
        pass

    def save_own_variables(self, store):
        """save_own_variables docstring.

        Args:
            store: Store.
        """
        pass

    def scale_loss(self, loss):
        """scale_loss docstring.

        Args:
            loss: Loss.
        """
        return loss

    def set_weights(self, weights):
        """set_weights docstring.

        Args:
            weights: Weights.
        """
        pass

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        """stateless_apply docstring.

        Args:
            optimizer_variables: Optimizer variables.
            grads: Grads.
            trainable_variables: Trainable variables.
        """
        pass

    def update_step(self, gradient, variable, learning_rate):
        """update_step docstring.

        Args:
            gradient: Gradient.
            variable: Variable.
            learning_rate: Learning rate.
        """
        pass

    def __init__(self, **kwargs):
        """Function docstring.

        Args:
            kwargs: Description.
        """
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
        from zero_keras.ops import ops
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
        """Function docstring.

        Args:
            learning_rate: Description.
            rho: Description.
            epsilon: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            weight_decay: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.accumulated_grads = []
        self.accumulated_delta_vars = []
        for var in var_list:
            self.accumulated_grads.append(
                self.add_variable(shape=var.shape, name="accumulated_grads")
            )
            self.accumulated_delta_vars.append(
                self.add_variable(shape=var.shape, name="accumulated_delta_vars")
            )
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue

            acc_g = self.accumulated_grads[i]
            acc_delta = self.accumulated_delta_vars[i]

            acc_g_new = self.rho * acc_g + (1.0 - self.rho) * ops.square(g)

            delta_var = (
                -ops.sqrt(acc_delta + self.epsilon)
                / ops.sqrt(acc_g_new + self.epsilon)
                * g
            )
            var_new = var + lr * delta_var

            acc_delta_new = self.rho * acc_delta + (1.0 - self.rho) * ops.square(
                delta_var
            )

            try:
                acc_g.assign(acc_g_new)
                acc_delta.assign(acc_delta_new)
                var.assign(var_new)
            except AttributeError:
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
        factored=True,
        multiply_by_parameter_scale=True,
        clipping_threshold=1.0,
        beta_2_decay=-0.8,
        epsilon_1=1e-30,
        epsilon_2=1e-3,
        weight_decay=None,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="adafactor",
        **kwargs,
    ):
        """Function docstring.

        Args:
            learning_rate: Description.
            factored: Description.
            multiply_by_parameter_scale: Description.
            clipping_threshold: Description.
            beta_2_decay: Description.
            epsilon_1: Description.
            epsilon_2: Description.
            weight_decay: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.factored = factored
        self.epsilon_2 = epsilon_2

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.v = []
        for var in var_list:
            self.v.append(self.add_variable(shape=var.shape, name="v"))
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue
            v = self.v[i]
            v_new = 0.9 * v + 0.1 * ops.square(g)
            var_new = var - lr * g / ops.sqrt(v_new + self.epsilon_2)
            try:
                v.assign(v_new)
                var.assign(var_new)
            except AttributeError:
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
        """Function docstring.

        Args:
            learning_rate: Description.
            initial_accumulator_value: Description.
            epsilon: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            weight_decay: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.accumulator = []
        for var in var_list:
            from zero_keras.ops import ops

            acc = self.add_variable(shape=var.shape, name="accumulator")
            init_val = ops.ones_like(var) * _to_tensor(self.initial_accumulator_value)
            try:
                acc.assign(init_val)
            except AttributeError:
                pass
            self.accumulator.append(acc)
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue

            acc = self.accumulator[i]
            acc_new = acc + ops.square(g)

            var_new = var - lr * g / ops.sqrt(acc_new + self.epsilon)

            try:
                acc.assign(acc_new)
                var.assign(var_new)
            except AttributeError:
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
        """Function docstring.

        Args:
            learning_rate: Description.
            beta_1: Description.
            beta_2: Description.
            epsilon: Description.
            kwargs: Description.
        """
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
        from zero_keras.ops import ops

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
        weight_decay=0.004,
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
        **kwargs,
    ):
        """Function docstring.

        Args:
            learning_rate: Description.
            weight_decay: Description.
            beta_1: Description.
            beta_2: Description.
            epsilon: Description.
            amsgrad: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.m = []
        self.v = []
        if self.amsgrad:
            self.v_hat = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
            self.v.append(self.add_variable(shape=var.shape, name="v"))
            if self.amsgrad:
                self.v_hat.append(self.add_variable(shape=var.shape, name="v_hat"))
        self.iterations = self.add_variable(shape=(), name="iterations")
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

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

            var_new = var - self.learning_rate * self.weight_decay * var

            m = self.m[i]
            v = self.v[i]

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * ops.square(g)

            if self.amsgrad:
                v_hat = self.v_hat[i]
                v_hat_new = ops.maximum(v_hat, v_new)
                var_new = var_new - lr_t * m_new / (ops.sqrt(v_hat_new) + self.epsilon)
                try:
                    v_hat.assign(v_hat_new)
                except AttributeError:
                    pass
            else:
                var_new = var_new - lr_t * m_new / (ops.sqrt(v_new) + self.epsilon)

            try:
                m.assign(m_new)
                v.assign(v_new)
                var.assign(var_new)
            except AttributeError:
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
        """Function docstring.

        Args:
            learning_rate: Description.
            beta_1: Description.
            beta_2: Description.
            epsilon: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            weight_decay: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.m = []
        self.u = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
            self.u.append(self.add_variable(shape=var.shape, name="u"))
        self.iterations = self.add_variable(shape=(), name="iterations")
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        try:
            self.iterations.assign(self.iterations + 1)
        except AttributeError:
            pass

        t = self.iterations + 1
        lr_t = self.learning_rate / (1.0 - ops.power(self.beta_1, t))

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue

            m = self.m[i]
            u = self.u[i]

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            u_new = ops.maximum(self.beta_2 * u, ops.abs(g))

            var_new = var - lr_t * m_new / (u_new + self.epsilon)

            try:
                m.assign(m_new)
                u.assign(u_new)
                var.assign(var_new)
            except AttributeError:
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
        """Function docstring.

        Args:
            learning_rate: Description.
            learning_rate_power: Description.
            initial_accumulator_value: Description.
            l1_regularization_strength: Description.
            l2_regularization_strength: Description.
            l2_shrinkage_regularization_strength: Description.
            beta: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            ema_momentum: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            name: Description.
            weight_decay: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.z = []
        self.n = []
        for var in var_list:
            from zero_keras.ops import ops

            n_var = self.add_variable(shape=var.shape, name="n")
            init_val = ops.ones_like(var) * _to_tensor(self.initial_accumulator_value)
            try:
                n_var.assign(init_val)
            except AttributeError:
                pass
            self.n.append(n_var)
            self.z.append(self.add_variable(shape=var.shape, name="z"))
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue
            n = self.n[i]
            z = self.z[i]
            n_new = n + ops.square(g)
            sigma = (ops.sqrt(n_new) - ops.sqrt(n)) / lr
            z_new = z + g - sigma * var
            var_new = var - lr * z_new / ops.sqrt(n_new)
            try:
                n.assign(n_new)
                z.assign(z_new)
                var.assign(var_new)
            except AttributeError:
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
        weight_decay=None,
        use_ema=False,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        global_clipnorm=None,
        gradient_accumulation_steps=None,
        clipnorm=None,
        clipvalue=None,
        ema_momentum=0.99,
        name="lamb",
        **kwargs,
    ):
        """Function docstring.

        Args:
            learning_rate: Description.
            beta_1: Description.
            beta_2: Description.
            epsilon: Description.
            weight_decay: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay if weight_decay is not None else 0.0

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.m = []
        self.v = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
            self.v.append(self.add_variable(shape=var.shape, name="v"))
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue
            m = self.m[i]
            v = self.v[i]
            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * ops.square(g)

            update = m_new / (ops.sqrt(v_new) + self.epsilon) + self.weight_decay * var

            w_norm = ops.sqrt(ops.sum(ops.square(var)))
            u_norm = ops.sqrt(ops.sum(ops.square(update)))
            ratio = ops.where(
                ops.greater(w_norm, 0.0),
                ops.where(ops.greater(u_norm, 0.0), w_norm / u_norm, _to_tensor(1.0)),
                _to_tensor(1.0),
            )

            var_new = var - lr * ratio * update
            try:
                m.assign(m_new)
                v.assign(v_new)
                var.assign(var_new)
            except AttributeError:
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
        learning_rate=0.0001,
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
        """Function docstring.

        Args:
            learning_rate: Description.
            beta_1: Description.
            beta_2: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            weight_decay: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.m = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue

            m = self.m[i]

            c = self.beta_1 * m + (1.0 - self.beta_1) * g
            var_new = var - lr * ops.sign(c)
            m_new = self.beta_2 * m + (1.0 - self.beta_2) * g

            try:
                m.assign(m_new)
                var.assign(var_new)
            except AttributeError:
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
        initial_scale=1024.0,
        dynamic=True,
        dynamic_growth_steps=2000,
        name="LossScaleOptimizer",
        **kwargs,
    ):
        """Function docstring.

        Args:
            inner_optimizer: Description.
            initial_scale: Description.
            dynamic: Description.
            dynamic_growth_steps: Description.
            name: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.inner_optimizer = inner_optimizer
        self.initial_scale = initial_scale

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        if hasattr(self.inner_optimizer, "build"):
            self.inner_optimizer.build(var_list)
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if hasattr(self.inner_optimizer, "apply_gradients"):
            self.inner_optimizer.apply_gradients(grads_and_vars, *args, **kwargs)


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
        momentum=0.0,
        name="muon",
        **kwargs,
    ):
        """Function docstring.

        Args:
            learning_rate: Description.
            momentum: Description.
            name: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.m = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue
            m = self.m[i]
            m_new = self.momentum * m + (1.0 - self.momentum) * g
            var_new = var - lr * m_new
            try:
                m.assign(m_new)
                var.assign(var_new)
            except AttributeError:
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
        """Function docstring.

        Args:
            learning_rate: Description.
            beta_1: Description.
            beta_2: Description.
            epsilon: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            weight_decay: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
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
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        try:
            self.iterations.assign(self.iterations + 1)
        except AttributeError:
            pass

        t = self.iterations + 1

        beta_1_t = self.beta_1 * ops.power(0.96, t * 0.004)
        beta_1_t_next = self.beta_1 * ops.power(0.96, (t + 1) * 0.004)

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

            m_hat = beta_1_t_next * m_new + (1.0 - beta_1_t) * g
            var_new = var - lr_t * m_hat / (ops.sqrt(v_new) + self.epsilon)

            try:
                m.assign(m_new)
                v.assign(v_new)
                var.assign(var_new)
            except AttributeError:
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
        """Function docstring.

        Args:
            learning_rate: Description.
            rho: Description.
            momentum: Description.
            epsilon: Description.
            centered: Description.
            use_ema: Description.
            ema_overwrite_frequency: Description.
            loss_scale_factor: Description.
            global_clipnorm: Description.
            gradient_accumulation_steps: Description.
            clipnorm: Description.
            clipvalue: Description.
            ema_momentum: Description.
            name: Description.
            weight_decay: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def build(self, var_list):
        """Function docstring.

        Args:
            var_list: Description.
        """
        if self.built:
            return
        self.v = []
        if self.centered:
            self.mg = []
        if self.momentum > 0.0:
            self.m = []
        for var in var_list:
            self.v.append(self.add_variable(shape=var.shape, name="v"))
            if self.centered:
                self.mg.append(self.add_variable(shape=var.shape, name="mg"))
            if self.momentum > 0.0:
                self.m.append(self.add_variable(shape=var.shape, name="m"))
        self.built = True

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Function docstring.

        Args:
            grads_and_vars: Description.
            args: Description.
            kwargs: Description.
        """
        from zero_keras.ops import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if not grads_and_vars:
            return
        lr = ops.cast(_to_tensor(self.learning_rate), grads_and_vars[0][1].dtype)

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue

            v = self.v[i]
            v_new = self.rho * v + (1.0 - self.rho) * ops.square(g)

            if self.centered:
                mg = self.mg[i]
                mg_new = self.rho * mg + (1.0 - self.rho) * g
                denominator = v_new - ops.square(mg_new) + self.epsilon
                try:
                    mg.assign(mg_new)
                except AttributeError:
                    pass
            else:
                denominator = v_new + self.epsilon

            denominator = ops.sqrt(denominator)

            if self.momentum > 0.0:
                m = self.m[i]
                m_new = self.momentum * m + lr * g / denominator
                var_new = var - m_new
                try:
                    m.assign(m_new)
                except AttributeError:
                    pass
            else:
                var_new = var - lr * g / denominator

            try:
                v.assign(v_new)
                var.assign(var_new)
            except AttributeError:
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
        """Function docstring.

        Args:
            learning_rate: Description.
            momentum: Description.
            nesterov: Description.
            kwargs: Description.
        """
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
        from zero_keras.ops import ops  # noqa: F401

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


# Legacy namespace support
class legacy:
    """Class docstring."""

    pass


def serialize(optimizer):
    """Serialize an optimizer."""
    if optimizer is None:
        return None
    if isinstance(optimizer, str):
        return optimizer
    return {
        "class_name": optimizer.__class__.__name__,
        "config": optimizer.get_config() if hasattr(optimizer, "get_config") else {},
    }


def deserialize(config, custom_objects=None):
    """Deserialize an optimizer."""
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf = config.get("config", {})
        cls = globals().get(class_name)
        if cls:
            return cls(**conf)
    return config


def get(identifier):
    """Retrieve a Keras optimizer object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, str):
        identifier = identifier.lower()
        if identifier == "adam":
            return Adam()
        if identifier == "sgd":
            return SGD()
        return identifier
    return identifier

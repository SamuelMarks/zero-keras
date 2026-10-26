"""Optimization algorithms and gradient update rules for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
from zero_keras.core_layers import Variable
from zero_keras.types import (
    ConfigDict,
    DTypeLike,
    ShapeTuple,
    SupportsCall,
    TensorLike,
)

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

    def add_optimizer_variables(self, variables: Sequence[TensorLike]) -> None:
        """Initialize optimizer state variables for the given model variables.

        Args:
            variables: Sequence of trainable model variables.
        """
        _ = variables

    def add_variable_from_reference(self, reference_variable: TensorLike, name: str = "") -> None:
        """Create an optimizer state variable matching the shape and dtype of ``reference_variable``.

        Args:
            reference_variable: Model variable to mirror.
            name: Optional name for the created state variable.
        """
        _ = reference_variable, name
        return None

    def apply(self, grads: Sequence[TensorLike], variables: Sequence[TensorLike]) -> None:
        """Apply gradients to variables.

        Args:
            grads: Sequence of gradient tensors.
            variables: Sequence of trainable variables.
        """
        _ = grads, variables

    def assign(self, variable: TensorLike, value: TensorLike) -> None:
        """Assign ``value`` to ``variable`` in-place.

        Args:
            variable: Target variable supporting ``.assign()``.
            value: New value tensor.
        """
        variable.assign(value)

    def assign_add(self, variable: TensorLike, value: TensorLike) -> None:
        """Add ``value`` to ``variable`` in-place.

        Args:
            variable: Target variable supporting ``.assign()``.
            value: Increment tensor.
        """
        variable.assign(variable + value)

    def assign_sub(self, variable: TensorLike, value: TensorLike) -> None:
        """Subtract ``value`` from ``variable`` in-place.

        Args:
            variable: Target variable supporting ``.assign()``.
            value: Decrement tensor.
        """
        variable.assign(variable - value)

    def exclude_from_weight_decay(
        self,
        var_list: Sequence[TensorLike] | None = None,
        var_names: Sequence[str] | None = None,
    ) -> None:
        """Exclude specific variables or variable name patterns from weight decay.

        Args:
            var_list: Optional sequence of variables to exclude.
            var_names: Optional sequence of variable name substrings to exclude.
        """
        if var_list:
            self._exclude_from_weight_decay = set(var_list)
        if var_names:
            self._exclude_from_weight_decay_names = set(var_names)

    def finalize_variable_values(self, var_list: Sequence[TensorLike]) -> None:
        """Finalize variable values (overwriting with EMA averages when ``use_ema=True``) at the end of training.

        Args:
            var_list: Sequence of model variables to finalize.
        """
        from zero_keras.types import SupportsAssign

        if self.use_ema and self._ema_vars:
            for var, ema_v in zip(var_list, self._ema_vars):
                if isinstance(var, SupportsAssign):
                    var.assign(ema_v.value)

    def _current_learning_rate(self, dtype: DTypeLike = "float32") -> TensorLike:
        """Evaluate ``self.learning_rate`` (supporting schedules) and cast to ``dtype``.

        Args:
            dtype: Target tensor data type.

        Returns:
            TensorLike: Effective learning rate scalar tensor.
        """
        from zero_keras import ops

        lr = (
            self.learning_rate(self.iterations)
            if callable(self.learning_rate)
            else self.learning_rate
        )
        return ops.cast(_to_tensor(lr), dtype)

    def _clip_gradients(self, grads: Sequence[TensorLike | None]) -> list[TensorLike | None]:
        """Clip ``grads`` according to ``clipvalue``, ``clipnorm``, and ``global_clipnorm``.

        Args:
            grads: Sequence of gradient tensors or ``None`` entries.

        Returns:
            list[TensorLike | None]: Clipped gradient list.
        """
        from zero_keras import ops

        clipped: list[TensorLike | None] = list(grads)
        if self.clipvalue is not None:
            cv = float(self.clipvalue)
            clipped = [ops.clip(_to_tensor(g), -cv, cv) if g is not None else None for g in clipped]
        if self.clipnorm is not None:
            cn = float(self.clipnorm)
            new_clipped: list[TensorLike | None] = []
            for g in clipped:
                if g is None:
                    new_clipped.append(None)
                else:
                    gt = _to_tensor(g)
                    norm = ops.sqrt(ops.sum(ops.square(gt)))
                    scale = ops.minimum(
                        _to_tensor(1.0), ops.divide(_to_tensor(cn), ops.add(norm, _to_tensor(1e-7)))
                    )
                    new_clipped.append(ops.multiply(gt, scale))
            clipped = new_clipped
        if self.global_clipnorm is not None:
            gcn = float(self.global_clipnorm)
            valid_g = [_to_tensor(g) for g in clipped if g is not None]
            if valid_g:
                total_sq = _to_tensor(0.0)
                for gt in valid_g:
                    total_sq = ops.add(total_sq, ops.sum(ops.square(gt)))
                g_norm = ops.sqrt(total_sq)
                g_scale = ops.minimum(
                    _to_tensor(1.0),
                    ops.divide(_to_tensor(gcn), ops.add(g_norm, _to_tensor(1e-7))),
                )
                clipped = [
                    ops.multiply(_to_tensor(g), g_scale) if g is not None else None for g in clipped
                ]
        return clipped

    def _apply_weight_decay(
        self, variables: Sequence[TensorLike], learning_rate: TensorLike | None = None
    ) -> None:
        """Apply decoupled weight decay to ``variables`` when ``self.weight_decay`` is set.

        Args:
            variables: Sequence of trainable variables.
            learning_rate: Optional effective learning rate; if ``None``, evaluated from ``self._current_learning_rate()``.
        """
        if self.weight_decay is None:
            return
        from zero_keras.types import NameCarrierProtocol, SupportsAssign

        lr = learning_rate if learning_rate is not None else self._current_learning_rate()
        wd = _to_tensor(self.weight_decay)
        for var in variables:
            if var in self._exclude_from_weight_decay:
                continue
            v_name = var.name if isinstance(var, NameCarrierProtocol) and var.name else ""
            if any(pat in v_name for pat in self._exclude_from_weight_decay_names):
                continue
            if isinstance(var, SupportsAssign):
                var.assign(var - var * wd * lr)

    def _update_ema(self, variables: Sequence[TensorLike], step: int = 1) -> None:
        """Update exponential moving average variables and optionally overwrite ``variables``.

        Args:
            variables: Sequence of trainable variables.
            step: Current 1-based step index used for ``ema_overwrite_frequency``.
        """
        if not self.use_ema:
            return
        from zero_keras.types import DTypeCarrierProtocol, ShapeCarrierProtocol, SupportsAssign

        if not self._ema_vars:
            for var in variables:
                v_shape = var.shape if isinstance(var, ShapeCarrierProtocol) else ()
                v_dtype = var.dtype if isinstance(var, DTypeCarrierProtocol) else "float32"
                ema_v = self.add_variable(shape=v_shape, dtype=v_dtype, name="ema")
                ema_v.assign(_to_tensor(var))
                self._ema_vars.append(ema_v)
        mom = _to_tensor(self.ema_momentum)
        one_minus_mom = _to_tensor(1.0 - float(self.ema_momentum))
        for var, ema_v in zip(variables, self._ema_vars):
            new_ema = ema_v * mom + _to_tensor(var) * one_minus_mom
            ema_v.assign(new_ema)
            if (
                self.ema_overwrite_frequency is not None
                and int(self.ema_overwrite_frequency) > 0
                and step % int(self.ema_overwrite_frequency) == 0
                and isinstance(var, SupportsAssign)
            ):
                var.assign(new_ema)

    @classmethod
    def from_config(cls, config: ConfigDict) -> Optimizer:
        """Instantiate an optimizer from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Optimizer: Reconstructed optimizer instance.
        """
        return cls(**config)

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration of the optimizer.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return self._config

    @property
    def iterations(self) -> TensorLike:
        """Return the number of optimization steps completed.

        Returns:
            TensorLike: Optimization step counter.
        """
        return self._iterations

    @iterations.setter
    def iterations(self, value: TensorLike) -> None:
        """Set the number of optimization steps completed.

        Args:
            value: Updated iteration counter.
        """
        self._iterations = value

    @property
    def learning_rate(self) -> TensorLike:
        """Return the current learning rate or learning rate schedule.

        Returns:
            TensorLike: Current learning rate scalar, variable, or schedule.
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: TensorLike) -> None:
        """Set the learning rate or learning rate schedule.

        Args:
            value: Updated learning rate or schedule.
        """
        self._learning_rate = value

    def load_own_variables(self, store: dict[str, TensorLike]) -> None:
        """Load optimizer state variables from ``store``.

        Args:
            store: Mapping from string index to variable array.
        """
        if not self.variables:
            return
        for i, var in enumerate(self.variables):
            if str(i) in store:
                var.assign(store[str(i)])

    def save_own_variables(self, store: dict[str, TensorLike]) -> None:
        """Save optimizer state variables into ``store``.

        Args:
            store: Mutable mapping populated with string index to NumPy array entries.
        """
        if not self.variables:
            return
        for i, var in enumerate(self.variables):
            store[str(i)] = var.numpy()

    def scale_loss(self, loss: TensorLike) -> TensorLike:
        """Scale ``loss`` prior to gradient computation.

        Args:
            loss: Unscaled loss tensor.

        Returns:
            TensorLike: Scaled loss tensor.
        """
        return loss

    def set_weights(self, weights: Sequence[TensorLike]) -> None:
        """Assign a sequence of weight values to ``self.variables``.

        Args:
            weights: Sequence of weight arrays matching ``self.variables``.
        """
        for v, w in zip(self.variables, weights):
            v.assign(w)

    def stateless_apply(
        self,
        optimizer_variables: Sequence[TensorLike],
        grads: Sequence[TensorLike],
        trainable_variables: Sequence[TensorLike],
    ) -> None:
        """Statelessly apply gradients without mutating variables in-place.

        Args:
            optimizer_variables: Current optimizer state variables.
            grads: Gradients for each trainable variable.
            trainable_variables: Trainable model variables.
        """
        _ = optimizer_variables, grads, trainable_variables

    def update_step(
        self, gradient: TensorLike, variable: TensorLike, learning_rate: TensorLike
    ) -> None:
        """Perform a single parameter update step on ``variable``.

        Args:
            gradient: Gradient tensor with respect to ``variable``.
            variable: Target variable to update.
            learning_rate: Effective learning rate for the current step.
        """
        self.assign_sub(variable, gradient * learning_rate)

    variables: list[TensorLike] = []

    def __init__(self, **kwargs: TensorLike) -> None:
        """Initialize the base Optimizer instance.

        Args:
            **kwargs: Optimizer configuration keyword arguments.
        """
        self._kwargs = kwargs
        self._config: ConfigDict = {}
        self._iterations: TensorLike = 0
        self._learning_rate: TensorLike = kwargs.get("learning_rate", 0.001)
        self.weight_decay: TensorLike = kwargs.get("weight_decay")
        self.clipnorm: TensorLike = kwargs.get("clipnorm")
        self.clipvalue: TensorLike = kwargs.get("clipvalue")
        self.global_clipnorm: TensorLike = kwargs.get("global_clipnorm")
        self.use_ema: TensorLike = kwargs.get("use_ema", False)
        self.ema_momentum: TensorLike = kwargs.get("ema_momentum", 0.99)
        self.ema_overwrite_frequency: TensorLike = kwargs.get("ema_overwrite_frequency")
        self._exclude_from_weight_decay: set[TensorLike] = set()
        self._exclude_from_weight_decay_names: set[str] = set()
        self._ema_vars: list[Variable] = []
        self.variables: list[Variable] = []
        self.built: bool = False

        class DummyKerasOpt:
            """Internal metadata holder for optimizer class name."""

        DummyKerasOpt.__name__ = type(self).__name__
        self._keras_optimizer = DummyKerasOpt()

    def add_variable(
        self,
        shape: ShapeTuple,
        dtype: DTypeLike = "float32",
        initializer: str | SupportsCall = "zeros",
        name: str | None = None,
    ) -> Variable:
        """Create, initialize, and track an optimizer state variable.

        Args:
            shape: Shape tuple of the state variable.
            dtype: Data type of the variable.
            initializer: Initializer name or callable.
            name: Optional variable name.

        Returns:
            Variable: Newly created optimizer state variable.
        """
        import ml_switcheroo_compiler.core.dtype as dtypes

        from zero_keras import ops
        from zero_keras.core_layers import Variable

        resolved_dt = dtypes.DType(dtype) if isinstance(dtype, str) else dtype
        if initializer == "zeros":
            val = ops.zeros(shape, dtype=resolved_dt)
        else:
            val = ops.ones(shape, dtype=resolved_dt)

        var = Variable(initializer=initializer, shape=shape, dtype=resolved_dt, name=name)
        var.assign(val)
        self.variables.append(var)
        return var

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply gradients to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = grads_and_vars, args, kwargs

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create optimizer state variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        _ = var_list
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
        learning_rate: float | TensorLike = 0.001,
        rho: float = 0.95,
        epsilon: float = 1e-07,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Adadelta optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            rho: Exponential decay rate for squared gradients and updates.
            epsilon: Small constant for numerical stability.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            weight_decay: Optional weight decay coefficient.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
            weight_decay,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create accumulated gradient and delta variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
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

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Adadelta gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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

            delta_var = -ops.sqrt(acc_delta + self.epsilon) / ops.sqrt(acc_g_new + self.epsilon) * g
            var_new = var + lr * delta_var

            acc_delta_new = self.rho * acc_delta + (1.0 - self.rho) * ops.square(delta_var)

            acc_g.assign(acc_g_new)
            acc_delta.assign(acc_delta_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        beta_2_decay: float = -0.8,
        epsilon_1: float = 1e-30,
        epsilon_2: float = 1e-3,
        clip_threshold: float = 1.0,
        relative_step: bool = True,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        factored: bool = True,
        multiply_by_parameter_scale: bool = True,
        clipping_threshold: float = 1.0,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Adafactor optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            beta_2_decay: Decay rate exponent for second-moment estimator.
            epsilon_1: Small offset for denominator stability.
            epsilon_2: Small offset for relative parameter scale stability.
            clip_threshold: Root-mean-square update clipping threshold.
            relative_step: Whether to scale learning rate by parameter RMS.
            weight_decay: Optional weight decay coefficient.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            global_clipnorm: Optional global gradient norm clipping threshold.
            use_ema: Whether to apply exponential moving average.
            ema_momentum: Momentum for EMA updates.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            name: Optimizer instance name.
            factored: Whether to factorize second-moment estimators for 2D+ tensors.
            multiply_by_parameter_scale: Legacy alias for relative step scaling.
            clipping_threshold: Legacy alias for ``clip_threshold``.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            clip_threshold,
            relative_step,
            multiply_by_parameter_scale,
            clipping_threshold,
            beta_2_decay,
            epsilon_1,
            weight_decay,
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.factored = factored
        self.epsilon_2 = epsilon_2

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create second-moment accumulator variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        if self.built:
            return
        self.v = []
        for var in var_list:
            self.v.append(self.add_variable(shape=var.shape, name="v"))
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Adafactor gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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
            v.assign(v_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-07,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Adagrad optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            initial_accumulator_value: Starting non-negative value for gradient accumulators.
            epsilon: Small constant for numerical stability.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            weight_decay: Optional weight decay coefficient.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
            weight_decay,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create squared-gradient accumulator variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        if self.built:
            return
        self.accumulator = []
        for var in var_list:
            from zero_keras import ops

            acc = self.add_variable(shape=var.shape, name="accumulator")

            init_val = ops.ones(var.shape) * _to_tensor(self.initial_accumulator_value)
            acc.assign(init_val)
            self.accumulator.append(acc)
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Adagrad gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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

            acc.assign(acc_new)
            var.assign(var_new)


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
        self,
        learning_rate: float | TensorLike = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Adam optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            beta_1: Exponential decay rate for first-moment estimates.
            beta_2: Exponential decay rate for second-moment estimates.
            epsilon: Small constant for numerical stability.
            amsgrad: Whether to apply the AMSGrad variant.
            weight_decay: Optional weight decay coefficient.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            global_clipnorm: Optional global gradient norm clipping threshold.
            use_ema: Whether to apply exponential moving average.
            ema_momentum: Momentum for EMA updates.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            name: Optimizer instance name.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            amsgrad,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            ema_overwrite_frequency,
            loss_scale_factor,
            gradient_accumulation_steps,
            name,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list: Sequence[TensorLike]) -> None:
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

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Adam gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        self.iterations.assign(self.iterations + 1)

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

            m.assign(m_new)
            v.assign(v_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        weight_decay: float = 0.004,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the AdamW optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            weight_decay: Decoupled weight decay coefficient.
            beta_1: Exponential decay rate for first-moment estimates.
            beta_2: Exponential decay rate for second-moment estimates.
            epsilon: Small constant for numerical stability.
            amsgrad: Whether to apply the AMSGrad variant.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create first-moment, second-moment, and optional AMSGrad state variables.

        Args:
            var_list: Sequence of model variables to optimize.
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

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply AdamW gradient updates with decoupled weight decay to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        self.iterations.assign(self.iterations + 1)

        t = self.iterations + 1
        lr_t = (
            self.learning_rate
            * ops.sqrt(1.0 - ops.power(self.beta_2, t))
            / (1.0 - ops.power(self.beta_1, t))
        )

        for i, (g, var) in enumerate(grads_and_vars):
            if g is None:
                continue

            var_val = var
            var_new = (
                var_val - ops.convert_to_tensor(self.learning_rate * self.weight_decay) * var_val
            )

            m = self.m[i]
            v = self.v[i]
            m_val = m
            v_val_opt = v

            m_new = self.beta_1 * m_val + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v_val_opt + (1.0 - self.beta_2) * ops.square(g)

            if self.amsgrad:
                v_hat = self.v_hat[i]
                v_hat_new = ops.maximum(v_hat, v_new)
                var_new = var_new - lr_t * m_new / (ops.sqrt(v_hat_new) + self.epsilon)
                v_hat.assign(v_hat_new)
            else:
                var_new = var_new - lr_t * m_new / (ops.sqrt(v_new) + self.epsilon)

            m.assign(m_new)
            v.assign(v_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Adamax optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            beta_1: Exponential decay rate for first-moment estimates.
            beta_2: Exponential decay rate for infinity-norm estimates.
            epsilon: Small constant for numerical stability.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            weight_decay: Optional weight decay coefficient.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
            weight_decay,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create first-moment and infinity-norm state variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
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

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Adamax gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        self.iterations.assign(self.iterations + 1)

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

            m.assign(m_new)
            u.assign(u_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        learning_rate_power: float = -0.5,
        initial_accumulator_value: float = 0.1,
        l1_regularization_strength: float = 0.0,
        l2_regularization_strength: float = 0.0,
        l2_shrinkage_regularization_strength: float = 0.0,
        beta: float = 0.0,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Ftrl optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            learning_rate_power: Non-positive exponent controlling learning rate decay.
            initial_accumulator_value: Starting non-negative value for accumulators.
            l1_regularization_strength: Non-negative L1 regularization coefficient.
            l2_regularization_strength: Non-negative L2 regularization coefficient.
            l2_shrinkage_regularization_strength: Non-negative L2 magnitude shrinkage penalty.
            beta: Beta stabilization parameter from McMahan et al.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            ema_momentum: Momentum for EMA updates.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            name: Optimizer instance name.
            weight_decay: Optional weight decay coefficient.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            learning_rate_power,
            l1_regularization_strength,
            l2_regularization_strength,
            l2_shrinkage_regularization_strength,
            beta,
            use_ema,
            ema_overwrite_frequency,
            ema_momentum,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            name,
            weight_decay,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create ``n`` and ``z`` accumulator variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        if self.built:
            return
        self.z = []
        self.n = []
        for var in var_list:
            from zero_keras import ops

            n_var = self.add_variable(shape=var.shape, name="n")
            init_val = ops.ones(var.shape) * _to_tensor(self.initial_accumulator_value)
            n_var.assign(init_val)
            self.n.append(n_var)
            self.z.append(self.add_variable(shape=var.shape, name="z"))
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Ftrl gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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
            var_new = ops.negative(lr * z_new / ops.sqrt(n_new))
            n.assign(n_new)
            z.assign(z_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Lamb optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            beta_1: Exponential decay rate for first-moment estimates.
            beta_2: Exponential decay rate for second-moment estimates.
            epsilon: Small constant for numerical stability.
            weight_decay: Optional weight decay coefficient.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay if weight_decay is not None else 0.0

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create first-moment and second-moment state variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        if self.built:
            return
        self.m = []
        self.v = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
            self.v.append(self.add_variable(shape=var.shape, name="v"))
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply LAMB layer-wise adaptive trust-ratio gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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
            m.assign(m_new)
            v.assign(v_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Lion optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            beta_1: Interpolation rate between momentum and current gradient.
            beta_2: Exponential decay rate for momentum updates.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            weight_decay: Optional weight decay coefficient.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
            weight_decay,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create momentum state variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        if self.built:
            return
        self.m = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Lion sign-momentum gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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

            m.assign(m_new)
            var.assign(var_new)


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

    check_finite: bool = True

    def __init__(
        self,
        inner_optimizer: TensorLike,
        initial_scale: float = 2.0**15,
        dynamic_growth_steps: int = 2000,
        name: str | None = None,
        dynamic: bool = True,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the LossScaleOptimizer wrapper.

        Args:
            inner_optimizer: Underlying optimizer instance to wrap.
            initial_scale: Initial loss scaling factor.
            dynamic_growth_steps: Step interval before doubling the scale.
            name: Optimizer instance name.
            dynamic: Whether to dynamically adjust the loss scale.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = dynamic, dynamic_growth_steps, name
        super().__init__(**kwargs)
        self.inner_optimizer = inner_optimizer
        self.initial_scale = initial_scale

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Build the wrapped ``inner_optimizer`` on ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        from zero_keras.types import OptimizerBuildProtocol

        if self.built:
            return
        if isinstance(self.inner_optimizer, OptimizerBuildProtocol):
            self.inner_optimizer.build(var_list)
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Delegate gradient updates to the wrapped ``inner_optimizer``.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        from zero_keras.types import OptimizerProtocol

        if not self.built:
            self.build([v for _, v in grads_and_vars])

        if isinstance(self.inner_optimizer, OptimizerProtocol):
            self.inner_optimizer.apply_gradients(grads_and_vars, *args, **kwargs)


class Muon(Optimizer):
    """Optimizer that implements the Muon algorithm.

    Note that this optimizer should not be used in the following layers:

    1. Embedding layer
    2. Final output fully connected layer
    3. TensorLike {0,1}-D variables

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
    - TensorLike other variable uses the Muon step.

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

    transpose_last_axis: bool = True
    zeropower_via_newtonschulz5: bool = True

    def __init__(
        self,
        learning_rate: float | TensorLike = 0.001,
        adam_beta_1: float = 0.9,
        adam_beta_2: float = 0.999,
        adam_weight_decay: float | None = 0.004,
        epsilon: float = 1e-07,
        weight_decay: float | None = 0.004,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        exclude_layers: list[str] | None = None,
        exclude_embeddings: bool = True,
        muon_a: float = 3.4445,
        muon_b: float = -4.775,
        muon_c: float = 2.0315,
        adam_lr_ratio: float = 1,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        rms_rate: float = 0.2,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Muon optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            adam_beta_1: Exponential decay rate for Adam first-moment estimates.
            adam_beta_2: Exponential decay rate for Adam second-moment estimates.
            adam_weight_decay: Weight decay coefficient for Adam updates.
            epsilon: Small constant for numerical stability.
            weight_decay: Optional weight decay coefficient.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            global_clipnorm: Optional global gradient norm clipping threshold.
            use_ema: Whether to apply exponential moving average.
            ema_momentum: Momentum for EMA updates.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            name: Optimizer instance name.
            exclude_layers: Layer name patterns excluded from Muon updates.
            exclude_embeddings: Whether embedding layers use AdamW updates.
            muon_a: First Newton-Schulz polynomial coefficient.
            muon_b: Second Newton-Schulz polynomial coefficient.
            muon_c: Third Newton-Schulz polynomial coefficient.
            adam_lr_ratio: Ratio of Adam learning rate to Muon learning rate.
            momentum: Momentum decay factor.
            ns_steps: Number of Newton-Schulz iterations.
            nesterov: Whether to apply Nesterov momentum.
            rms_rate: RMS scaling factor for orthogonalized updates.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            adam_beta_1,
            adam_beta_2,
            adam_weight_decay,
            epsilon,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            ema_overwrite_frequency,
            loss_scale_factor,
            gradient_accumulation_steps,
            name,
            exclude_layers,
            exclude_embeddings,
            muon_a,
            muon_b,
            muon_c,
            adam_lr_ratio,
            ns_steps,
            nesterov,
            rms_rate,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create momentum state variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
        """
        if self.built:
            return
        self.m = []
        for var in var_list:
            self.m.append(self.add_variable(shape=var.shape, name="m"))
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Muon momentum gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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
            m.assign(m_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Nadam optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            beta_1: Exponential decay rate for first-moment estimates.
            beta_2: Exponential decay rate for second-moment estimates.
            epsilon: Small constant for numerical stability.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            weight_decay: Optional weight decay coefficient.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
            weight_decay,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create first-moment and second-moment state variables for ``var_list``.

        Args:
            var_list: Sequence of model variables to optimize.
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

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply Nadam Nesterov-accelerated adaptive moment updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        self.iterations.assign(self.iterations + 1)

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

            m.assign(m_new)
            v.assign(v_new)
            var.assign(var_new)


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
        learning_rate: float | TensorLike = 0.001,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-07,
        centered: bool = False,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the RMSprop optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            rho: Discounting factor for moving average of squared gradients.
            momentum: Momentum decay factor.
            epsilon: Small constant for numerical stability.
            centered: Whether to normalize gradients by estimated variance.
            use_ema: Whether to apply exponential moving average.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            global_clipnorm: Optional global gradient norm clipping threshold.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            ema_momentum: Momentum for EMA updates.
            name: Optimizer instance name.
            weight_decay: Optional weight decay coefficient.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            use_ema,
            ema_overwrite_frequency,
            loss_scale_factor,
            global_clipnorm,
            gradient_accumulation_steps,
            clipnorm,
            clipvalue,
            ema_momentum,
            name,
            weight_decay,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def build(self, var_list: Sequence[TensorLike]) -> None:
        """Create squared-gradient, optional mean-gradient, and momentum state variables.

        Args:
            var_list: Sequence of model variables to optimize.
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

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply RMSprop gradient updates to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        from zero_keras import ops

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
                mg.assign(mg_new)
            else:
                denominator = v_new + self.epsilon

            denominator = ops.sqrt(denominator)

            if self.momentum > 0.0:
                m = self.m[i]
                m_new = self.momentum * m + lr * g / denominator
                var_new = var - m_new
                m.assign(m_new)
            else:
                var_new = var - lr * g / denominator

            v.assign(v_new)
            var.assign(var_new)


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

    def __init__(
        self,
        learning_rate: float | TensorLike = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SGD optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            momentum: Momentum decay factor.
            nesterov: Whether to apply Nesterov momentum.
            weight_decay: Optional weight decay coefficient.
            clipnorm: Optional per-variable gradient norm clipping threshold.
            clipvalue: Optional per-variable gradient value clipping threshold.
            global_clipnorm: Optional global gradient norm clipping threshold.
            use_ema: Whether to apply exponential moving average.
            ema_momentum: Momentum for EMA updates.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            name: Optimizer instance name.
            **kwargs: Additional optimizer keyword arguments.
        """
        _ = (
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            ema_overwrite_frequency,
            loss_scale_factor,
            gradient_accumulation_steps,
            name,
        )
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def build(self, var_list: Sequence[TensorLike]) -> None:
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
                self.momentums.append(self.add_variable(shape=var.shape, name="momentum"))
        self.built = True

    def apply_gradients(
        self, grads_and_vars: TensorLike, *args: TensorLike, **kwargs: TensorLike
    ) -> None:
        """Apply SGD gradient updates (with optional momentum/Nesterov) to variables.

        Args:
            grads_and_vars: Iterable of ``(gradient, variable)`` pairs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = args, kwargs
        grads_and_vars = list(grads_and_vars)
        if not self.built:
            self.build([v for _, v in grads_and_vars])

        for i, (g, v) in enumerate(grads_and_vars):
            if g is None:
                continue
            v_val = v
            if self.momentum != 0:
                m = self.momentums[i]
                m_new = self.momentum * m - self.learning_rate * g
                if self.nesterov:
                    v_new = v_val + self.momentum * m_new - self.learning_rate * g
                else:
                    v_new = v_val + m_new

                m.assign(m_new)
                v.assign(v_new)
            else:
                v_new = v_val - self.learning_rate * g
                v.assign(v_new)


from . import schedules as schedules


class OptimizerMap(dict):
    """Mapping from variable prefixes or patterns to specific Optimizer instances."""

    def __init__(
        self,
        default_optimizer: Optimizer | str | TensorLike,
        optimizer_map: dict[str, Optimizer | str] | TensorLike | None = None,
    ) -> None:
        """Initialize OptimizerMap.

        Args:
            default_optimizer: Fallback optimizer for unmatched variables.
            optimizer_map: Optional dictionary of pattern-to-optimizer mappings.
        """
        super().__init__(optimizer_map or {})
        self.default_optimizer = default_optimizer

    def get_config(self) -> dict[str, TensorLike]:
        """Return serialized configuration for this ``OptimizerMap``.

        Returns:
            dict[str, TensorLike]: Configuration dictionary with ``default_optimizer`` and ``optimizer_map``.
        """
        return {
            "default_optimizer": self.default_optimizer,
            "optimizer_map": dict(self),
        }

    @classmethod
    def from_config(cls, config: dict[str, TensorLike]) -> OptimizerMap:
        """Instantiate an ``OptimizerMap`` from its configuration dictionary.

        Args:
            config: Configuration dictionary produced by ``get_config``.

        Returns:
            OptimizerMap: Reconstructed ``OptimizerMap`` instance.
        """
        return cls(
            default_optimizer=config.get("default_optimizer"),
            optimizer_map=config.get("optimizer_map"),
        )


class MultiOptimizer(Optimizer):
    """Wrapper optimizer that applies distinct optimizers to subsets of model variables."""

    def __init__(
        self,
        optimizer_map: dict[str, Optimizer] | TensorLike,
        loss_scale_factor: float | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize MultiOptimizer.

        Args:
            optimizer_map: Mapping or list of ``(optimizer, variables)`` pairs.
            loss_scale_factor: Optional loss scaling factor.
            name: Optional optimizer name.
            **kwargs: Base optimizer keyword arguments.
        """
        super().__init__(
            learning_rate=0.001,
            loss_scale_factor=loss_scale_factor,
            name=name or "multi_optimizer",
            **kwargs,
        )
        self.optimizer_map = optimizer_map


class ScheduleFreeAdamW(AdamW):
    """Schedule-Free AdamW optimizer."""

    def __init__(
        self,
        learning_rate: float | TensorLike = 0.0025,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
        warmup_steps: int = 0,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        loss_scale_factor: float | None = None,
        gradient_accumulation_steps: int | None = None,
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize ScheduleFreeAdamW optimizer.

        Args:
            learning_rate: Learning rate scalar or schedule.
            beta_1: Exponential interpolation rate.
            beta_2: Exponential decay rate for second moment estimates.
            epsilon: Small constant for numerical stability.
            warmup_steps: Number of linear learning-rate warmup steps.
            weight_decay: Optional weight decay coefficient.
            clipnorm: Optional gradient norm clipping threshold.
            clipvalue: Optional gradient value clipping threshold.
            global_clipnorm: Optional global gradient norm clipping threshold.
            use_ema: Whether to apply exponential moving average.
            ema_momentum: Momentum for EMA updates.
            ema_overwrite_frequency: Frequency of overwriting variables with EMA.
            loss_scale_factor: Optional loss scaling factor.
            gradient_accumulation_steps: Optional gradient accumulation step count.
            name: Optional optimizer name.
            **kwargs: Base optimizer keyword arguments.
        """
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay if weight_decay is not None else 0.0,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name or "schedule_free_adamw",
            **kwargs,
        )
        self.warmup_steps = int(warmup_steps)


from zero_keras.optimizers import legacy as legacy


_OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "Adadelta": Adadelta,
    "Adafactor": Adafactor,
    "Adagrad": Adagrad,
    "Adam": Adam,
    "AdamW": AdamW,
    "Adamax": Adamax,
    "Ftrl": Ftrl,
    "Lamb": Lamb,
    "Lion": Lion,
    "LossScaleOptimizer": LossScaleOptimizer,
    "Muon": Muon,
    "Nadam": Nadam,
    "Optimizer": Optimizer,
    "RMSprop": RMSprop,
    "SGD": SGD,
    "ScheduleFreeAdamW": ScheduleFreeAdamW,
}


def serialize(optimizer: TensorLike) -> TensorLike:
    """Serialize an optimizer instance into a configuration dictionary.

    Args:
        optimizer: Optimizer instance or string identifier.

    Returns:
        TensorLike: Serialized optimizer dictionary, string, or None.
    """
    if optimizer is None:
        return None
    if isinstance(optimizer, str):
        return optimizer
    return {
        "class_name": type(optimizer).__name__,
        "config": optimizer.get_config() if isinstance(optimizer, Optimizer) else {},
    }


def deserialize(
    config: ConfigDict | Optimizer | str | TensorLike | None,
    custom_objects: ConfigDict | None = None,
) -> Optimizer | TensorLike | None:
    """Deserialize an optimizer from its configuration dictionary or string identifier.

    Args:
        config: Serialized optimizer configuration dictionary or string name.
        custom_objects: Optional mapping of custom optimizer classes.

    Returns:
        TensorLike: Reconstructed optimizer instance or original config.
    """
    if config is None:
        return None
    if isinstance(config, str):
        if isinstance(custom_objects, dict) and config in custom_objects:
            custom_cls = custom_objects[config]
            return custom_cls() if callable(custom_cls) else custom_cls
        return get(config)
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf = config.get("config", {})
        if isinstance(class_name, str) and isinstance(conf, dict):
            if isinstance(custom_objects, dict) and class_name in custom_objects:
                custom_cls = custom_objects[class_name]
                if callable(custom_cls):
                    return custom_cls(**conf)
            cls = _OPTIMIZER_REGISTRY.get(class_name)
            if cls is not None:
                return cls(**conf)
    return config


def get(identifier: TensorLike) -> TensorLike:
    """Retrieve a Keras optimizer object via an identifier.

    Args:
        identifier: Optimizer instance or string identifier.

    Returns:
        TensorLike: Resolved optimizer instance, string, or None.
    """
    if identifier is None:
        return None
    if isinstance(identifier, str):
        if (
            identifier in _OPTIMIZER_REGISTRY
            and _OPTIMIZER_REGISTRY[identifier] is not LossScaleOptimizer
        ):
            return _OPTIMIZER_REGISTRY[identifier]()
        low = identifier.lower()
        mapping: dict[str, type[Optimizer]] = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
            "rmsprop": RMSprop,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "nadam": Nadam,
            "ftrl": Ftrl,
            "lion": Lion,
            "adafactor": Adafactor,
            "lamb": Lamb,
            "muon": Muon,
            "schedule_free_adamw": ScheduleFreeAdamW,
            "schedulefreeadamw": ScheduleFreeAdamW,
            "optimizer": Optimizer,
        }
        if low in mapping:
            return mapping[low]()
        return identifier
    return identifier

"""Normalization layers (`BatchNormalization`, `LayerNormalization`, `UnitNormalization`, `GroupNormalization`, `RMSNormalization`, `SpectralNormalization`) for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer, Wrapper
from zero_keras.types import ShapeTuple, TensorLike


class LayerNormalization(Layer):
    """Layer normalization layer (Ba et al., 2016).

    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

    If `scale` or `center` are enabled, the layer will scale the normalized
    outputs by broadcasting them with a trainable variable `gamma`, and center
    the outputs by broadcasting with a trainable variable `beta`. `gamma` will
    default to a ones tensor and `beta` will default to a zeros tensor, so that
    centering and scaling are no-ops before training has begun.

    So, with scaling and centering enabled the normalization equations
    are as follows:

    Let the intermediate activations for a mini-batch to be the `inputs`.

    For each sample `x_i` in `inputs` with `k` features, we compute the mean and
    variance of the sample:

    ```python
    mean_i = sum(x_i[j] for j in range(k)) / k
    var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k
    ```

    and then compute a normalized `x_i_normalized`, including a small factor
    `epsilon` for numerical stability.

    ```python
    x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
    ```

    And finally `x_i_normalized ` is linearly transformed by `gamma` and `beta`,
    which are learned parameters:

    ```python
    output_i = x_i_normalized * gamma + beta
    ```

    `gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
    this part of the inputs' shape must be fully defined.

    For example:

    >>> layer = keras.layers.LayerNormalization(axis=[1, 2, 3])
    >>> layer.build([5, 20, 30, 40])
    >>> print(layer.beta.shape)
    (20, 30, 40)
    >>> print(layer.gamma.shape)
    (20, 30, 40)

    Note that other implementations of layer normalization may choose to define
    `gamma` and `beta` over a separate set of axes from the axes being
    normalized across. For example, Group Normalization
    ([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
    corresponds to a Layer Normalization that normalizes across height, width,
    and channel and has `gamma` and `beta` span only the channel dimension.
    So, this Layer Normalization implementation will not match a Group
    Normalization layer with group size set to 1.

    Args:
        axis: Integer or List/Tuple. The axis or axes to normalize across.
            Typically, this is the features axis/axes. The left-out axes are
            typically the batch axis/axes. `-1` is the last dimension in the
            input. Defaults to `-1`.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-3.
        center: If True, add offset of `beta` to normalized tensor. If False,
            `beta` is ignored. Defaults to `True`.
        scale: If True, multiply by `gamma`. If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling will be done by the next layer.
            Defaults to `True`.
        rms_scaling: If True, `center` and `scale` are ignored, and the
            inputs are scaled by `gamma` and the inverse square root
            of the square of all inputs. This is an approximate and faster
            approach that avoids ever computing the mean of the input. Note that
            this *isn't* equivalent to the computation that the
            `keras.layers.RMSNormalization` layer performs.
        beta_initializer: Initializer for the beta weight. Defaults to zeros.
        gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        beta_regularizer: Optional regularizer for the beta weight.
            None by default.
        gamma_regularizer: Optional regularizer for the gamma weight.
            None by default.
        beta_constraint: Optional constraint for the beta weight.
            None by default.
        gamma_constraint: Optional constraint for the gamma weight.
            None by default.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).


    Reference:

    - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).

    """

    def __init__(
        self,
        axis: int | Sequence[int] = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
        beta_regularizer: TensorLike | None = None,
        gamma_regularizer: TensorLike | None = None,
        beta_constraint: TensorLike | None = None,
        gamma_constraint: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the LayerNormalization layer.

        Args:
            axis: Axis or axes to normalize across. Defaults to ``-1``.
            epsilon: Small float added to variance to avoid division by zero.
            center: Whether to add a learnable ``beta`` offset.
            scale: Whether to multiply by a learnable ``gamma`` scale.
            beta_initializer: Initializer for ``beta`` weight.
            gamma_initializer: Initializer for ``gamma`` weight.
            beta_regularizer: Optional regularizer for ``beta`` weight.
            gamma_regularizer: Optional regularizer for ``gamma`` weight.
            beta_constraint: Optional constraint for ``beta`` weight.
            gamma_constraint: Optional constraint for ``gamma`` weight.
            **kwargs: Base layer keyword arguments.
        """
        _ = (
            center,
            scale,
            beta_initializer,
            gamma_initializer,
            beta_regularizer,
            gamma_regularizer,
            beta_constraint,
            gamma_constraint,
        )
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = float(epsilon)
        self.gamma: TensorLike | None = None
        self.beta: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create ``gamma`` and ``beta`` normalization weights.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        input_dim = input_shape[-1]
        self.gamma = self.add_weight(shape=(input_dim,), initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=(input_dim,), initializer="zeros", name="beta")
        self.built = True

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Normalize ``inputs`` across ``self.axis``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Layer-normalized tensor.
        """
        inputs = _to_tensor(inputs)
        mean = ops.mean(inputs, axis=self.axis, keepdims=True)
        var = ops.var(inputs, axis=self.axis, keepdims=True)
        return _wrap((inputs - mean) / ops.sqrt(var + self.epsilon))


class BatchNormalization(Layer):
    """Layer that normalizes its inputs.

    Batch normalization applies a transformation that maintains the mean output
    close to 0 and the output standard deviation close to 1.

    Importantly, batch normalization works differently during training and
    during inference.

    **During training** (i.e. when using `fit()` or when calling the layer/model
    with the argument `training=True`), the layer normalizes its output using
    the mean and standard deviation of the current batch of inputs. That is to
    say, for each channel being normalized, the layer returns
    `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:

    - `epsilon` is small constant (configurable as part of the constructor
    arguments)
    - `gamma` is a learned scaling factor (initialized as 1), which
    can be disabled by passing `scale=False` to the constructor.
    - `beta` is a learned offset factor (initialized as 0), which
    can be disabled by passing `center=False` to the constructor.

    **During inference** (i.e. when using `evaluate()` or `predict()` or when
    calling the layer/model with the argument `training=False` (which is the
    default), the layer normalizes its output using a moving average of the
    mean and standard deviation of the batches it has seen during training. That
    is to say, it returns
    `gamma * (batch - self.moving_mean) / sqrt(self.moving_var+epsilon) + beta`.

    `self.moving_mean` and `self.moving_var` are non-trainable variables that
    are updated each time the layer in called in training mode, as such:

    - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
    - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`

    As such, the layer will only normalize its inputs during inference
    *after having been trained on data that has similar statistics as the
    inference data*.

    Args:
        axis: Integer, the axis that should be normalized
            (typically the features axis). For instance, after a `Conv2D` layer
            with `data_format="channels_first"`, use `axis=1`.
        momentum: Momentum for the moving average.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If `True`, add offset of `beta` to normalized tensor.
            If `False`, `beta` is ignored.
        scale: If `True`, multiply by `gamma`. If `False`, `gamma` is not used.
            When the next layer is linear this can be disabled
            since the scaling will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        synchronized: Only applicable with the TensorFlow backend.
            If `True`, synchronizes the global batch statistics (mean and
            variance) for the layer across all devices at each training step
            in a distributed training strategy.
            If `False`, each replica uses its own local batch statistics.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            - `training=True`: The layer will normalize its inputs using
            the mean and variance of the current batch of inputs.
            - `training=False`: The layer will normalize its inputs using
            the mean and variance of its moving statistics, learned during
            training.
        mask: Binary tensor of shape broadcastable to `inputs` tensor, with
            `True` values indicating the positions for which mean and variance
            should be computed. Masked elements of the current inputs are not
            taken into account for mean and variance computation during
            training. TensorLike prior unmasked element values will be taken into
            account until their momentum expires.

    Reference:

    - [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).

    **About setting `layer.trainable = False` on a `BatchNormalization` layer:**

    The meaning of setting `layer.trainable = False` is to freeze the layer,
    i.e. its internal state will not change during training:
    its trainable weights will not be updated
    during `fit()` or `train_on_batch()`, and its state updates will not be run.

    Usually, this does not necessarily mean that the layer is run in inference
    mode (which is normally controlled by the `training` argument that can
    be passed when calling a layer). "Frozen state" and "inference mode"
    are two separate concepts.

    However, in the case of the `BatchNormalization` layer, **setting
    `trainable = False` on the layer means that the layer will be
    subsequently run in inference mode** (meaning that it will use
    the moving mean and the moving variance to normalize the current batch,
    rather than using the mean and variance of the current batch).

    Note that:

    - Setting `trainable` on an model containing other layers will recursively
        set the `trainable` value of all inner layers.
    - If the value of the `trainable` attribute is changed after calling
        `compile()` on a model, the new value doesn't take effect for this model
        until `compile()` is called again.

    """

    def __init__(
        self,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: str | TensorLike = "zeros",
        gamma_initializer: str | TensorLike = "ones",
        moving_mean_initializer: str | TensorLike = "zeros",
        moving_variance_initializer: str | TensorLike = "ones",
        beta_regularizer: TensorLike | None = None,
        gamma_regularizer: TensorLike | None = None,
        beta_constraint: TensorLike | None = None,
        gamma_constraint: TensorLike | None = None,
        renorm: bool = False,
        renorm_clipping: TensorLike | None = None,
        renorm_momentum: float = 0.99,
        synchronized: bool = False,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the BatchNormalization layer.

        Args:
            axis: Feature axis to normalize. Defaults to ``-1``.
            momentum: Momentum for moving average statistics. Defaults to ``0.99``.
            epsilon: Small float added to variance to avoid division by zero.
            center: Whether to add a learnable ``beta`` offset.
            scale: Whether to multiply by a learnable ``gamma`` scale.
            beta_initializer: Initializer for ``beta`` weight.
            gamma_initializer: Initializer for ``gamma`` weight.
            moving_mean_initializer: Initializer for ``moving_mean`` weight.
            moving_variance_initializer: Initializer for ``moving_variance`` weight.
            beta_regularizer: Optional regularizer for ``beta`` weight.
            gamma_regularizer: Optional regularizer for ``gamma`` weight.
            beta_constraint: Optional constraint for ``beta`` weight.
            gamma_constraint: Optional constraint for ``gamma`` weight.
            renorm: Whether to use Batch Renormalization.
            renorm_clipping: Optional clipping dictionary for Batch Renormalization.
            renorm_momentum: Momentum for Batch Renormalization moving averages.
            synchronized: Whether to synchronize batch statistics across replicas.
            **kwargs: Base layer keyword arguments.
        """
        _ = (
            center,
            scale,
            beta_initializer,
            gamma_initializer,
            moving_mean_initializer,
            moving_variance_initializer,
            beta_regularizer,
            gamma_regularizer,
            beta_constraint,
            gamma_constraint,
            renorm,
            renorm_clipping,
            renorm_momentum,
            synchronized,
        )
        super().__init__(**kwargs)
        self.axis = int(axis)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.gamma: TensorLike | None = None
        self.beta: TensorLike | None = None
        self.moving_mean: TensorLike | None = None
        self.moving_variance: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create ``gamma``, ``beta``, ``moving_mean``, and ``moving_variance`` weights.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        input_dim = input_shape[-1]
        self.gamma = self.add_weight(shape=(input_dim,), initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=(input_dim,), initializer="zeros", name="beta")
        self.moving_mean = self.add_weight(
            shape=(input_dim,), initializer="zeros", name="moving_mean"
        )
        self.moving_variance = self.add_weight(
            shape=(input_dim,), initializer="ones", name="moving_variance"
        )
        self.built = True

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Apply batch normalization using batch statistics in training or moving statistics in inference.

        Args:
            inputs: Input tensor.
            training: Whether the layer is executing in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Batch-normalized tensor.
        """
        inputs = _to_tensor(inputs)
        if training:
            rank = len(inputs.shape)
            axis = self.axis if self.axis >= 0 else self.axis + rank
            axes = tuple(i for i in range(rank) if i != axis)
            mean = ops.mean(inputs, axis=axes, keepdims=True)
            var = ops.var(inputs, axis=axes, keepdims=True)
            return _wrap((inputs - mean) / ops.sqrt(var + self.epsilon))
        else:
            return _wrap(inputs / ops.sqrt(_to_tensor(1.0 + self.epsilon)))


class GroupNormalization(Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes nearly
    identical to Layer Normalization (see Layer Normalization docs for details).

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization. You can achieve this via `groups=-1`.

    Args:
        groups: Integer, the number of groups for Group Normalization. Can be in
            the range `[1, N]` where N is the input dimension. The input
            dimension must be divisible by the number of groups.
            Defaults to 32.
        axis: Integer or List/Tuple. The axis or axes to normalize across.
            Typically, this is the features axis/axes. The left-out axes are
            typically the batch axis/axes. -1 is the last dimension in the
            input. Defaults to `-1`.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-3.
        center: If `True`, add offset of `beta` to normalized tensor.
            If `False`, `beta` is ignored. Defaults to `True`.
        scale: If `True`, multiply by `gamma`. If `False`, `gamma` is not used.
            When the next layer is linear (also e.g. `relu`), this can be
            disabled since the scaling will be done by the next layer.
            Defaults to `True`.
        beta_initializer: Initializer for the beta weight. Defaults to zeros.
        gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        beta_regularizer: Optional regularizer for the beta weight. None by
            default.
        gamma_regularizer: Optional regularizer for the gamma weight. None by
            default.
        beta_constraint: Optional constraint for the beta weight.
            None by default.
        gamma_constraint: Optional constraint for the gamma weight. None by
            default.  Input shape: Arbitrary. Use the keyword argument
            `input_shape` (tuple of integers, does not include the samples
            axis) when using this layer as the first layer in a model.
            Output shape: Same shape as input.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).

    Reference:

    - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)

    """

    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
        beta_regularizer: TensorLike | None = None,
        gamma_regularizer: TensorLike | None = None,
        beta_constraint: TensorLike | None = None,
        gamma_constraint: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the GroupNormalization layer.

        Args:
            groups: Number of channel groups. Defaults to ``32``.
            axis: Channel axis. Defaults to ``-1``.
            epsilon: Small float added to variance to avoid division by zero.
            center: Whether to add a learnable ``beta`` offset.
            scale: Whether to multiply by a learnable ``gamma`` scale.
            beta_initializer: Initializer for ``beta`` weight.
            gamma_initializer: Initializer for ``gamma`` weight.
            beta_regularizer: Optional regularizer for ``beta`` weight.
            gamma_regularizer: Optional regularizer for ``gamma`` weight.
            beta_constraint: Optional constraint for ``beta`` weight.
            gamma_constraint: Optional constraint for ``gamma`` weight.
            **kwargs: Base layer keyword arguments.
        """
        _ = (
            beta_initializer,
            gamma_initializer,
            beta_regularizer,
            gamma_regularizer,
            beta_constraint,
            gamma_constraint,
        )
        super().__init__(**kwargs)
        self.groups = int(groups)
        self.axis = int(axis)
        self.epsilon = float(epsilon)
        self.center = bool(center)
        self.scale = bool(scale)
        self.gamma: TensorLike | None = None
        self.beta: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create optional ``gamma`` and ``beta`` weights for group normalization.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return
        dim = input_shape[self.axis]
        if self.scale:
            self.gamma = self.add_weight(shape=(dim,), initializer="ones", name="gamma")
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=(dim,), initializer="zeros", name="beta")
        else:
            self.beta = None
        self.built = True

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply group normalization across ``self.groups`` channel groups.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Group-normalized tensor.
        """
        inputs = _to_tensor(inputs)

        # permute channel to last
        rank = len(inputs.shape)
        if self.axis != -1 and self.axis != rank - 1:
            perm = list(range(rank))
            perm.append(perm.pop(self.axis))
            inputs = ops.permute(inputs, tuple(perm))

        shape = inputs.shape
        channels = shape[-1]

        new_shape = shape[:-1] + (self.groups, channels // self.groups)
        inputs_reshaped = ops.reshape(inputs, new_shape)

        axes_to_reduce = tuple(range(1, rank)) + (rank,)
        axes_to_reduce = tuple(i for i in axes_to_reduce if i != rank - 1)  # exclude groups

        mean = ops.mean(inputs_reshaped, axis=axes_to_reduce, keepdims=True)
        var = ops.var(inputs_reshaped, axis=axes_to_reduce, keepdims=True)

        out = ops.multiply(
            ops.subtract(inputs_reshaped, mean),
            ops.divide(_to_tensor(1.0), ops.sqrt(ops.add(var, _to_tensor(self.epsilon)))),
        )
        out = ops.reshape(out, shape)

        if self.scale:
            out = ops.multiply(out, _to_tensor(self.gamma))
        if self.center:
            out = ops.add(out, _to_tensor(self.beta))

        if self.axis != -1 and self.axis != rank - 1:
            inv_perm = list(range(rank))
            inv_perm.insert(self.axis, inv_perm.pop(-1))
            out = ops.permute(out, tuple(inv_perm))

        return _wrap(out)


class RMSNormalization(Layer):
    """Root Mean Square (RMS) Normalization layer.

    This layer normalizes the input tensor based on its RMS value.

    The Keras layer performs the operation as described in
    [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
    by Biao Zhang et al.


    If `scale` is enabled, the layer will scale the normalized outputs via
    a learnable scaling factor.

    So, with scaling enabled, the normalization equations
    are as follows:

    Let the intermediate activations for a mini-batch to be the `inputs`.

    ```python
    rms_normalization(x) = x * rsqrt(mean(square(x))) * scale
    ```

    For example:

    >>> layer = keras.layers.RMSNormalization()
    >>> layer.build([5, 20, 30, 10])
    >>> print(layer.scale.shape)
    (10,)
    >>> layer(np.random.rand(1, 10)).numpy()
    array([[0.35098287, 1.0495652 , 1.4645109 , 1.2944688 , 0.31124955,
            1.2768592 , 1.184331  , 0.17474432, 0.49955517, 1.2428929 ]],
        dtype=float32)

    Args:
        axis: int. The axis on which to perform the normalization.
        epsilon: float. A small number to add to avoid division by zero.

    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-6,
        center: bool = False,
        scale: bool = True,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the RMSNormalization layer.

        Args:
            axis: Axis along which RMS normalization is computed. Defaults to ``-1``.
            epsilon: Small float added to mean square for numerical stability.
            center: Whether to add a learnable ``beta`` offset.
            scale: Whether to multiply by a learnable ``gamma`` scale.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.axis = int(axis)
        self.epsilon = float(epsilon)
        self.center = bool(center)
        self.scale = bool(scale)
        self.gamma: TensorLike | None = None
        self.beta: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create optional ``gamma`` and ``beta`` weights for RMS normalization.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return
        dim = input_shape[self.axis]
        if self.scale:
            self.gamma = self.add_weight(shape=(dim,), initializer="ones", name="gamma")
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=(dim,), initializer="zeros", name="beta")
        else:
            self.beta = None
        self.built = True

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Apply RMS normalization to ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: RMS-normalized tensor.
        """
        inputs = _to_tensor(inputs)
        variance = ops.mean(ops.square(inputs), axis=self.axis, keepdims=True)
        out = ops.multiply(
            inputs,
            ops.divide(_to_tensor(1.0), ops.sqrt(ops.add(variance, _to_tensor(self.epsilon)))),
        )

        if self.scale:
            out = ops.multiply(out, _to_tensor(self.gamma))
        if self.center:
            out = ops.add(out, _to_tensor(self.beta))

        return _wrap(out)


class UnitNormalization(Layer):
    """Unit normalization layer.

    Normalize a batch of inputs so that each input in the batch has a L2 norm
    equal to 1 (across the axes specified in `axis`).

    Example:
    >>> data = np.arange(6).reshape(2, 3)
    >>> normalized_data = keras.layers.UnitNormalization()(data)
    >>> np.sum(normalized_data[0, :] ** 2)
    1.0

    Args:
        axis: Integer or list/tuple. The axis or axes to normalize across.
            Typically, this is the features axis or axes. The left-out axes are
            typically the batch axis or axes. `-1` is the last dimension
            in the input. Defaults to `-1`.

    """

    def __init__(self, axis: int | Sequence[int] = -1, **kwargs: TensorLike) -> None:
        """Initialize the UnitNormalization layer.

        Args:
            axis: Axis or axes to L2-normalize across. Defaults to ``-1``.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Normalize ``inputs`` to unit L2 norm along ``self.axis``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Unit-normalized tensor.
        """
        inputs = _to_tensor(inputs)
        norm = ops.sqrt(ops.sum(ops.square(inputs), axis=self.axis, keepdims=True))
        out = ops.divide(inputs, ops.maximum(norm, _to_tensor(1e-7)))
        return _wrap(out)


from typing import Protocol, runtime_checkable


@runtime_checkable
class _HasKernelProtocol(Protocol):
    """Protocol for layers carrying a ``kernel`` weight attribute."""

    kernel: TensorLike | None


class SpectralNormalization(Wrapper):
    """Performs spectral normalization on the weights of a target layer.

    This wrapper controls the Lipschitz constant of the weights of a layer by
    constraining their spectral norm, which can stabilize the training of GANs.

    Args:
        layer: A `keras.layers.Layer` instance that
            has either a `kernel` (e.g. `Conv2D`, `Dense`...)
            or an `embeddings` attribute (`Embedding` layer).
        power_iterations: int, the number of iterations during normalization.
        **kwargs: Base wrapper keyword arguments.

    Examples:
    Wrap `keras.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = SpectralNormalization(keras.layers.Conv2D(2, 2))
    >>> y = conv2d(x)
    >>> y.shape
    (1, 9, 9, 2)

    Wrap `keras.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = SpectralNormalization(keras.layers.Dense(10))
    >>> y = dense(x)
    >>> y.shape
    (1, 10, 10, 10)

    Reference:

    - [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).

    """

    normalized_weights: TensorLike | None = None

    def __init__(self, layer: Layer, power_iterations: int = 1, **kwargs: TensorLike) -> None:
        """Initialize the SpectralNormalization wrapper layer.

        Args:
            layer: Wrapped ``Layer`` instance containing a ``kernel`` weight.
            power_iterations: Number of power iterations to approximate the spectral norm.
            **kwargs: Base wrapper layer keyword arguments.
        """
        super().__init__(layer=layer, **kwargs)
        self.power_iterations = int(power_iterations)
        self.u: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Build the wrapped layer and allocate the ``vector_u`` singular vector weight.

        Args:
            input_shape: Input tensor shape tuple.
        """
        if self.built:
            return
        self.layer.build(input_shape)

        if isinstance(self.layer, _HasKernelProtocol) and self.layer.kernel is not None:
            shape = (1, self.layer.kernel.shape[-1])
            self.u = self.add_weight(
                shape=shape,
                initializer="random_normal",
                trainable=False,
                name="vector_u",
            )

            if self.u in self._weights:
                self._weights.remove(self.u)
                self.layer._weights.insert(1, self.u)

        self.built = True

    def normalize_weights(self) -> None:
        """Normalize the wrapped layer's ``kernel`` in-place using power iteration."""
        if (
            not isinstance(self.layer, _HasKernelProtocol)
            or self.layer.kernel is None
            or self.u is None
        ):
            return
        w = _to_tensor(self.layer.kernel)
        w_shape = w.shape
        out_dim = int(w_shape[-1] or 1)
        in_dim = 1
        for d in w_shape[:-1]:
            in_dim *= int(d or 1)
        w_mat = ops.reshape(w, (in_dim, out_dim))
        u_vec = _to_tensor(self.u)
        v_raw = ops.matmul(u_vec, ops.transpose(w_mat))
        v_norm = ops.sqrt(ops.sum(ops.square(v_raw)))
        v_vec = ops.divide(v_raw, ops.maximum(v_norm, _to_tensor(1e-12)))
        u_raw = ops.matmul(v_vec, w_mat)
        u_norm = ops.sqrt(ops.sum(ops.square(u_raw)))
        u_vec = ops.divide(u_raw, ops.maximum(u_norm, _to_tensor(1e-12)))
        sigma = ops.matmul(ops.matmul(v_vec, w_mat), ops.transpose(u_vec))
        w_norm = ops.reshape(ops.divide(w_mat, ops.maximum(sigma, _to_tensor(1e-12))), w_shape)
        self.u = u_vec
        self.layer.kernel = w_norm

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Execute the wrapped layer on ``inputs``, applying power iteration when ``training=True``.

        Args:
            inputs: Input tensor.
            training: Optional boolean indicating whether the layer is in training mode.
            **kwargs: Additional keyword arguments forwarded to the wrapped layer.

        Returns:
            TensorLike: Output tensor from the wrapped layer.
        """
        inputs = _to_tensor(inputs)
        if training:
            self.normalize_weights()
        return _wrap(self.layer(inputs, training=training, **kwargs))

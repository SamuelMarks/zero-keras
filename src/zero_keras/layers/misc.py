"""Miscellaneous and wrapper layers (`Embedding`, `EinsumDense`, `Identity`, `TorchModuleWrapper`, `FlaxLayer`, `JaxLayer`, `TFSMLayer`) for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.activations import get as get_activation
from zero_keras.core_layers import Layer, Wrapper
from zero_keras.types import (
    ActivationSpec,
    ConfigDict,
    ConstraintSpec,
    InitializerSpec,
    RegularizerSpec,
    ShapeTuple,
    SupportsCall,
    TensorLike,
)


class EinsumDense(Layer):
    """A layer that uses `einsum` as the backing computation.

    This layer can perform einsum calculations of arbitrary dimensionality.

    Args:
        equation: An equation describing the einsum to perform.
            This equation must be a valid einsum string of the form
            `ab,bc->ac`, `...ab,bc->...ac`, or
            `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
            axis expression sequence.
        output_shape: The expected shape of the output tensor
            (excluding the batch dimension and any dimensions
            represented by ellipses). You can specify `None` for any dimension
            that is unknown or can be inferred from the input shape.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied
            (that is, a "linear" activation: `a(x) = x`).
        bias_axes: A string containing the output dimension(s)
            to apply a bias to. Each character in the `bias_axes` string
            should correspond to a character in the output portion
            of the `equation` string.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices
            (the factorization happens on the last dimension).
            This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `EinsumDense` layer by calling `layer.enable_lora(rank)`.
         lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Examples:
    **Biased dense layer with einsums**

    This example shows how to instantiate a standard Keras dense layer using
    einsum operations. This example is equivalent to
    `keras.layers.Dense(64, use_bias=True)`.

    >>> layer = keras.layers.EinsumDense("ab,bc->ac",
    ...                                       output_shape=64,
    ...                                       bias_axes="c")
    >>> input_tensor = keras.Input(shape=[32])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 64)

    **Applying a dense layer to a sequence**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence. Here, the `output_shape` has two
    values (since there are two non-batch dimensions in the output); the first
    dimension in the `output_shape` is `None`, because the sequence dimension
    `b` has an unknown shape.

    >>> layer = keras.layers.EinsumDense("abc,cd->abd",
    ...                                       output_shape=(None, 64),
    ...                                       bias_axes="d")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)

    **Applying a dense layer to a sequence using ellipses**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence, but uses the ellipsis notation
    instead of specifying the batch and sequence dimensions.

    Because we are using ellipsis notation and have specified only one axis, the
    `output_shape` arg is a single value. When instantiated in this way, the
    layer can handle any number of sequence dimensions - including the case
    where no sequence dimension exists.

    >>> layer = keras.layers.EinsumDense("...x,xy->...y",
    ...                                       output_shape=64,
    ...                                       bias_axes="y")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)

    """

    kernel: TensorLike | None = None

    def __init__(
        self,
        equation: str,
        output_shape: ShapeTuple | int,
        activation: ActivationSpec = None,
        bias_axes: str | None = None,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        gptq_unpacked_column_size: int | None = None,
        quantization_config: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the EinsumDense layer.

        Args:
            equation: Einsum equation describing the contraction.
            output_shape: Expected non-batch output shape tuple or integer.
            activation: Activation function or identifier string.
            bias_axes: Output axis labels to which bias is applied.
            kernel_initializer: Initializer for the contraction kernel.
            bias_initializer: Initializer for the bias tensor.
            kernel_regularizer: Optional regularizer for ``kernel``.
            bias_regularizer: Optional regularizer for ``bias``.
            kernel_constraint: Optional constraint for ``kernel``.
            bias_constraint: Optional constraint for ``bias``.
            lora_rank: Optional Low-Rank Adaptation rank.
            lora_alpha: Optional Low-Rank Adaptation scaling factor.
            gptq_unpacked_column_size: Optional GPTQ unpacked column size.
            quantization_config: Optional quantization configuration.
            *args: Additional positional arguments.
            **kwargs: Base layer keyword arguments.
        """
        _ = (
            args,
            kernel_regularizer,
            bias_regularizer,
            kernel_constraint,
            bias_constraint,
            gptq_unpacked_column_size,
            quantization_config,
        )
        super().__init__(**kwargs)
        self.equation = str(equation)
        self.output_shape_tuple = (
            output_shape if isinstance(output_shape, (tuple, list)) else (output_shape,)
        )
        self.activation = get_activation(activation)
        self.bias_axes = bias_axes
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel: TensorLike | None = None
        self.bias: TensorLike | None = None
        self.lora_rank: int | None = int(lora_rank) if lora_rank is not None else None
        self.lora_alpha: float | None = float(lora_alpha) if lora_alpha is not None else None
        self.lora_enabled: bool = self.lora_rank is not None
        self.lora_kernel_a: TensorLike | None = None
        self.lora_kernel_b: TensorLike | None = None

    def enable_lora(self, rank: int, lora_alpha: float | None = None) -> None:
        """Enable Low-Rank Adaptation (LoRA) on the EinsumDense kernel.

        Args:
            rank: LoRA rank dimension.
            lora_alpha: Optional scaling factor; defaults to ``rank``.
        """
        self.lora_rank = int(rank)
        self.lora_alpha = float(lora_alpha) if lora_alpha is not None else float(rank)
        self.lora_enabled = True
        if self.built:
            k_shape = self.kernel.shape
            self.lora_kernel_a = self.add_weight(
                shape=k_shape[:-1] + (self.lora_rank,),
                initializer="he_uniform",
                trainable=True,
                name="lora_kernel_a",
            )
            self.lora_kernel_b = self.add_weight(
                shape=(self.lora_rank, k_shape[-1]),
                initializer="zeros",
                trainable=True,
                name="lora_kernel_b",
            )

    def build(self, input_shape: ShapeTuple) -> None:
        """Create the contraction ``kernel`` and optional ``bias`` weights.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return

        # Parse equation (e.g. "abc,cd->abd")
        in_labels, rest = self.equation.split(",")
        weight_labels, out_labels = rest.split("->")

        dim_map = {}
        in_labels_clean = in_labels.replace("...", "")
        for i, char in enumerate(reversed(in_labels_clean)):
            dim_map[char] = input_shape[-1 - i] if input_shape is not None else None

        for char, val in zip(reversed(out_labels), reversed(self.output_shape_tuple)):
            dim_map[char] = val

        kernel_shape = tuple(dim_map.get(char, None) for char in weight_labels)

        if self.kernel is None:
            self.kernel = self.add_weight(
                shape=kernel_shape, initializer=self.kernel_initializer, name="kernel"
            )
        if self.bias_axes is not None and self.bias is None:
            bias_shape = []
            for char in self.bias_axes:
                bias_shape.append(dim_map[char])
            self.bias = self.add_weight(
                shape=tuple(bias_shape),
                initializer=self.bias_initializer,
                name="bias",
            )
        self.built = True
        if self.lora_rank is not None:
            self.enable_lora(self.lora_rank, self.lora_alpha)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Contract ``inputs`` with ``self.kernel`` via ``ops.einsum``.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Contracted and activated output tensor.
        """
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)
        from zero_keras import ops

        kernel = self.kernel
        if self.lora_enabled and self.lora_kernel_a is not None and self.lora_kernel_b is not None:
            rank = float(self.lora_rank or 1)
            alpha = self.lora_alpha if self.lora_alpha is not None else rank
            delta = ops.matmul(_to_tensor(self.lora_kernel_a), _to_tensor(self.lora_kernel_b)) * (
                alpha / rank
            )
            kernel = ops.add(_to_tensor(kernel), delta)

        out = ops.einsum(self.equation, inputs, kernel)
        if self.bias_axes is not None:
            out = ops.add(out, self.bias)
        if self.activation:
            out = self.activation(out)
        return _wrap(out)


class Embedding(Layer):
    """Turns nonnegative integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used on nonnegative integer inputs of a fixed range.

    Example:
    >>> model = keras.Sequential()
    >>> model.add(keras.layers.Embedding(1000, 64))
    >>> # The model will take as input an integer matrix of size (batch,
    >>> # input_length), and the largest integer (i.e. word index) in the input
    >>> # should be no larger than 999 (vocabulary size).
    >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
    >>> # dimension.
    >>> input_array = np.random.randint(1000, size=(32, 10))
    >>> model.compile('rmsprop', 'mse')
    >>> output_array = model.predict(input_array)
    >>> print(output_array.shape)
    (32, 10, 64)

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
            This is useful when using recurrent layers which
            may take variable length input. If this is `True`,
            then all subsequent layers in the model need
            to support masking or an exception will be raised.
            If `mask_zero` is set to `True`, as a consequence,
            index 0 cannot be used in the vocabulary (`input_dim` should
            equal size of vocabulary + 1).
        weights: Optional floating-point matrix of size
            `(input_dim, output_dim)`. The initial embeddings values
            to use.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's embeddings
            matrix to non-trainable and replaces it with a delta over the
            original matrix, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large embedding layers.
            You can also enable LoRA on an existing
            `Embedding` layer by calling `layer.enable_lora(rank)`.
        lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.

    Input shape:
        2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, input_length, output_dim)`.

    """

    embeddings: TensorLike | None = None
    enable_lora: TensorLike | None = None

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer: InitializerSpec = "uniform",
        embeddings_regularizer: RegularizerSpec = None,
        embeddings_constraint: ConstraintSpec = None,
        mask_zero: bool = False,
        weights: TensorLike | None = None,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        quantization_config: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Embedding lookup layer.

        Args:
            input_dim: Vocabulary size (maximum integer index + 1).
            output_dim: Dimension of each dense embedding vector.
            embeddings_initializer: Initializer for the embedding table.
            embeddings_regularizer: Regularizer for the embedding table.
            embeddings_constraint: Constraint for the embedding table.
            mask_zero: Whether input token 0 is treated as a padding mask value.
            weights: Optional initial embedding weight matrices.
            lora_rank: Optional Low-Rank Adaptation rank.
            lora_alpha: Optional Low-Rank Adaptation scaling factor.
            quantization_config: Optional quantization configuration.
            **kwargs: Base layer keyword arguments.
        """
        _ = weights, lora_rank, lora_alpha, quantization_config
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint
        self.mask_zero = bool(mask_zero)
        self.embeddings: TensorLike | None = None

    def build(self, input_shape: ShapeTuple | None = None) -> None:
        """Create the ``(input_dim, output_dim)`` embedding weight matrix.

        Args:
            input_shape: Optional input tensor shape tuple.
        """
        if self.built:
            return
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            name="embeddings",
        )
        self.built = True

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Look up dense embedding vectors for integer token indices in ``inputs``.

        Args:
            inputs: Integer tensor of vocabulary indices.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Embedded tensor with trailing dimension ``output_dim``.
        """
        from zero_keras import ops

        inputs = _to_tensor(inputs)
        out = ops.take(_to_tensor(self.embeddings), inputs, axis=0)
        return _wrap(out)


class FlaxLayer(Layer):
    """Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module.

    This layer enables the use of Flax components in the form of
    [`flax.linen.Module`](
        https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)
    instances within Keras when using JAX as the backend for Keras.

    The module method to use for the forward pass can be specified via the
    `method` argument and is `__call__` by default. This method must take the
    following arguments with these exact names:

    - `self` if the method is bound to the module, which is the case for the
        default of `__call__`, and `module` otherwise to pass the module.
    - `inputs`: the inputs to the model, a JAX array or a `PyTree` of arrays.
    - `training` *(optional)*: an argument specifying if we're in training mode
        or inference mode, `True` is passed in training mode.

    `FlaxLayer` handles the non-trainable state of your model and required RNGs
    automatically. Note that the `mutable` parameter of
    [`flax.linen.Module.apply()`](
        https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply)
    is set to `DenyList(["params"])`, therefore making the assumption that all
    the variables outside of the "params" collection are non-trainable weights.

    This example shows how to create a `FlaxLayer` from a Flax `Module` with
    the default `__call__` method and no training argument:

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, inputs):
            x = inputs
            x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
            x = flax.linen.relu(x)
            x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = flax.linen.Dense(features=200)(x)
            x = flax.linen.relu(x)
            x = flax.linen.Dense(features=10)(x)
            x = flax.linen.softmax(x)
            return x

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(flax_module)
    ```

    This example shows how to wrap the module method to conform to the required
    signature. This allows having multiple input arguments and a training
    argument that has a different name and values. This additionally shows how
    to use a function that is not bound to the module.

    ```python
    class MyFlaxModule(flax.linen.Module):
        @flax.linen.compact
        def forward(self, input1, input2, deterministic):
            ...
            return outputs

    def my_flax_module_wrapper(module, inputs, training):
        input1, input2 = inputs
        return module.forward(input1, input2, not training)

    flax_module = MyFlaxModule()
    keras_layer = FlaxLayer(
        module=flax_module,
        method=my_flax_module_wrapper,
    )
    ```

    Args:
        module: An instance of `flax.linen.Module` or subclass.
        method: The method to call the model. This is generally a method in the
            `Module`. If not provided, the `__call__` method is used. `method`
            can also be a function not defined in the `Module`, in which case it
            must take the `Module` as the first argument. It is used for both
            `Module.init` and `Module.apply`. Details are documented in the
            `method` argument of [`flax.linen.Module.apply()`](
              https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply).
        variables: A `dict` containing all the variables of the module in the
            same format as what is returned by [`flax.linen.Module.init()`](
              https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.init).
            It should contain a "params" key and, if applicable, other keys for
            collections of variables for non-trainable state. This allows
            passing trained parameters and learned non-trainable state or
            controlling the initialization. If `None` is passed, the module's
            `init` function is called at build time to initialize the variables
            of the model.

    """

    def __init__(
        self,
        module: SupportsCall,
        method: SupportsCall | None = None,
        variables: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the FlaxLayer wrapper.

        Args:
            module: Wrapped Flax module or callable.
            method: Optional custom method taking ``(module, inputs, ...)``.
            variables: Optional initial variables dictionary.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.module = module
        self._variables = variables
        self.method = method

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Invoke the wrapped Flax module or ``self.method`` on ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments forwarded to the Flax callable.

        Returns:
            TensorLike: Wrapped output tensor or tuple of tensors.
        """
        from zero_keras import ops

        inputs = _to_tensor(inputs)
        if self.method is not None:
            out = self.method(self.module, inputs.data, **kwargs)
        else:
            out = self.module(inputs.data, **kwargs)
        if isinstance(out, tuple):
            return tuple(_wrap(ops.asarray(o)) for o in out)
        return _wrap(ops.asarray(out))


class JaxLayer(Layer):
    """Keras Layer that wraps a JAX model.

    This layer enables the use of JAX components within Keras when using JAX as
    the backend for Keras.

    ## Model function

    This layer accepts JAX models in the form of a function, `call_fn`, which
    must take the following arguments with these exact names:

    - `params`: trainable parameters of the model.
    - `state` (*optional*): non-trainable state of the model. Can be omitted if
        the model has no non-trainable state.
    - `rng` (*optional*): a `jax.random.PRNGKey` instance. Can be omitted if the
        model does not need RNGs, neither during training nor during inference.
    - `inputs`: inputs to the model, a JAX array or a `PyTree` of arrays.
    - `training` (*optional*): an argument specifying if we're in training mode
        or inference mode, `True` is passed in training mode. Can be omitted if
        the model behaves the same in training mode and inference mode.

    The `inputs` argument is mandatory. Inputs to the model must be provided via
    a single argument. If the JAX model takes multiple inputs as separate
    arguments, they must be combined into a single structure, for instance in a
    `tuple` or a `dict`.

    ## Model weights initialization

    The initialization of the `params` and `state` of the model can be handled
    by this layer, in which case the `init_fn` argument must be provided. This
    allows the model to be initialized dynamically with the right shape.
    Alternatively, and if the shape is known, the `params` argument and
    optionally the `state` argument can be used to create an already initialized
    model.

    The `init_fn` function, if provided, must take the following arguments with
    these exact names:

    - `rng`: a `jax.random.PRNGKey` instance.
    - `inputs`: a JAX array or a `PyTree` of arrays with placeholder values to
        provide the shape of the inputs.
    - `training` (*optional*): an argument specifying if we're in training mode
        or inference mode. `True` is always passed to `init_fn`. Can be omitted
        regardless of whether `call_fn` has a `training` argument.

    ## Models with non-trainable state

    For JAX models that have non-trainable state:

    - `call_fn` must have a `state` argument
    - `call_fn` must return a `tuple` containing the outputs of the model and
        the new non-trainable state of the model
    - `init_fn` must return a `tuple` containing the initial trainable params of
        the model and the initial non-trainable state of the model.

    This code shows a possible combination of `call_fn` and `init_fn` signatures
    for a model with non-trainable state. In this example, the model has a
    `training` argument and an `rng` argument in `call_fn`.

    ```python
    def stateful_call(params, state, rng, inputs, training):
        outputs = ...
        new_state = ...
        return outputs, new_state

    def stateful_init(rng, inputs):
        initial_params = ...
        initial_state = ...
        return initial_params, initial_state
    ```

    ## Models without non-trainable state

    For JAX models with no non-trainable state:

    - `call_fn` must not have a `state` argument
    - `call_fn` must return only the outputs of the model
    - `init_fn` must return only the initial trainable params of the model.

    This code shows a possible combination of `call_fn` and `init_fn` signatures
    for a model without non-trainable state. In this example, the model does not
    have a `training` argument and does not have an `rng` argument in `call_fn`.

    ```python
    def stateless_call(params, inputs):
        outputs = ...
        return outputs

    def stateless_init(rng, inputs):
        initial_params = ...
        return initial_params
    ```

    ## Conforming to the required signature

    If a model has a different signature than the one required by `JaxLayer`,
    one can easily write a wrapper method to adapt the arguments. This example
    shows a model that has multiple inputs as separate arguments, expects
    multiple RNGs in a `dict`, and has a `deterministic` argument with the
    opposite meaning of `training`. To conform, the inputs are combined in a
    single structure using a `tuple`, the RNG is split and used the populate the
    expected `dict`, and the Boolean flag is negated:

    ```python
    def my_model_fn(params, rngs, input1, input2, deterministic):
        ...
        if not deterministic:
            dropout_rng = rngs["dropout"]
            keep = jax.random.bernoulli(dropout_rng, dropout_rate, x.shape)
            x = jax.numpy.where(keep, x / dropout_rate, 0)
            ...
        ...
        return outputs

    def my_model_wrapper_fn(params, rng, inputs, training):
        input1, input2 = inputs
        rng1, rng2 = jax.random.split(rng)
        rngs = {"dropout": rng1, "preprocessing": rng2}
        deterministic = not training
        return my_model_fn(params, rngs, input1, input2, deterministic)

    keras_layer = JaxLayer(my_model_wrapper_fn, params=initial_params)
    ```

    ## Usage with Haiku modules

    `JaxLayer` enables the use of [Haiku](https://dm-haiku.readthedocs.io)
    components in the form of
    [`haiku.Module`](https://dm-haiku.readthedocs.io/en/latest/api.html#module).
    This is achieved by transforming the module per the Haiku pattern and then
    passing `module.apply` in the `call_fn` parameter and `module.init` in the
    `init_fn` parameter if needed.

    If the model has non-trainable state, it should be transformed with
    [`haiku.transform_with_state`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform_with_state).
    If the model has no non-trainable state, it should be transformed with
    [`haiku.transform`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform).
    Additionally, and optionally, if the module does not use RNGs in "apply", it
    can be transformed with
    [`haiku.without_apply_rng`](
      https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng).

    The following example shows how to create a `JaxLayer` from a Haiku module
    that uses random number generators via `hk.next_rng_key()` and takes a
    training positional argument:

    ```python
    class MyHaikuModule(hk.Module):
        def __call__(self, x, training):
            x = hk.Conv2D(32, (3, 3))(x)
            x = jax.nn.relu(x)
            x = hk.AvgPool((1, 2, 2, 1), (1, 2, 2, 1), "VALID")(x)
            x = hk.Flatten()(x)
            x = hk.Linear(200)(x)
            if training:
                x = hk.dropout(rng=hk.next_rng_key(), rate=0.3, x=x)
            x = jax.nn.relu(x)
            x = hk.Linear(10)(x)
            x = jax.nn.softmax(x)
            return x

    def my_haiku_module_fn(inputs, training):
        module = MyHaikuModule()
        return module(inputs, training)

    transformed_module = hk.transform(my_haiku_module_fn)

    keras_layer = JaxLayer(
        call_fn=transformed_module.apply,
        init_fn=transformed_module.init,
    )
    ```

    Args:
        call_fn: The function to call the model. See description above for the
            list of arguments it takes and the outputs it returns.
        init_fn: the function to call to initialize the model. See description
            above for the list of arguments it takes and the outputs it returns.
            If `None`, then `params` and/or `state` must be provided.
      params: A `PyTree` containing all the model trainable parameters. This
            allows passing trained parameters or controlling the initialization.
            If both `params` and `state` are `None`, `init_fn` is called at
            build time to initialize the trainable parameters of the model.
      state: A `PyTree` containing all the model non-trainable state. This
            allows passing learned state or controlling the initialization. If
            both `params` and `state` are `None`, and `call_fn` takes a `state`
            argument, then `init_fn` is called at build time to initialize the
            non-trainable state of the model.
      seed: Seed for random number generator. Optional.
      dtype: The dtype of the layer's computations and weights. Can also be a
            `keras.DTypePolicy`. Optional. Defaults to the default policy.

    """

    def __init__(
        self,
        call_fn: SupportsCall | None,
        init_fn: SupportsCall | None = None,
        params: TensorLike | None = None,
        state: TensorLike | None = None,
        seed: int | None = None,
        native_serialization_platforms: Sequence[str] | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the JaxLayer wrapper.

        Args:
            call_fn: JAX forward function accepting ``(params, inputs, ...)``.
            init_fn: Optional JAX initialization function.
            params: Trainable parameter PyTree.
            state: Optional non-trainable state PyTree.
            seed: Optional random seed.
            native_serialization_platforms: Optional native serialization platform list.
            **kwargs: Base layer keyword arguments.
        """
        _ = init_fn, seed, native_serialization_platforms
        super().__init__(**kwargs)
        self.call_fn = call_fn
        self.params = params
        self.state = state

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Evaluate the wrapped JAX function on ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments passed to ``call_fn``.

        Returns:
            TensorLike: Wrapped output tensor.
        """
        from zero_keras import ops

        inputs = _to_tensor(inputs)
        if callable(self.call_fn):
            out = self.call_fn(self.params, inputs.data, **kwargs)
            return _wrap(ops.asarray(out))
        return _wrap(inputs)


class TFSMLayer(Layer):
    """Reload a Keras model/layer that was saved via SavedModel / ExportArchive.

    Arguments:
        filepath: `str` or `pathlib.Path` object. The path to the SavedModel.
        call_endpoint: Name of the endpoint to use as the `call()` method
            of the reloaded layer. If the SavedModel was created
            via `model.export()`,
            then the default endpoint name is `'serve'`. In other cases
            it may be named `'serving_default'`.

    Example:
    ```python
    model.export("path/to/artifact")
    reloaded_layer = TFSMLayer("path/to/artifact")
    outputs = reloaded_layer(inputs)
    ```

    The reloaded object can be used like a regular Keras layer, and supports
    training/fine-tuning of its trainable weights. Note that the reloaded
    object retains none of the internal structure or custom methods of the
    original object -- it's a brand new layer created around the saved
    function.

    **Limitations:**

    * Only call endpoints with a single `inputs` tensor argument
    (which may optionally be a dict/tuple/list of tensors) are supported.
    For endpoints with multiple separate input tensor arguments, consider
    subclassing `TFSMLayer` and implementing a `call()` method with a
    custom signature.
    * If you need training-time behavior to differ from inference-time behavior
    (i.e. if you need the reloaded object to support a `training=True` argument
    in `__call__()`), make sure that the training-time call function is
    saved as a standalone endpoint in the artifact, and provide its name
    to the `TFSMLayer` via the `call_training_endpoint` argument.

    """

    def __init__(
        self,
        filepath: str | Path,
        call_endpoint: str = "serve",
        call_training_endpoint: str | None = None,
        trainable: bool = True,
        name: str | None = None,
        dtype: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the TFSMLayer from a SavedModel or Keras archive path.

        Args:
            filepath: Filesystem path to the SavedModel or ``.keras`` archive.
            call_endpoint: Name of the inference endpoint. Defaults to ``"serve"``.
            call_training_endpoint: Optional name of the training endpoint.
            trainable: Whether reloaded variables should be trainable.
            name: Optional layer name.
            dtype: Optional computation dtype.
            **kwargs: Base layer keyword arguments.
        """
        if name is not None:
            kwargs["name"] = name
        if dtype is not None:
            kwargs["dtype"] = dtype
        kwargs["trainable"] = trainable
        super().__init__(**kwargs)
        self.filepath = filepath
        self.call_endpoint = call_endpoint
        self.call_training_endpoint = call_training_endpoint
        self.module: TensorLike | None = None

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Invoke the loaded SavedModel endpoint or ``self.module`` on ``inputs``.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments forwarded to the endpoint.

        Returns:
            TensorLike: Output tensor or tuple of tensors from the endpoint.
        """
        from zero_keras import ops

        inputs = _to_tensor(inputs)
        if self.module is None and os.path.isfile(str(self.filepath)):
            from zero_keras.models import load_model

            self.module = load_model(str(self.filepath))
        if self.module is not None:
            out = self.module(inputs.data, **kwargs)
            if isinstance(out, tuple):
                return tuple(_wrap(ops.asarray(o)) for o in out)
            return _wrap(ops.asarray(out))
        return _wrap(inputs)


class TorchModuleWrapper(Layer):
    """Torch module wrapper layer.

    `TorchModuleWrapper` is a wrapper class that can turn any
    `torch.nn.Module` into a Keras layer, in particular by making its
    parameters trackable by Keras.

    `TorchModuleWrapper` is only compatible with the PyTorch backend and
    cannot be used with the TensorFlow or JAX backends.

    Args:
        module: `torch.nn.Module` instance. If it's a `LazyModule`
            instance, then its parameters must be initialized before
            passing the instance to `TorchModuleWrapper` (e.g. by calling
            it once).
        output_shape :The shape of the output of this layer. It helps Keras
            perform automatic shape inference.
        name: The name of the layer (string).

    Example:
    Here's an example of how the `TorchModuleWrapper` can be used with vanilla
    PyTorch modules.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import zero_keras as keras
    from zero_keras.layers import TorchModuleWrapper

    class Classifier(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Wrap `torch.nn.Module`s with `TorchModuleWrapper`
            # if they contain parameters
            self.conv1 = TorchModuleWrapper(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
            )
            self.conv2 = TorchModuleWrapper(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
            )
            self.pool = nn.MaxPool2d(kernel_size=(2, 2))
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=0.5)
            self.fc = TorchModuleWrapper(nn.Linear(1600, 10))

        def call(self, inputs):
            x = F.relu(self.conv1(inputs))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.fc(x)
            return F.softmax(x, dim=1)


    model = Classifier()
    model.build((1, 28, 28))
    print("Output shape:", model(torch.ones(1, 1, 28, 28).to("cuda")).shape)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(train_loader, epochs=5)
    ```

    """

    parameters: list[TensorLike] = []

    def __init__(
        self,
        module: SupportsCall,
        name: str | None = None,
        output_shape: ShapeTuple | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the TorchModuleWrapper layer.

        Args:
            module: Wrapped PyTorch module or callable.
            name: Optional layer name.
            output_shape: Optional expected output shape tuple.
            **kwargs: Base layer keyword arguments.
        """
        _ = output_shape
        if name is not None:
            kwargs["name"] = name
        super().__init__(**kwargs)
        self.module = module

    def call(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Forward ``inputs`` through the wrapped PyTorch module.

        Args:
            inputs: Input tensor.
            **kwargs: Additional keyword arguments forwarded to ``self.module``.

        Returns:
            TensorLike: Wrapped output tensor or tuple of tensors.
        """
        from zero_keras import ops

        inputs = _to_tensor(inputs)
        out = self.module(inputs.data, **kwargs)
        if isinstance(out, tuple):
            return tuple(_wrap(ops.asarray(o)) for o in out)
        return _wrap(ops.asarray(out))


class TimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every temporal slice of an input.

    Every input should be at least 3D, and the dimension of index one of the
    first input will be considered to be the temporal dimension.

    Consider a batch of 32 video samples, where each sample is a 128x128 RGB
    image with `channels_last` data format, across 10 timesteps.
    The batch input shape is `(32, 10, 128, 128, 3)`.

    You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
    of the 10 timesteps, independently:

    >>> inputs = layers.Input(shape=(10, 128, 128, 3), batch_size=32)
    >>> conv_2d_layer = layers.Conv2D(64, (3, 3))
    >>> outputs = layers.TimeDistributed(conv_2d_layer)(inputs)
    >>> outputs.shape
    (32, 10, 126, 126, 64)

    Because `TimeDistributed` applies the same instance of `Conv2D` to each of
    the timestamps, the same set of weights are used at each timestamp.

    Args:
        layer: a `keras.layers.Layer` instance.

    Call arguments:
        inputs: Input tensor of shape (batch, time, ...) or nested tensors,
            and each of which has shape (batch, time, ...).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            wrapped layer (only if the layer supports this argument).
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked. This argument is passed to the
            wrapped layer (only if the layer supports this argument).

    """

    def __init__(self, layer: Layer, **kwargs: TensorLike) -> None:
        """Initialize the TimeDistributed wrapper around ``layer``.

        Args:
            layer: Child layer to apply across each temporal slice.
            **kwargs: Base wrapper keyword arguments.
        """
        super().__init__(layer=layer, **kwargs)
        self.layer = layer

    def build(self, input_shape: ShapeTuple) -> None:
        """Build the wrapped child layer using the per-timestep slice shape.

        Args:
            input_shape: Shape tuple ``(batch, time, ...)`` of the input tensor.
        """
        if self.built:
            return
        self.layer.build((input_shape[0],) + input_shape[2:])
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        training: bool | None = None,
        mask: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Apply ``self.layer`` independently to every timestep slice of ``inputs``.

        Args:
            inputs: Input tensor of shape ``(batch, time, ...)``.
            training: Whether the layer is executing in training mode.
            mask: Optional boolean timestep mask.
            **kwargs: Additional keyword arguments forwarded to ``self.layer``.

        Returns:
            TensorLike: Output tensor of shape ``(batch, time, *output_dims)``.
        """
        inputs = _to_tensor(inputs)
        shape = inputs.shape
        # (batch, time, ...) -> (batch*time, ...)
        new_shape = (shape[0] * shape[1],) + shape[2:]
        reshaped = ops.reshape(inputs, new_shape)

        out = _to_tensor(self.layer(reshaped, training=training, **kwargs))

        out_shape = (shape[0], shape[1]) + out.shape[1:]
        return _wrap(ops.reshape(out, out_shape))


class Pipeline(Layer):
    """Applies a series of layers to an input.

    This class is useful to build a preprocessing pipeline,
    in particular an image data augmentation pipeline.
    Compared to a `Sequential` model, `Pipeline` features
    a few important differences:

    - It's not a `Model`, just a plain layer.
    - When the layers in the pipeline are compatible
        with `tf.data`, the pipeline will also
        remain `tf.data` compatible. That is to say,
        the pipeline will not attempt to convert
        its inputs to backend-native tensors
        when in a tf.data context (unlike a `Sequential`
        model).

    Example:
    ```python
    from zero_keras import layers
    preprocessing_pipeline = layers.Pipeline([
        layers.AutoContrast(),
        layers.RandomZoom(0.2),
        layers.RandomRotation(0.2),
    ])

    # `ds` is a tf.data.Dataset
    preprocessed_ds = ds.map(
        preprocessing_pipeline,
        num_parallel_calls=4,
    )
    ```

    """

    layers: list[Layer] = []

    def __init__(
        self,
        layers: Sequence[Layer],
        name: str | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Pipeline with a sequence of ``layers``.

        Args:
            layers: Ordered sequence of layers to apply sequentially.
            name: Optional layer name.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self._layers = list(layers)
        self.layers = self._layers

    def call(
        self, inputs: TensorLike, training: bool | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Pass ``inputs`` sequentially through every layer in ``self._layers``.

        Args:
            inputs: Initial input tensor.
            training: Whether the pipeline is executing in training mode.
            **kwargs: Additional keyword arguments forwarded to each layer.

        Returns:
            TensorLike: Output of the final layer in the pipeline.
        """
        for layer in self._layers:
            inputs = layer(inputs, training=training, **kwargs)
        return inputs


class ReversibleEmbedding(Embedding):
    """Embedding layer capable of projecting hidden states back to vocabulary logits."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        tie_weights: bool = True,
        embeddings_initializer: InitializerSpec = "uniform",
        embeddings_regularizer: RegularizerSpec = None,
        embeddings_constraint: ConstraintSpec = None,
        mask_zero: bool = False,
        reverse_dtype: str | None = None,
        logit_soft_cap: float | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize ReversibleEmbedding.

        Args:
            input_dim: Size of the vocabulary.
            output_dim: Dimension of the dense embedding.
            tie_weights: Whether to tie input embedding and output projection weights.
            embeddings_initializer: Initializer for the embeddings matrix.
            embeddings_regularizer: Regularizer for the embeddings matrix.
            embeddings_constraint: Constraint for the embeddings matrix.
            mask_zero: Whether input value 0 is a special padding value.
            reverse_dtype: Optional dtype for the reverse projection computation.
            logit_soft_cap: Optional soft cap value for output logits.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            **kwargs,
        )
        self.tie_weights = bool(tie_weights)
        self.reverse_dtype = reverse_dtype
        self.logit_soft_cap = logit_soft_cap

    def call(self, inputs: TensorLike, reverse: bool = False, **kwargs: TensorLike) -> TensorLike:
        """Project token IDs to embeddings or hidden states back to logits when ``reverse=True``.

        Args:
            inputs: Token indices or hidden state tensor.
            reverse: If True, project embeddings back to vocabulary logits.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Embedded tensor or vocabulary logits.
        """
        if not reverse:
            return super().call(inputs, **kwargs)
        from zero_keras import ops

        t = _to_tensor(inputs)
        if not self.built:
            self.build(t.shape)
        emb = _to_tensor(self.embeddings)
        logits = ops.matmul(t, ops.transpose(emb))
        if self.logit_soft_cap is not None:
            cap = float(self.logit_soft_cap)
            logits = ops.multiply(ops.tanh(ops.divide(logits, cap)), cap)
        return _wrap(logits)

    def get_config(self) -> ConfigDict:
        """Return serializable layer configuration.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "tie_weights": self.tie_weights,
                "reverse_dtype": self.reverse_dtype,
                "logit_soft_cap": self.logit_soft_cap,
            }
        )
        return cfg


_MISC_LAYER_REGISTRY: dict[str, type[Layer]] = {
    "EinsumDense": EinsumDense,
    "Embedding": Embedding,
    "FlaxLayer": FlaxLayer,
    "JaxLayer": JaxLayer,
    "Layer": Layer,
    "Pipeline": Pipeline,
    "ReversibleEmbedding": ReversibleEmbedding,
    "TFSMLayer": TFSMLayer,
    "TimeDistributed": TimeDistributed,
    "TorchModuleWrapper": TorchModuleWrapper,
    "Wrapper": Wrapper,
}


def serialize(layer: TensorLike) -> TensorLike:
    """Serialize a layer instance into a configuration dictionary.

    Args:
        layer: Layer instance or string identifier.

    Returns:
        TensorLike: Serialized layer dictionary, string, or None.
    """
    if layer is None:
        return None
    if isinstance(layer, str):
        return layer
    return {
        "class_name": type(layer).__name__,
        "config": layer.get_config() if isinstance(layer, Layer) else {},
    }


def deserialize(
    config: TensorLike, custom_objects: ConfigDict | TensorLike | None = None
) -> TensorLike:
    """Deserialize a layer from its configuration dictionary or string identifier.

    Args:
        config: Serialized layer configuration dictionary or string name.
        custom_objects: Optional mapping of custom layer classes.

    Returns:
        TensorLike: Reconstructed layer instance or original config.
    """
    if config is None:
        return None
    from zero_keras.saving.registry import lookup_class

    if isinstance(config, str):
        cls = _MISC_LAYER_REGISTRY.get(config) or lookup_class("layers", config)
        if cls is not None:
            return cls()
        return config
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf = config.get("config", {})

        for k, v in list(conf.items()):
            if isinstance(v, dict) and "class_name" in v and "config" in v:
                conf[k] = deserialize(v, custom_objects=custom_objects)

        cls = (
            (_MISC_LAYER_REGISTRY.get(class_name) or lookup_class("layers", class_name))
            if isinstance(class_name, str)
            else None
        )
        if cls is not None:
            return cls(**conf)

        if isinstance(custom_objects, dict) and class_name in custom_objects:
            return custom_objects[class_name](**conf)

    return config

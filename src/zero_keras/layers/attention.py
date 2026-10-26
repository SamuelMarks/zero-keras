"""Attention layers (`MultiHeadAttention`, `Attention`, `AdditiveAttention`, `GroupQueryAttention`) for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer
from zero_keras.types import (
    ConstraintSpec,
    InitializerSpec,
    RegularizerSpec,
    ShapeTuple,
    TensorLike,
)

from .misc import EinsumDense


class AdditiveAttention(Layer):
    """Additive attention layer, a.k.a. Bahdanau-style attention.

    Inputs are a list with 2 or 3 elements:
    1. A `query` tensor of shape `(batch_size, Tq, dim)`.
    2. A `value` tensor of shape `(batch_size, Tv, dim)`.
    3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none
        supplied, `value` will be used as `key`.

    The calculation follows the steps:
    1. Calculate attention scores using `query` and `key` with shape
        `(batch_size, Tq, Tv)` as a non-linear sum
        `scores = reduce_sum(tanh(query + key), axis=-1)`.
    2. Use scores to calculate a softmax distribution with shape
        `(batch_size, Tq, Tv)`.
    3. Use the softmax distribution to create a linear combination of `value`
        with shape `(batch_size, Tq, dim)`.

    Args:
        use_scale: If `True`, will create a scalar variable to scale the
            attention scores.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            attention scores. Defaults to `0.0`.

    Call arguments:
        inputs: List of the following tensors:
            - `query`: Query tensor of shape `(batch_size, Tq, dim)`.
            - `value`: Value tensor of shape `(batch_size, Tv, dim)`.
            - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If
                not given, will use `value` for both `key` and `value`, which is
                the most common case.
        mask: List of the following tensors:
            - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`.
                If given, the output will be zero at the positions where
                `mask==False`.
            - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`.
                If given, will apply the mask such that values at positions
                 where `mask==False` do not contribute to the result.
        return_attention_scores: bool, it `True`, returns the attention scores
            (after masking and softmax) as an additional output argument.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
        use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds
            a mask such that position `i` cannot attend to positions `j > i`.
            This prevents the flow of information from the future towards the
            past. Defaults to `False`.

    Output:
        Attention outputs of shape `(batch_size, Tq, dim)`.
        (Optional) Attention scores after masking and softmax with shape
            `(batch_size, Tq, Tv)`.

    """

    def __init__(
        self,
        use_scale: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the AdditiveAttention layer.

        Args:
            use_scale: Whether to create a learnable scalar scale weight.
            dropout: Dropout rate applied to attention scores.
            causal: Whether to apply a causal mask by default.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.head_dim = kwargs.get("head_dim")
        self.use_scale = bool(use_scale)
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.scale: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create the optional ``scale`` weight.

        Args:
            input_shape: Shape tuple or list of shape tuples for ``[query, value, ...]``.
        """
        if self.built:
            return
        self.scale = (
            self.add_weight(shape=(), initializer="ones", name="scale") if self.use_scale else None
        )
        self.built = True

    def call(
        self,
        inputs: Sequence[TensorLike],
        mask: TensorLike | None = None,
        training: bool | None = None,
        return_attention_scores: bool = False,
        use_causal_mask: bool = False,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Compute Bahdanau-style additive attention over ``inputs``.

        Args:
            inputs: Sequence of ``[query, value]`` or ``[query, value, key]`` tensors.
            mask: Optional list of query and value masks.
            training: Whether the layer is executing in training mode.
            return_attention_scores: Whether to also return the softmax attention weights.
            use_causal_mask: Whether to apply a causal attention mask.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Attention output tensor, or ``(output, scores)`` tuple.
        """
        query = _to_tensor(inputs[0])
        value = _to_tensor(inputs[1])
        key = _to_tensor(inputs[2]) if len(inputs) > 2 else value

        q_exp = ops.expand_dims(query, 2)
        k_exp = ops.expand_dims(key, 1)

        from zero_keras.activations import softmax, tanh

        scores = _to_tensor(tanh(ops.add(q_exp, k_exp)))
        if self.use_scale:
            scores = ops.multiply(scores, _to_tensor(self.scale))
        scores = ops.sum(scores, axis=-1)

        if use_causal_mask or self.causal:
            seq_len_q = query.shape[-2]
            seq_len_k = key.shape[-2]
            causal_mask = ops.tril(ops.ones((seq_len_q, seq_len_k)))
            causal_mask = ops.cast(causal_mask, scores.dtype)
            scores = ops.where(causal_mask > 0.5, scores, _to_tensor(-1e9))

        if isinstance(mask, (list, tuple)) and len(mask) >= 2 and mask[1] is not None:
            v_mask = ops.cast(_to_tensor(mask[1]), scores.dtype)
            v_mask = ops.expand_dims(v_mask, axis=1)
            scores = ops.where(v_mask > 0.5, scores, _to_tensor(-1e9))

        scores = _to_tensor(softmax(scores))
        if training and self.dropout > 0.0:
            scores = _to_tensor(ops.dropout(scores, self.dropout))

        out = ops.matmul(scores, value)
        if isinstance(mask, (list, tuple)) and len(mask) >= 1 and mask[0] is not None:
            q_mask = ops.cast(_to_tensor(mask[0]), out.dtype)
            q_mask = ops.expand_dims(q_mask, axis=-1)
            out = ops.multiply(out, q_mask)

        if return_attention_scores:
            return _wrap(out), _wrap(scores)
        return _wrap(out)


class Attention(Layer):
    """Dot-product attention layer, a.k.a. Luong-style attention.

    Inputs are a list with 2 or 3 elements:
    1. A `query` tensor of shape `(batch_size, Tq, dim)`.
    2. A `value` tensor of shape `(batch_size, Tv, dim)`.
    3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none
        supplied, `value` will be used as a `key`.

    The calculation follows the steps:
    1. Calculate attention scores using `query` and `key` with shape
        `(batch_size, Tq, Tv)`.
    2. Use scores to calculate a softmax distribution with shape
        `(batch_size, Tq, Tv)`.
    3. Use the softmax distribution to create a linear combination of `value`
        with shape `(batch_size, Tq, dim)`.

    Args:
        use_scale: If `True`, will create a scalar variable to scale the
            attention scores.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            attention scores. Defaults to `0.0`.
        seed: A Python integer to use as random seed in case of `dropout`.
        score_mode: Function to use to compute attention scores, one of
            `{"dot", "concat"}`. `"dot"` refers to the dot product between the
            query and key vectors. `"concat"` refers to the hyperbolic tangent
            of the concatenation of the `query` and `key` vectors.

    Call arguments:
        inputs: List of the following tensors:
            - `query`: Query tensor of shape `(batch_size, Tq, dim)`.
            - `value`: Value tensor of shape `(batch_size, Tv, dim)`.
            - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If
                not given, will use `value` for both `key` and `value`, which is
                the most common case.
        mask: List of the following tensors:
            - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`.
                If given, the output will be zero at the positions where
                `mask==False`.
            - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`.
                If given, will apply the mask such that values at positions
                 where `mask==False` do not contribute to the result.
        return_attention_scores: bool, it `True`, returns the attention scores
            (after masking and softmax) as an additional output argument.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
        use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds
            a mask such that position `i` cannot attend to positions `j > i`.
            This prevents the flow of information from the future towards the
            past. Defaults to `False`.

    Output:
        Attention outputs of shape `(batch_size, Tq, dim)`.
        (Optional) Attention scores after masking and softmax with shape
            `(batch_size, Tq, Tv)`.

    """

    def __init__(
        self,
        use_scale: bool = False,
        score_mode: str = "dot",
        dropout: float = 0.0,
        seed: int | None = None,
        causal: bool = False,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Luong-style dot-product Attention layer.

        Args:
            use_scale: Whether to create a learnable scalar scale weight.
            score_mode: Attention scoring function (``"dot"`` or ``"concat"``).
            dropout: Dropout rate applied to attention scores.
            seed: Optional random seed for dropout.
            causal: Whether to apply a causal mask by default.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        if score_mode not in ("dot", "concat"):
            raise ValueError(f"Invalid score_mode: {score_mode}. Expected 'dot' or 'concat'.")
        self.head_dim = kwargs.get("head_dim")
        self.use_scale = bool(use_scale)
        self.score_mode = str(score_mode)
        self.dropout = float(dropout)
        self.seed = seed
        self.causal = bool(causal)
        self.scale: TensorLike | None = None
        self.concat_score_weight: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Create the optional ``scale`` and ``concat_score_weight`` weights.

        Args:
            input_shape: Shape tuple or list of shape tuples for ``[query, value, ...]``.
        """
        if self.built:
            return
        if self.use_scale:
            self.scale = self.add_weight(shape=(), initializer="ones", name="scale")
        if self.score_mode == "concat":
            self.concat_score_weight = self.add_weight(
                shape=(), initializer="ones", name="concat_score_weight"
            )
        self.built = True

    def call(
        self,
        inputs: Sequence[TensorLike],
        mask: TensorLike | None = None,
        training: bool | None = None,
        return_attention_scores: bool = False,
        use_causal_mask: bool = False,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Compute Luong-style dot-product or concat attention over ``inputs``.

        Args:
            inputs: Sequence of ``[query, value]`` or ``[query, value, key]`` tensors.
            mask: Optional list of query and value masks.
            training: Whether the layer is executing in training mode.
            return_attention_scores: Whether to also return the softmax attention weights.
            use_causal_mask: Whether to apply a causal attention mask.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Attention output tensor, or ``(output, scores)`` tuple.
        """
        query = _to_tensor(inputs[0])
        value = _to_tensor(inputs[1])
        key = _to_tensor(inputs[2]) if len(inputs) > 2 else value

        from zero_keras.activations import softmax, tanh

        if self.score_mode == "concat":
            q_exp = ops.expand_dims(query, 2)
            k_exp = ops.expand_dims(key, 1)
            energies = _to_tensor(tanh(ops.add(q_exp, k_exp)))
            if self.use_scale and self.scale is not None:
                energies = ops.multiply(energies, _to_tensor(self.scale))
            energies = ops.multiply(energies, _to_tensor(self.concat_score_weight))
            scores = ops.sum(energies, axis=-1)
        else:
            k_t = ops.permute(key, (0, 2, 1))
            scores = ops.matmul(query, k_t)
            if self.use_scale:
                scores = ops.multiply(scores, _to_tensor(self.scale))

        if use_causal_mask or self.causal:
            seq_len_q = query.shape[-2]
            seq_len_k = key.shape[-2]
            causal_mask = ops.tril(ops.ones((seq_len_q, seq_len_k)))
            causal_mask = ops.cast(causal_mask, scores.dtype)
            scores = ops.where(causal_mask > 0.5, scores, _to_tensor(-1e9))

        if isinstance(mask, (list, tuple)) and len(mask) >= 2 and mask[1] is not None:
            v_mask = ops.cast(_to_tensor(mask[1]), scores.dtype)
            v_mask = ops.expand_dims(v_mask, axis=1)
            scores = ops.where(v_mask > 0.5, scores, _to_tensor(-1e9))

        scores = _to_tensor(softmax(scores))
        if training and self.dropout > 0.0:
            scores = _to_tensor(ops.dropout(scores, self.dropout, seed=self.seed))

        out = ops.matmul(scores, value)
        if isinstance(mask, (list, tuple)) and len(mask) >= 1 and mask[0] is not None:
            q_mask = ops.cast(_to_tensor(mask[0]), out.dtype)
            q_mask = ops.expand_dims(q_mask, axis=-1)
            out = ops.multiply(out, q_mask)

        if return_attention_scores:
            return _wrap(out), _wrap(scores)
        return _wrap(out)


class MultiHeadAttention(Layer):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need"
    [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as `value_dim` can take
    a linear projection and return.

    Args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        value_dim: Size of each attention head for value.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        output_shape: The expected shape of an output tensor, besides the batch
            and sequence dims. If not specified, projects back to the query
            feature dim (the query input's last dimension).
        attention_axes: axes over which the attention is applied. `None` means
            attention over all axes, but batch, heads, and features.
        flash_attention: If `None`, the layer attempts to use flash
            attention for faster and more memory-efficient attention
            computations when possible. This behavior can be configured using
            `keras.config.enable_flash_attention()` or
            `keras.config.disable_flash_attention()`.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.
        seed: Optional integer to seed the dropout layer.

    Call arguments:
        query: Query tensor of shape `(B, T, dim)`, where `B` is the batch size,
            `T` is the target sequence length, and dim is the feature dimension.
        value: Value tensor of shape `(B, S, dim)`, where `B` is the batch size,
            `S` is the source sequence length, and dim is the feature dimension.
        key: Optional key tensor of shape `(B, S, dim)`. If not given, will
            use `value` for both `key` and `value`, which is the most common
            case.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output should
            be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Will go with either using the training mode of the parent
            layer/model, or `False` (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: The result of the computation, of shape `(B, T, E)`,
            where `T` is for target sequence shapes and `E` is the query input
            last dimension if `output_shape` is `None`. Otherwise, the
            multi-head outputs are projected to the shape specified by
            `output_shape`.
        attention_scores: (Optional) multi-head attention coefficients over
            attention axes.

    """

    attention_axes: int | Sequence[int] | None = None
    dropout: float = 0.0
    key_dense: TensorLike | None = None
    key_dim: int = 0
    num_heads: int = 0
    output_dense: TensorLike | None = None
    query_dense: TensorLike | None = None
    use_bias: bool = True
    value_dense: TensorLike | None = None
    value_dim: int | None = None

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int | None = None,
        dropout: float = 0.0,
        use_bias: bool = True,
        output_shape: ShapeTuple | int | None = None,
        attention_axes: int | Sequence[int] | None = None,
        sliding_window: int | None = None,
        flash_attention: bool | None = None,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        use_gate: bool = False,
        seed: int | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MultiHeadAttention layer.

        Args:
            num_heads: Number of attention heads.
            key_dim: Size of each attention head for query and key.
            value_dim: Size of each attention head for value; defaults to ``key_dim``.
            dropout: Dropout probability on attention weights.
            use_bias: Whether projection dense layers include bias vectors.
            output_shape: Expected non-batch output feature shape.
            attention_axes: Axes over which attention is applied.
            sliding_window: Optional sliding attention window size.
            flash_attention: Whether to enable flash attention when supported.
            kernel_initializer: Initializer for projection kernels.
            bias_initializer: Initializer for projection biases.
            kernel_regularizer: Regularizer for projection kernels.
            bias_regularizer: Regularizer for projection biases.
            activity_regularizer: Regularizer for layer output activity.
            kernel_constraint: Constraint for projection kernels.
            bias_constraint: Constraint for projection biases.
            use_gate: Whether to apply gating on attention outputs.
            seed: Optional random seed for dropout.
            **kwargs: Base layer keyword arguments.
        """
        _ = sliding_window, use_gate
        super().__init__(**kwargs)
        self.head_dim = kwargs.get("head_dim")
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim) if value_dim is not None else int(key_dim)
        self.dropout = float(dropout)
        self.use_bias = bool(use_bias)
        self.output_shape_tuple = output_shape
        self.attention_axes = attention_axes
        self.flash_attention = flash_attention
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.seed = seed
        self._query_dense: TensorLike | None = None
        self._key_dense: TensorLike | None = None
        self._value_dense: TensorLike | None = None
        self._output_dense: TensorLike | None = None

    def build(self, input_shape: ShapeTuple) -> None:
        """Build the query, key, value, and output ``EinsumDense`` projection layers.

        Args:
            input_shape: Shape tuple of the query tensor or tuple of input shape tuples.
        """
        if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (tuple, list))
        ):
            q_shape = tuple(input_shape[0])
            v_shape = (
                tuple(input_shape[1])
                if len(input_shape) > 1 and isinstance(input_shape[1], (tuple, list))
                else q_shape
            )
            k_shape = (
                tuple(input_shape[2])
                if len(input_shape) > 2 and isinstance(input_shape[2], (tuple, list))
                else v_shape
            )
        else:
            q_shape = input_shape
            v_shape = input_shape
            k_shape = input_shape

        out_dim = (
            q_shape[-1]
            if self.output_shape_tuple is None
            else (
                self.output_shape_tuple[-1]
                if isinstance(self.output_shape_tuple, tuple)
                else self.output_shape_tuple
            )
        )

        # EinsumDense strings
        self._query_dense = EinsumDense(
            equation="...X,XHD->...HD",
            output_shape=(self.num_heads, self.key_dim),
            bias_axes="HD" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="query",
        )
        self._key_dense = EinsumDense(
            equation="...X,XHD->...HD",
            output_shape=(self.num_heads, self.key_dim),
            bias_axes="HD" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="key",
        )
        self._value_dense = EinsumDense(
            equation="...X,XHD->...HD",
            output_shape=(self.num_heads, self.value_dim),
            bias_axes="HD" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="value",
        )
        self._output_dense = EinsumDense(
            equation="...HD,HDO->...O",
            output_shape=(out_dim,),
            bias_axes="O" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="attention_output",
        )
        self._query_dense.build(q_shape)
        self._key_dense.build(k_shape)
        self._value_dense.build(v_shape)

        # Output dense input shape is ...HD
        out_shape_input = tuple(q_shape[:-1]) + (self.num_heads, self.value_dim)
        self._output_dense.build(out_shape_input)
        self.query_dense = self._query_dense
        self.key_dense = self._key_dense
        self.value_dense = self._value_dense
        self.output_dense = self._output_dense
        self.built = True

    def call(
        self,
        query: TensorLike,
        value: TensorLike,
        key: TensorLike | None = None,
        attention_mask: TensorLike | None = None,
        return_attention_scores: bool = False,
        training: bool | None = False,
        use_causal_mask: bool = False,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Compute scaled dot-product multi-head attention.

        Args:
            query: Query tensor of shape ``(B, T, dim)``.
            value: Value tensor of shape ``(B, S, dim)``.
            key: Optional key tensor of shape ``(B, S, dim)``; defaults to ``value``.
            attention_mask: Optional additive or boolean attention mask.
            return_attention_scores: Whether to also return attention weight scores.
            training: Whether the layer is executing in training mode.
            use_causal_mask: Whether to apply a lower-triangular causal mask.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Attention output tensor, or ``(attention_output, attention_scores)``.
        """
        query = _to_tensor(query)
        value = _to_tensor(value)
        key = value if key is None else _to_tensor(key)

        query = _to_tensor(self._query_dense(query))
        key = _to_tensor(self._key_dense(key))
        value = _to_tensor(self._value_dense(value))

        import math
        import string

        from zero_keras import ops

        letters = string.ascii_lowercase
        b_dims = len(query.shape) - 3
        b_str = letters[:b_dims]
        nxt = letters[b_dims:]
        t_str, h_str, d_str, s_str = nxt[0], nxt[1], nxt[2], nxt[3]

        eq_scores = f"{b_str}{t_str}{h_str}{d_str},{b_str}{s_str}{h_str}{d_str}->{b_str}{h_str}{t_str}{s_str}"
        attention_scores = ops.einsum(eq_scores, query, key)
        attention_scores = ops.multiply(
            attention_scores, _to_tensor(1.0 / math.sqrt(float(self.key_dim)))
        )

        if use_causal_mask:
            seq_len_q = query.shape[-3]
            seq_len_k = key.shape[-3]
            causal_mask = ops.tril(ops.ones((seq_len_q, seq_len_k)))
            causal_mask = ops.cast(causal_mask, attention_scores.dtype)
            attention_scores = ops.where(causal_mask > 0.5, attention_scores, _to_tensor(-1e9))

        if attention_mask is not None:
            attention_mask = _to_tensor(attention_mask)
            attention_scores = ops.add(attention_scores, attention_mask)

        from zero_keras.ops import softmax

        attention_scores = softmax(attention_scores, axis=-1)
        if training and self.dropout > 0.0:
            attention_scores = ops.dropout(attention_scores, self.dropout, seed=self.seed)

        eq_out = f"{b_str}{h_str}{t_str}{s_str},{b_str}{s_str}{h_str}{d_str}->{b_str}{t_str}{h_str}{d_str}"
        attention_output = ops.einsum(eq_out, attention_scores, value)

        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return _wrap(attention_output), _wrap(attention_scores)
        return _wrap(attention_output)


class GroupQueryAttention(MultiHeadAttention):
    """Grouped Query Attention (GQA) layer sharing key/value heads across query groups."""

    def __init__(
        self,
        head_dim: int,
        num_query_heads: int,
        num_key_value_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        sliding_window: int | None = None,
        flash_attention: bool | None = None,
        kernel_initializer: InitializerSpec = "glorot_uniform",
        bias_initializer: InitializerSpec = "zeros",
        kernel_regularizer: RegularizerSpec = None,
        bias_regularizer: RegularizerSpec = None,
        activity_regularizer: RegularizerSpec = None,
        kernel_constraint: ConstraintSpec = None,
        bias_constraint: ConstraintSpec = None,
        use_gate: bool = False,
        seed: int | None = None,
        num_heads: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
        output_shape: ShapeTuple | int | None = None,
        attention_axes: int | Sequence[int] | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the GroupQueryAttention layer.

        Args:
            head_dim: Size of each attention head for query and key.
            num_query_heads: Total number of query attention heads.
            num_key_value_heads: Number of key/value attention heads; must divide ``num_query_heads``.
            dropout: Dropout probability on attention weights.
            use_bias: Whether projection dense layers include bias vectors.
            sliding_window: Optional sliding attention window size.
            flash_attention: Whether to enable flash attention when supported.
            kernel_initializer: Initializer for projection kernels.
            bias_initializer: Initializer for projection biases.
            kernel_regularizer: Regularizer for projection kernels.
            bias_regularizer: Regularizer for projection biases.
            activity_regularizer: Regularizer for layer output activity.
            kernel_constraint: Constraint for projection kernels.
            bias_constraint: Constraint for projection biases.
            use_gate: Whether to apply gating on attention outputs.
            seed: Optional random seed for dropout.
            num_heads: Optional alias for ``num_query_heads``.
            key_dim: Optional alias for ``head_dim``.
            value_dim: Size of each attention head for value; defaults to ``head_dim``.
            output_shape: Expected non-batch output feature shape.
            attention_axes: Axes over which attention is applied.
            **kwargs: Base layer keyword arguments.
        """
        resolved_heads = int(num_heads) if num_heads is not None else int(num_query_heads)
        resolved_key_dim = int(key_dim) if key_dim is not None else int(head_dim)
        super().__init__(
            num_heads=resolved_heads,
            key_dim=resolved_key_dim,
            value_dim=value_dim,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=output_shape,
            attention_axes=attention_axes,
            sliding_window=sliding_window,
            flash_attention=flash_attention,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            use_gate=use_gate,
            seed=seed,
            **kwargs,
        )
        self.head_dim = resolved_key_dim
        self.num_query_heads = resolved_heads
        self.num_key_value_heads = int(num_key_value_heads)
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )

    def build(self, input_shape: ShapeTuple) -> None:
        """Build the grouped query, key, value, and output ``EinsumDense`` projection layers.

        Args:
            input_shape: Shape tuple of the query tensor or sequence of input shape tuples.
        """
        if input_shape is None:
            return
        if (
            isinstance(input_shape, tuple)
            and isinstance(input_shape[0], tuple)
            or isinstance(input_shape, list)
            and isinstance(input_shape[0], tuple)
        ):
            q_shape = input_shape[0]

        else:
            q_shape = input_shape

        out_dim = (
            (q_shape[-1] if q_shape is not None else None)
            if self.output_shape_tuple is None
            else (
                self.output_shape_tuple[-1]
                if isinstance(self.output_shape_tuple, tuple)
                else self.output_shape_tuple
            )
        )

        from zero_keras.layers.misc import EinsumDense

        self._query_dense = EinsumDense(
            equation="...X,XHD->...HD",
            output_shape=(self.num_heads, self.key_dim),
            bias_axes="HD" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="query",
        )
        self._key_dense = EinsumDense(
            equation="...X,XHD->...HD",
            output_shape=(self.num_key_value_heads, self.key_dim),
            bias_axes="HD" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="key",
        )
        self._value_dense = EinsumDense(
            equation="...X,XHD->...HD",
            output_shape=(self.num_key_value_heads, self.value_dim),
            bias_axes="HD" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="value",
        )
        self._output_dense = EinsumDense(
            equation="...HD,HDO->...O",
            output_shape=(out_dim,),
            bias_axes="O" if self.use_bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="attention_output",
        )
        self._query_dense.build(q_shape)
        if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) > 1
            and isinstance(input_shape[1], (tuple, list))
        ):
            v_shape = tuple(input_shape[1])
            k_shape = (
                tuple(input_shape[2])
                if len(input_shape) > 2 and isinstance(input_shape[2], (tuple, list))
                else v_shape
            )
            self._key_dense.build(k_shape)
            self._value_dense.build(v_shape)

        out_shape_input = (
            (tuple(q_shape[:-1]) + (self.num_heads, self.value_dim))
            if q_shape is not None
            else None
        )
        self._output_dense.build(out_shape_input)
        self.query_dense = self._query_dense
        self.key_dense = self._key_dense
        self.value_dense = self._value_dense
        self.output_dense = self._output_dense
        self.built = True

    def call(
        self,
        query: TensorLike,
        value: TensorLike,
        key: TensorLike | None = None,
        attention_mask: TensorLike | None = None,
        return_attention_scores: bool = False,
        training: bool | None = False,
        use_causal_mask: bool = False,
    ) -> TensorLike:
        """Compute grouped-query scaled dot-product attention.

        Args:
            query: Query tensor of shape ``(B, T, dim)``.
            value: Value tensor of shape ``(B, S, dim)``.
            key: Optional key tensor of shape ``(B, S, dim)``; defaults to ``value``.
            attention_mask: Optional attention mask tensor.
            return_attention_scores: Whether to also return attention weight scores.
            training: Whether the layer is executing in training mode.
            use_causal_mask: Whether to apply a lower-triangular causal mask.

        Returns:
            TensorLike: Attention output tensor, or ``(attention_output, attention_scores)``.
        """
        if query is None:
            return (None, None) if return_attention_scores else None

        if key is None:
            key = value

        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        from zero_keras import ops
        from zero_keras.types import ShapeCarrierProtocol

        num_repeats = self.num_heads // self.num_key_value_heads
        if num_repeats > 1:
            key = ops.repeat(key, num_repeats, axis=-2)
            value = ops.repeat(value, num_repeats, axis=-2)

        import math

        scale = 1.0 / math.sqrt(float(self.key_dim))

        import string

        letters = string.ascii_lowercase
        q_dims = query.shape if isinstance(query, ShapeCarrierProtocol) else (1, 1, 1, 1)
        b_dims = len(q_dims) - 3
        b_str = letters[:b_dims]
        nxt = letters[b_dims:]
        t_str, h_str, d_str, s_str = nxt[0], nxt[1], nxt[2], nxt[3]
        eq_scores = f"{b_str}{t_str}{h_str}{d_str},{b_str}{s_str}{h_str}{d_str}->{b_str}{h_str}{t_str}{s_str}"
        attention_scores = ops.einsum(eq_scores, query, key)
        attention_scores = ops.multiply(attention_scores, scale)

        if attention_mask is not None:
            mask = ops.cast(attention_mask, "float32")
            attention_scores -= 1e9 * (1.0 - mask)

        if use_causal_mask:
            seq_len = (query.shape if isinstance(query, ShapeCarrierProtocol) else (0, 0, 0))[-3]
            idx = ops.arange(seq_len)
            mask = idx[:, None] >= idx[None, :]
            mask = ops.cast(mask, "float32")
            attention_scores -= 1e9 * (1.0 - mask)

        attention_scores = ops.softmax(attention_scores, axis=-1)

        if training and self.dropout > 0.0:
            attention_scores = ops.dropout(attention_scores, self.dropout)

        eq_out = f"{b_str}{h_str}{t_str}{s_str},{b_str}{s_str}{h_str}{d_str}->{b_str}{t_str}{h_str}{d_str}"
        attention_output = ops.einsum(eq_out, attention_scores, value)
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

"""Recurrent neural network layers (`RNN`, `SimpleRNN`, `LSTM`, `GRU`, `ConvLSTM1D/2D/3D`, `Bidirectional`, `TimeDistributed`) for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer


class RNN(Layer):
    """Base class for recurrent layers.

    Args:
        cell: A RNN cell instance or a list of RNN cell instances.
            A RNN cell is a class that has:
            - A `call(input_at_t, states_at_t)` method, returning
            `(output_at_t, states_at_t_plus_1)`. The call method of the
            cell can also take the optional argument `constants`, see
            section "Note on passing external constants" below.
            - A `state_size` attribute. This can be a single integer
            (single state) in which case it is the size of the recurrent
            state. This can also be a list/tuple of integers
            (one size per state).
            - A `output_size` attribute, a single integer.
            - A `get_initial_state(batch_size=None)`
            method that creates a tensor meant to be fed to `call()` as the
            initial state, if the user didn't specify any initial state
            via other means. The returned initial state should have
            shape `(batch_size, cell.state_size)`.
            The cell might choose to create a tensor full of zeros,
            or other values based on the cell's implementation.
            `inputs` is the input tensor to the RNN layer, with shape
            `(batch_size, timesteps, features)`.
            If this method is not implemented
            by the cell, the RNN layer will create a zero filled tensor
            with shape `(batch_size, cell.state_size)`.
            In the case that `cell` is a list of RNN cell instances, the cells
            will be stacked on top of each other in the RNN, resulting in an
            efficient stacked RNN.
        return_sequences: Boolean (default `False`). Whether to return the last
            output in the output sequence, or the full sequence.
        return_state: Boolean (default `False`).
            Whether to return the last state in addition to the output.
        go_backwards: Boolean (default `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default `False`). If True, the last state
            for each sample at index `i` in a batch will be used as initial
            state for the sample of index `i` in the following batch.
        unroll: Boolean (default `False`).
            If True, the network will be unrolled, else a symbolic loop will be
            used. Unrolling can speed-up a RNN, although it tends to be more
            memory-intensive. Unrolling is only suitable for short sequences.
        zero_output_for_mask: Boolean (default `False`).
            Whether the output should use zeros for the masked timesteps.
            Note that this field is only used when `return_sequences`
            is `True` and `mask` is provided.
            It can useful if you want to reuse the raw output sequence of
            the RNN without interference from the masked timesteps, e.g.,
            merging bidirectional RNNs.

    Call arguments:
        sequences: A 3-D tensor with shape `(batch_size, timesteps, features)`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.
        mask: Binary tensor of shape `[batch_size, timesteps]`
            indicating whether a given timestep should be masked.
            An individual `True` entry indicates that the corresponding
            timestep should be utilized, while a `False` entry indicates
            that the corresponding timestep should be ignored.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed
            to the cell when calling it.
            This is for use with cells that use dropout.

    Output shape:

    - If `return_state`: a list of tensors. The first tensor is
    the output. The remaining tensors are the last states,
    each with shape `(batch_size, state_size)`, where `state_size` could
    be a high dimension tensor shape.
    - If `return_sequences`: 3D tensor with shape
    `(batch_size, timesteps, output_size)`.

    Masking:

    This layer supports masking for input data with a variable number
    of timesteps. To introduce masks to your data,
    use a `keras.layers.Embedding` layer with the `mask_zero` parameter
    set to `True`.

    Note on using statefulness in RNNs:

    You can set RNN layers to be 'stateful', which means that the states
    computed for the samples in one batch will be reused as initial states
    for the samples in the next batch. This assumes a one-to-one mapping
    between samples in different successive batches.

    To enable statefulness:

    - Specify `stateful=True` in the layer constructor.
    - Specify a fixed batch size for your model, by passing
        `batch_size=...` to the `Input` layer(s) of your model.
        Remember to also specify the same `batch_size=...` when
        calling `fit()`, or otherwise use a generator-like
        data source like a `keras.utils.PyDataset` or a
        `tf.data.Dataset`.
    - Specify `shuffle=False` when calling `fit()`, since your
        batches are expected to be temporally ordered.

    To reset the states of your model, call `.reset_state()` on either
    a specific layer, or on your entire model.

    Note on specifying the initial state of RNNs:

    You can specify the initial state of RNN layers symbolically by
    calling them with the keyword argument `initial_state`. The value of
    `initial_state` should be a tensor or list of tensors representing
    the initial state of the RNN layer.

    You can specify the initial state of RNN layers numerically by
    calling `reset_state()` with the keyword argument `states`. The value of
    `states` should be a numpy array or list of numpy arrays representing
    the initial state of the RNN layer.

    Examples:
    ```python
    from zero_keras.layers import RNN
    from zero_keras import ops

    # First, let's define a RNN Cell, as a layer subclass.
    class MinimalRNNCell(keras.Layer):

        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.state_size = units

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')

        def call(self, inputs, states):
            prev_output = states[0]
            h = ops.matmul(inputs, self.kernel)
            output = h + ops.matmul(prev_output, self.recurrent_kernel)
            return output, [output]

    # Let's use this cell in a RNN layer:

    cell = MinimalRNNCell(32)
    x = keras.Input((None, 5))
    layer = RNN(cell)
    y = layer(x)

    # Here's how to use the cell to build a stacked RNN:

    cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    x = keras.Input((None, 5))
    layer = RNN(cells)
    y = layer(x)
    ```

    """

    def __init__(
        self,
        cell: TensorLike,
        return_sequences: TensorLike = False,
        return_state: TensorLike = False,
        go_backwards: TensorLike = False,
        stateful: TensorLike = False,
        unroll: TensorLike = False,
        time_major: TensorLike = False,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            cell: Description.
            return_sequences: Description.
            return_state: Description.
            go_backwards: Description.
            stateful: Description.
            unroll: Description.
            time_major: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        step_input_shape = (input_shape[0],) + tuple(input_shape[2:])
        self.cell.build(step_input_shape)
        self.built = True

    def reset_states(self, states: TensorLike = None) -> TensorLike:
        """reset_states function.

        Args:
        states: Parameter states.

        Returns:
        TensorLike: Return value.

        """
        if hasattr(self, "states") and self.states is not None:
            if states is None:
                from ml_switcheroo_compiler.ops import zeros

                self.states = tuple(zeros(s.shape, dtype=s.dtype) for s in self.states)
            else:
                self.states = states
        else:
            self.states = None

    def call(
        self,
        inputs: TensorLike,
        initial_state: TensorLike = None,
        mask: TensorLike = None,
        training: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        initial_state: Parameter initial_state.
        mask: Parameter mask.
        training: Parameter training.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        inputs = _to_tensor(inputs)

        if initial_state is None:
            if self.stateful and (self.states if hasattr(self, "states") else None) is not None:
                initial_state = self.states
            else:
                batch_size = inputs.shape[1] if self.time_major else inputs.shape[0]
                if hasattr(self.cell, "state_size"):
                    state_size = self.cell.state_size
                    if isinstance(state_size, int):
                        initial_state = (ops.zeros((batch_size, state_size), dtype=inputs.dtype),)
                    else:

                        def _get_shape(s: TensorLike) -> TensorLike:
                            """_get_shape function.

                            Args:
                            s: Parameter s.

                            Returns:
                            TensorLike: Return value.

                            """
                            return (
                                (batch_size,) + tuple(s)
                                if isinstance(s, (tuple, list))
                                else (batch_size, s)
                            )

                        initial_state = tuple(
                            ops.zeros(_get_shape(s), dtype=inputs.dtype) for s in state_size
                        )
                else:
                    initial_state = (ops.zeros((batch_size, self.cell.units), dtype=inputs.dtype),)
                if self.stateful:
                    self.states = initial_state

        if not isinstance(initial_state, (tuple, list)):
            initial_state = (initial_state,)

        def cell_fn(inputs: TensorLike, state: TensorLike) -> TensorLike:
            """cell_fn function.

            Args:
            inputs: Parameter inputs.
            state: Parameter state.

            Returns:
            TensorLike: Return value.

            """
            out, new_state = self.cell(inputs, state, training=training, **kwargs)
            if not isinstance(new_state, (tuple, list)):
                new_state = (new_state,)
            return out, new_state

        outputs, final_state = ops.rnn(
            inputs,
            initial_state,
            cell_fn,
            time_major=self.time_major,
            go_backwards=self.go_backwards,
        )

        if not self.return_sequences:
            outputs = outputs[-1] if self.time_major else outputs[:, -1, :]

        if self.return_state:
            if len(final_state) == 1:
                return _wrap(outputs), _wrap(final_state[0])
            return _wrap(outputs), *[_wrap(s) for s in final_state]

        return _wrap(outputs)


class SimpleRNNCell(Layer):
    """Cell class for SimpleRNN.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.SimpleRNN` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.

    Call arguments:
        sequence: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:
    ```python
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    rnn = keras.layers.RNN(keras.layers.SimpleRNNCell(4))
    output = rnn(inputs)  # The output has shape `(32, 4)`.
    rnn = keras.layers.RNN(
        keras.layers.SimpleRNNCell(4),
        return_sequences=True,
        return_state=True
    )
    # whole_sequence_output has shape `(32, 10, 4)`.
    # final_state has shape `(32, 4)`.
    whole_sequence_output, final_state = rnn(inputs)
    ```

    """

    def __init__(
        self,
        units: TensorLike,
        activation: TensorLike = "tanh",
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        recurrent_initializer: TensorLike = "orthogonal",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            units: Description.
            activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            recurrent_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.units = units
        from zero_keras import activations

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.state_size = self.units
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        input_dim = input_shape[-1]
        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                name="kernel",
            )
        if (self.recurrent_kernel if hasattr(self, "recurrent_kernel") else None) is None:
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer=self.recurrent_initializer,
                name="recurrent_kernel",
            )
        if self.use_bias:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.units,), initializer=self.bias_initializer, name="bias"
                )
        else:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = None
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        states: TensorLike,
        training: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        states: Parameter states.
        training: Parameter training.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        if not self.built:
            self.build(inputs.shape)

        prev_output = (
            _to_tensor(states[0]) if isinstance(states, (tuple, list)) else _to_tensor(states)
        )

        h = ops.dot(inputs, _to_tensor(self.kernel))
        h = ops.add(h, ops.dot(prev_output, _to_tensor(self.recurrent_kernel)))
        if self.use_bias:
            h = ops.add(h, _to_tensor(self.bias))

        if self.activation is not None:
            h = self.activation(h)

        return _wrap(h), _wrap(h)


class SimpleRNN(RNN):
    """Fully-connected RNN where the output is to be fed back as the new input.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer uses
            a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state.  Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.  Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation
            of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for the linear transformation of the
            recurrent state. Default: 0.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state
            in addition to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as the
            initial state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up an RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    Call arguments:
        sequence: A 3D tensor, with shape `[batch, timesteps, feature]`.
        mask: Binary tensor of shape `[batch, timesteps]` indicating whether
            a given timestep should be masked. An individual `True` entry
            indicates that the corresponding timestep should be utilized,
            while a `False` entry indicates that the corresponding timestep
            should be ignored.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            This argument is passed to the cell when calling it.
            This is only relevant if `dropout` or `recurrent_dropout` is used.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.

    Example:
    ```python
    inputs = np.random.random((32, 10, 8))
    simple_rnn = keras.layers.SimpleRNN(4)
    output = simple_rnn(inputs)  # The output has shape `(32, 4)`.
    simple_rnn = keras.layers.SimpleRNN(
        4, return_sequences=True, return_state=True
    )
    # whole_sequence_output has shape `(32, 10, 4)`.
    # final_state has shape `(32, 4)`.
    whole_sequence_output, final_state = simple_rnn(inputs)
    ```

    """

    def __init__(
        self,
        units: TensorLike,
        activation: TensorLike = "tanh",
        use_bias: TensorLike = True,
        return_sequences: TensorLike = False,
        return_state: TensorLike = False,
        go_backwards: TensorLike = False,
        stateful: TensorLike = False,
        unroll: TensorLike = False,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            units: Description.
            activation: Description.
            use_bias: Description.
            return_sequences: Description.
            return_state: Description.
            go_backwards: Description.
            stateful: Description.
            unroll: Description.
            kwargs: Description.
        """
        cell = SimpleRNNCell(units, activation=activation, use_bias=use_bias)
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )


class GRUCell(Layer):
    """Cell class for the GRU layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.GRU` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before",
            True = "after" (default and cuDNN compatible).
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:
    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.GRUCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.GRUCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)

    """

    def __init__(
        self,
        units: TensorLike,
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "sigmoid",
        use_bias: TensorLike = True,
        reset_after: TensorLike = True,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            units: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            reset_after: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.units = units
        from zero_keras import activations

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.reset_after = reset_after
        self.state_size = self.units

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        input_dim = input_shape[-1]
        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=(input_dim, self.units * 3),
                initializer="glorot_uniform",
                name="kernel",
            )
        if (self.recurrent_kernel if hasattr(self, "recurrent_kernel") else None) is None:
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units * 3),
                initializer="orthogonal",
                name="recurrent_kernel",
            )
        if self.use_bias:
            if self.reset_after:
                if (self.bias if hasattr(self, "bias") else None) is None:
                    self.bias = self.add_weight(
                        shape=(2, self.units * 3), initializer="zeros", name="bias"
                    )
            else:
                if (self.bias if hasattr(self, "bias") else None) is None:
                    self.bias = self.add_weight(
                        shape=(self.units * 3,), initializer="zeros", name="bias"
                    )
        else:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = None
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        states: TensorLike,
        training: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        states: Parameter states.
        training: Parameter training.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        if not self.built:
            self.build(inputs.shape)
        state = states[0] if isinstance(states, (tuple, list)) else states
        state = (
            _to_tensor(state)
            if state is not None
            else ops.zeros((inputs.shape[0] if inputs.shape else 1, self.units), dtype=self.dtype)
        )

        out, new_state = ops.gru_cell(
            inputs,
            state,
            _to_tensor(self.kernel),
            _to_tensor(self.recurrent_kernel),
            _to_tensor(self.bias)
            if (self.bias if hasattr(self, "bias") else None) is not None
            else None,
        )
        return _wrap(out), _wrap(new_state)


class GRU(RNN):
    """Gated Recurrent Unit - Cho et al. 2014.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or backend-native)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the cuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation
    when using the TensorFlow backend.

    The requirements to use the cuDNN implementation are:

    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. `reset_after` is `True`
    7. Inputs, if use masking, are strictly right-padded.
    8. Eager execution is enabled in the outermost context.

    There are two variants of the GRU implementation. The default one is based
    on [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to
    hidden state before matrix multiplication. The other one is based on
    [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. To use this variant, set `reset_after=True` and
    `recurrent_activation='sigmoid'`.

    For example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> gru = keras.layers.GRU(4)
    >>> output = gru(inputs)
    >>> output.shape
    (32, 4)
    >>> gru = keras.layers.GRU(4, return_sequences=True, return_state=True)
    >>> whole_sequence_output, final_state = gru(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). `False` is `"before"`,
            `True` is `"after"` (default and cuDNN compatible).
        use_cudnn: Whether to use a cuDNN-backed implementation. `"auto"` will
            attempt to use cuDNN when feasible, and will fallback to the
            default implementation if not.

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.

    """

    def __init__(
        self,
        units: TensorLike,
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "sigmoid",
        use_bias: TensorLike = True,
        return_sequences: TensorLike = False,
        return_state: TensorLike = False,
        go_backwards: TensorLike = False,
        stateful: TensorLike = False,
        unroll: TensorLike = False,
        reset_after: TensorLike = True,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            units: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            return_sequences: Description.
            return_state: Description.
            go_backwards: Description.
            stateful: Description.
            unroll: Description.
            reset_after: Description.
            kwargs: Description.
        """
        cell = GRUCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            reset_after=reset_after,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )


class LSTMCell(Layer):
    """Cell class for the LSTM layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.LSTM` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:
    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.LSTMCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.LSTMCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)

    """

    def __init__(
        self,
        units: TensorLike,
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "sigmoid",
        use_bias: TensorLike = True,
        unit_forget_bias: TensorLike = True,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            units: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            unit_forget_bias: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.units = units
        from zero_keras import activations

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.unit_forget_bias = unit_forget_bias
        self.state_size = (self.units, self.units)

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        input_dim = input_shape[-1]
        if (self.kernel if hasattr(self, "kernel") else None) is None:
            self.kernel = self.add_weight(
                shape=(input_dim, self.units * 4),
                initializer="glorot_uniform",
                name="kernel",
            )
        if (self.recurrent_kernel if hasattr(self, "recurrent_kernel") else None) is None:
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units * 4),
                initializer="orthogonal",
                name="recurrent_kernel",
            )
        if self.use_bias:

            def bias_init(shape: TensorLike, dtype: TensorLike = None) -> TensorLike:
                """bias_init function.

                Args:
                shape: Parameter shape.
                dtype: Parameter dtype.

                Returns:
                TensorLike: Return value.

                """
                b = ops.zeros(shape, dtype=dtype)
                if self.unit_forget_bias:
                    b = ops.dynamic_update_slice(
                        b, ops.ones((self.units,), dtype=dtype), (self.units,)
                    )
                return b

            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = self.add_weight(
                    shape=(self.units * 4,), initializer=bias_init, name="bias"
                )
        else:
            if (self.bias if hasattr(self, "bias") else None) is None:
                self.bias = None
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        states: TensorLike,
        training: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        states: Parameter states.
        training: Parameter training.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        if not self.built:
            self.build(inputs.shape)

        h, c = _to_tensor(states[0]), _to_tensor(states[1])

        out, new_state = ops.lstm_cell(
            inputs,
            (h, c),
            _to_tensor(self.kernel),
            _to_tensor(self.recurrent_kernel),
            _to_tensor(self.bias)
            if (self.bias if hasattr(self, "bias") else None) is not None
            else None,
        )
        return _wrap(out), [_wrap(s) for s in new_state]


class LSTM(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or backend-native)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the cuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation
    when using the TensorFlow backend.
    The requirements to use the cuDNN implementation are:

    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. Inputs, if use masking, are strictly right-padded.
    7. Eager execution is enabled in the outermost context.

    For example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> lstm = keras.layers.LSTM(4)
    >>> output = lstm(inputs)
    >>> output.shape
    (32, 4)
    >>> lstm = keras.layers.LSTM(
    ...     4, return_sequences=True, return_state=True)
    >>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
    >>> whole_seq_output.shape
    (32, 10, 4)
    >>> final_memory_state.shape
    (32, 4)
    >>> final_carry_state.shape
    (32, 4)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        use_cudnn: Whether to use a cuDNN-backed implementation. `"auto"` will
            attempt to use cuDNN when feasible, and will fallback to the
            default implementation if not.

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.

    """

    def __init__(
        self,
        units: TensorLike,
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "sigmoid",
        use_bias: TensorLike = True,
        unit_forget_bias: TensorLike = True,
        return_sequences: TensorLike = False,
        return_state: TensorLike = False,
        go_backwards: TensorLike = False,
        stateful: TensorLike = False,
        unroll: TensorLike = False,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            units: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            unit_forget_bias: Description.
            return_sequences: Description.
            return_state: Description.
            go_backwards: Description.
            stateful: Description.
            unroll: Description.
            kwargs: Description.
        """
        cell = LSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            unit_forget_bias=unit_forget_bias,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )


class Bidirectional(Layer):
    """Bidirectional wrapper for RNNs.

    Args:
        layer: `keras.layers.RNN` instance, such as
            `keras.layers.LSTM` or `keras.layers.GRU`.
            It could also be a `keras.layers.Layer` instance
            that meets the following criteria:
            1. Be a sequence-processing layer (accepts 3D+ inputs).
            2. Have a `go_backwards`, `return_sequences` and `return_state`
            attribute (with the same semantics as for the `RNN` class).
            3. Have an `input_spec` attribute.
            4. Implement serialization via `get_config()` and `from_config()`.
            Note that the recommended way to create new RNN layers is to write a
            custom RNN cell and use it with `keras.layers.RNN`, instead of
            subclassing `keras.layers.Layer` directly.
            When `return_sequences` is `True`, the output of the masked
            timestep will be zero regardless of the layer's original
            `zero_output_for_mask` value.
        merge_mode: Mode by which outputs of the forward and backward RNNs
            will be combined. One of `{"sum", "mul", "concat", "ave", None}`.
            If `None`, the outputs will not be combined,
            they will be returned as a list. Defaults to `"concat"`.
        backward_layer: Optional `keras.layers.RNN`,
            or `keras.layers.Layer` instance to be used to handle
            backwards input processing.
            If `backward_layer` is not provided, the layer instance passed
            as the `layer` argument will be used to generate the backward layer
            automatically.
            Note that the provided `backward_layer` layer should have properties
            matching those of the `layer` argument, in particular
            it should have the same values for `stateful`, `return_states`,
            `return_sequences`, etc. In addition, `backward_layer`
            and `layer` should have different `go_backwards` argument values.
            A `ValueError` will be raised if these requirements are not met.

    Call arguments:
        The call arguments for this layer are the same as those of the
        wrapped RNN layer. Beware that when passing the `initial_state`
        argument during the call of this layer, the first half in the
        list of elements in the `initial_state` list will be passed to
        the forward RNN call and the last half in the list of elements
        will be passed to the backward RNN call.

    Note: instantiating a `Bidirectional` layer from an existing RNN layer
    instance will not reuse the weights state of the RNN layer instance -- the
    `Bidirectional` layer will have freshly initialized weights.

    Examples:
    ```python
    model = Sequential([
        Input(shape=(5, 10)),
        Bidirectional(LSTM(10, return_sequences=True),
        Bidirectional(LSTM(10)),
        Dense(5, activation="softmax"),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # With custom backward layer
    forward_layer = LSTM(10, return_sequences=True)
    backward_layer = LSTM(10, activation='relu', return_sequences=True,
                          go_backwards=True)
    model = Sequential([
        Input(shape=(5, 10)),
        Bidirectional(forward_layer, backward_layer=backward_layer),
        Dense(5, activation="softmax"),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```

    """

    def __init__(
        self,
        layer: TensorLike,
        merge_mode: TensorLike = "concat",
        weights: TensorLike = None,
        backward_layer: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            layer: Description.
            merge_mode: Description.
            weights: Description.
            backward_layer: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.forward_layer = layer
        if backward_layer is None:
            config = layer._kwargs.copy() if hasattr(layer, "_kwargs") else {}
            args = layer.args if hasattr(layer, "args") else ()
            if hasattr(layer, "units"):
                args = (layer.units,)
            elif hasattr(layer, "cell") and hasattr(layer.cell, "units"):
                args = (layer.cell.units,)
            self.backward_layer = type(layer)(*args, **config)
        else:
            self.backward_layer = backward_layer
        self.backward_layer.go_backwards = True
        self.merge_mode = merge_mode
        self.return_sequences = layer.return_sequences
        self.return_state = layer.return_state

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        self.forward_layer.build(input_shape)
        self.backward_layer.build(input_shape)
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        initial_state: TensorLike = None,
        training: TensorLike = None,
        mask: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        initial_state: Parameter initial_state.
        training: Parameter training.
        mask: Parameter mask.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        if initial_state is not None:
            half = len(initial_state) // 2
            forward_state = initial_state[:half]
            backward_state = initial_state[half:]
        else:
            forward_state = None
            backward_state = None

        y_forward = self.forward_layer(
            inputs, initial_state=forward_state, training=training, mask=mask, **kwargs
        )
        y_backward = self.backward_layer(
            inputs, initial_state=backward_state, training=training, mask=mask, **kwargs
        )

        if self.return_state:
            out_f, *state_f = y_forward
            out_b, *state_b = y_backward
        else:
            out_f = y_forward
            out_b = y_backward

        if self.return_sequences:
            out_b = ops.reverse(out_b, dims=(1,))

        if self.merge_mode == "concat":
            outputs = ops.concatenate([out_f, out_b], axis=-1)
        elif self.merge_mode == "sum":
            outputs = ops.add(out_f, out_b)
        elif self.merge_mode == "ave":
            outputs = ops.mean(ops.stack([out_f, out_b]), axis=0)
        elif self.merge_mode == "mul":
            outputs = ops.multiply(out_f, out_b)
        else:
            outputs = out_f

        if self.return_state:
            return (
                _wrap(outputs),
                *[_wrap(s) for s in state_f],
                *[_wrap(s) for s in state_b],
            )
        return _wrap(outputs)


class StackedRNNCells(Layer):
    """Wrapper allowing a stack of RNN cells to behave as a single cell.

    Used to implement efficient stacked RNNs.

    Args:
      cells: List of RNN cell instances.

    Example:
    ```python
    batch_size = 3
    sentence_length = 5
    num_features = 2
    new_shape = (batch_size, sentence_length, num_features)
    x = np.reshape(np.arange(30), new_shape)

    rnn_cells = [keras.layers.LSTMCell(128) for _ in range(2)]
    stacked_lstm = keras.layers.StackedRNNCells(rnn_cells)
    lstm_layer = keras.layers.RNN(stacked_lstm)

    result = lstm_layer(x)
    ```

    """

    def __init__(self, cells: TensorLike, **kwargs: TensorLike) -> None:
        """Function docstring.

        Args:
            cells: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.cells = cells

    @property
    def state_size(self) -> TensorLike:
        """state_size function.

        Returns:
        TensorLike: Return value.

        """
        sizes = []
        for c in self.cells:
            if isinstance(c.state_size, (list, tuple)):
                sizes.extend(c.state_size)
            else:
                sizes.append(c.state_size)
        return tuple(sizes)

    def call(
        self,
        inputs: TensorLike,
        states: TensorLike,
        training: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        states: Parameter states.
        training: Parameter training.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        new_states = []
        state_idx = 0
        for cell in self.cells:
            if isinstance(cell.state_size, (list, tuple)):
                state = states[state_idx : state_idx + len(cell.state_size)]
                state_idx += len(cell.state_size)
            else:
                state = states[state_idx : state_idx + 1]
                state_idx += 1
            inputs, new_state = cell(inputs, state, training=training, **kwargs)
            if isinstance(new_state, (list, tuple)):
                new_states.extend(new_state)
            else:
                new_states.append(new_state)
        return _wrap(inputs), tuple(_wrap(s) for s in new_states)


class ConvLSTM1D(RNN):
    """1D Convolutional LSTM.

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 1 integer, specifying the size of
            the convolution window.
        strides: int or tuple/list of 1 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the
            same height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function to use. By default hyperbolic tangent
            activation function is applied (`tanh(x)`).
        recurrent_activation: Activation function to use for the recurrent step.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean. If `True`, add 1 to the bias of
            the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al., 2015](
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.


    Call arguments:
        inputs: A 4D tensor.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether a
            given timestep should be masked.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            This is only relevant if `dropout` or `recurrent_dropout` are set.

    Input shape:

    - If `data_format="channels_first"`:
        4D tensor with shape: `(samples, time, channels, rows)`
    - If `data_format="channels_last"`:
        4D tensor with shape: `(samples, time, rows, channels)`

    Output shape:

    - If `return_state`: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states,
        each 3D tensor with shape: `(samples, filters, new_rows)` if
        `data_format='channels_first'`
        or shape: `(samples, new_rows, filters)` if
        `data_format='channels_last'`.
        `rows` values might have changed due to padding.
    - If `return_sequences`: 4D tensor with shape: `(samples, timesteps,
        filters, new_rows)` if data_format='channels_first'
        or shape: `(samples, timesteps, new_rows, filters)` if
        `data_format='channels_last'`.
    - Else, 3D tensor with shape: `(samples, filters, new_rows)` if
        `data_format='channels_first'`
        or shape: `(samples, new_rows, filters)` if
        `data_format='channels_last'`.

    References:
    - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
        (the current implementation does not include the feedback loop on the
        cells output).

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = 1,
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "hard_sigmoid",
        use_bias: TensorLike = True,
        return_sequences: TensorLike = False,
        return_state: TensorLike = False,
        go_backwards: TensorLike = False,
        stateful: TensorLike = False,
        dropout: TensorLike = 0.0,
        recurrent_dropout: TensorLike = 0.0,
        kernel_initializer: TensorLike = "glorot_uniform",
        recurrent_initializer: TensorLike = "orthogonal",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            return_sequences: Description.
            return_state: Description.
            go_backwards: Description.
            stateful: Description.
            dropout: Description.
            recurrent_dropout: Description.
            kernel_initializer: Description.
            recurrent_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        cell = ConvLSTMCell(
            filters=filters,
            kernel_size=kernel_size,
            rank=1,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )


class ConvLSTM2D(RNN):
    """2D Convolutional LSTM.

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 2 integers, specifying the size of the
            convolution window.
        strides: int or tuple/list of 2 integers, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function to use. By default hyperbolic tangent
            activation function is applied (`tanh(x)`).
        recurrent_activation: Activation function to use for the recurrent step.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean. If `True`, add 1 to the bias of the forget
            gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al., 2015](
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.


    Call arguments:
        inputs: A 5D tensor.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether a
            given timestep should be masked.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            This is only relevant if `dropout` or `recurrent_dropout` are set.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.

    Input shape:

    - If `data_format='channels_first'`:
        5D tensor with shape: `(samples, time, channels, rows, cols)`
    - If `data_format='channels_last'`:
        5D tensor with shape: `(samples, time, rows, cols, channels)`

    Output shape:

    - If `return_state`: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states,
        each 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        `data_format='channels_first'`
        or shape: `(samples, new_rows, new_cols, filters)` if
        `data_format='channels_last'`. `rows` and `cols` values might have
        changed due to padding.
    - If `return_sequences`: 5D tensor with shape: `(samples, timesteps,
        filters, new_rows, new_cols)` if data_format='channels_first'
        or shape: `(samples, timesteps, new_rows, new_cols, filters)` if
        `data_format='channels_last'`.
    - Else, 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        `data_format='channels_first'`
        or shape: `(samples, new_rows, new_cols, filters)` if
        `data_format='channels_last'`.

    References:
    - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
        (the current implementation does not include the feedback loop on the
        cells output).

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = (1, 1),
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = (1, 1),
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "hard_sigmoid",
        use_bias: TensorLike = True,
        return_sequences: TensorLike = False,
        return_state: TensorLike = False,
        go_backwards: TensorLike = False,
        stateful: TensorLike = False,
        dropout: TensorLike = 0.0,
        recurrent_dropout: TensorLike = 0.0,
        kernel_initializer: TensorLike = "glorot_uniform",
        recurrent_initializer: TensorLike = "orthogonal",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            return_sequences: Description.
            return_state: Description.
            go_backwards: Description.
            stateful: Description.
            dropout: Description.
            recurrent_dropout: Description.
            kernel_initializer: Description.
            recurrent_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        cell = ConvLSTMCell(
            filters=filters,
            kernel_size=kernel_size,
            rank=2,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )


class ConvLSTM3D(RNN):
    """3D Convolutional LSTM.

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 3 integers, specifying the size of the
            convolution window.
        strides: int or tuple/list of 3 integers, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 3 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function to use. By default hyperbolic tangent
            activation function is applied (`tanh(x)`).
        recurrent_activation: Activation function to use for the recurrent step.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean. If `True`, add 1 to the bias of the forget
            gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al., 2015](
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default: `False`).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.


    Call arguments:
        inputs: A 6D tensor.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether a
            given timestep should be masked.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            This is only relevant if `dropout` or `recurrent_dropout` are set.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.

    Input shape:

    - If `data_format='channels_first'`:
        5D tensor with shape: `(samples, time, channels, *spatial_dims)`
    - If `data_format='channels_last'`:
        5D tensor with shape: `(samples, time, *spatial_dims, channels)`

    Output shape:

    - If `return_state`: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states,
        each 4D tensor with shape: `(samples, filters, *spatial_dims)` if
        `data_format='channels_first'`
        or shape: `(samples, *spatial_dims, filters)` if
        `data_format='channels_last'`.
    - If `return_sequences`: 5D tensor with shape: `(samples, timesteps,
        filters, *spatial_dims)` if data_format='channels_first'
        or shape: `(samples, timesteps, *spatial_dims, filters)` if
        `data_format='channels_last'`.
    - Else, 4D tensor with shape: `(samples, filters, *spatial_dims)` if
        `data_format='channels_first'`
        or shape: `(samples, *spatial_dims, filters)` if
        `data_format='channels_last'`.

    References:
    - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
        (the current implementation does not include the feedback loop on the
        cells output).

    """

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        strides: TensorLike = (1, 1, 1),
        padding: TensorLike = "valid",
        data_format: TensorLike = None,
        dilation_rate: TensorLike = (1, 1, 1),
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "hard_sigmoid",
        use_bias: TensorLike = True,
        return_sequences: TensorLike = False,
        return_state: TensorLike = False,
        go_backwards: TensorLike = False,
        stateful: TensorLike = False,
        dropout: TensorLike = 0.0,
        recurrent_dropout: TensorLike = 0.0,
        kernel_initializer: TensorLike = "glorot_uniform",
        recurrent_initializer: TensorLike = "orthogonal",
        bias_initializer: TensorLike = "zeros",
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            strides: Description.
            padding: Description.
            data_format: Description.
            dilation_rate: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            return_sequences: Description.
            return_state: Description.
            go_backwards: Description.
            stateful: Description.
            dropout: Description.
            recurrent_dropout: Description.
            kernel_initializer: Description.
            recurrent_initializer: Description.
            bias_initializer: Description.
            kwargs: Description.
        """
        cell = ConvLSTMCell(
            filters=filters,
            kernel_size=kernel_size,
            rank=3,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )


class ConvLSTMCell(Layer):
    """ConvLSTMCell class."""

    def __init__(
        self,
        filters: TensorLike,
        kernel_size: TensorLike,
        rank: TensorLike = 2,
        strides: TensorLike = 1,
        padding: TensorLike = "valid",
        dilation_rate: TensorLike = 1,
        activation: TensorLike = "tanh",
        recurrent_activation: TensorLike = "hard_sigmoid",
        use_bias: TensorLike = True,
        kernel_initializer: TensorLike = "glorot_uniform",
        recurrent_initializer: TensorLike = "orthogonal",
        bias_initializer: TensorLike = "zeros",
        dropout: TensorLike = 0.0,
        recurrent_dropout: TensorLike = 0.0,
        **kwargs: TensorLike,
    ) -> None:
        """Function docstring.

        Args:
            filters: Description.
            kernel_size: Description.
            rank: Description.
            strides: Description.
            padding: Description.
            dilation_rate: Description.
            activation: Description.
            recurrent_activation: Description.
            use_bias: Description.
            kernel_initializer: Description.
            recurrent_initializer: Description.
            bias_initializer: Description.
            dropout: Description.
            recurrent_dropout: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * rank
        self.kernel_size = kernel_size
        self.rank = rank
        if isinstance(strides, int):
            strides = (strides,) * rank
        self.strides = strides
        self.padding = padding
        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate,) * rank
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        from zero_keras import activations

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.state_size = None

    def build(self, input_shape: TensorLike) -> TensorLike:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        if self.built:
            return
        in_channels = input_shape[-1]

        kernel_shape = self.kernel_size + (in_channels, self.filters * 4)
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)

        self.kernel = self.add_weight(
            shape=kernel_shape, initializer=self.kernel_initializer, name="kernel"
        )
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters * 4,),
                initializer=self.bias_initializer,
                name="bias",
            )
        else:
            self.bias = None

        out_spatial = []
        for d, k, s, p, d_r in zip(
            input_shape[1:-1],
            self.kernel_size,
            self.strides,
            (self.padding,) * self.rank if isinstance(self.padding, str) else self.padding,
            self.dilation_rate,
        ):
            if d is None:
                out_spatial.append(None)
                continue
            eff_k = k + (k - 1) * (d_r - 1)
            if isinstance(self.padding, str) and self.padding.lower() == "same":
                out_spatial.append((d + s - 1) // s)
            else:
                out_spatial.append((d - eff_k) // s + 1)
        self.state_size = (
            tuple(out_spatial) + (self.filters,),
            tuple(out_spatial) + (self.filters,),
        )
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        states: TensorLike,
        training: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        states: Parameter states.
        training: Parameter training.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        h_tm1, c_tm1 = states

        inputs = _to_tensor(inputs)
        h_tm1 = _to_tensor(h_tm1)
        c_tm1 = _to_tensor(c_tm1)

        from ml_switcheroo_compiler.ops.configs import ConvConfig

        config_obj = ConvConfig(
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=None,
            rhs_dilation=self.dilation_rate,
            dimension_numbers=((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))
            if self.rank == 2
            else ((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3))
            if self.rank == 3
            else ((0, 2, 1), (2, 1, 0), (0, 2, 1)),
        )

        from zero_keras import ops

        conv_general_dilated = ops.conv_general_dilated

        inputs_dropped = inputs
        if training and self.dropout > 0.0:
            inputs_dropped = ops.dropout(inputs, rate=self.dropout)
        h_tm1_dropped = h_tm1
        if training and self.recurrent_dropout > 0.0:
            h_tm1_dropped = ops.dropout(h_tm1, rate=self.recurrent_dropout)
        z = conv_general_dilated(inputs_dropped, _to_tensor(self.kernel), config_obj)
        z_recurrent = conv_general_dilated(
            h_tm1_dropped, _to_tensor(self.recurrent_kernel), config_obj
        )

        z = ops.add(z, z_recurrent)
        if self.use_bias:
            z = ops.add(z, _to_tensor(self.bias))

        z0, z1, z2, z3 = ops.split(z, 4, axis=-1)

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c_p = self.activation(z2)
        o = self.recurrent_activation(z3)

        c = ops.add(ops.multiply(f, c_tm1), ops.multiply(i, c_p))
        h = ops.multiply(o, self.activation(c))

        return _wrap(h), (_wrap(h), _wrap(c))

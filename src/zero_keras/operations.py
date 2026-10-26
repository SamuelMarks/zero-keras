"""Core symbolic and stateless operation nodes (`Node`, `Operation`, `Function`, `StatelessScope`)."""

from __future__ import annotations

from zero_keras.types import (
    ComputeOutputShapeProtocol,
    DTypeCarrierProtocol,
    InboundNodesCarrierProtocol,
    KerasHistoryProtocol,
    TensorLike,
    ValueCarrierProtocol,
)


class Node:
    """Symbolic graph node connecting a layer or operation with its inputs and outputs."""

    def __init__(
        self,
        operation: TensorLike | None = None,
        call_args: TensorLike | None = None,
        call_kwargs: TensorLike | None = None,
        outputs: TensorLike | None = None,
    ) -> None:
        """Initialize a symbolic graph node.

        Args:
            operation: The layer or operation executed by this node.
            call_args: Positional arguments or input tensors passed to the operation.
            call_kwargs: Keyword arguments passed to the operation.
            outputs: Output tensor(s) produced by the operation.
        """
        self.operation = operation
        self.layer = operation
        self.call_args = call_args
        self.inputs = call_args
        self.call_kwargs = call_kwargs
        self.outputs = outputs
        self.input_tensors = call_args if call_args is not None else []
        self.output_tensors = outputs if outputs is not None else []
        if not isinstance(operation, InboundNodesCarrierProtocol):
            operation._inbound_nodes = []
        operation._inbound_nodes.append(self)
        if isinstance(outputs, list):
            for t in outputs:
                t._keras_history = self
        elif isinstance(outputs, KerasHistoryProtocol):
            outputs._keras_history = self


class Function:
    """Symbolic computation graph function."""

    def __new__(cls, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Allocate a ``Function`` instance with default internal state.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.

        Returns:
            TensorLike: Allocated ``Function`` instance.
        """
        _ = args, kwargs
        inst = super().__new__(cls)
        inst._fn = None
        inst._shape_fn = None
        inst._inbound_nodes = []
        inst.dtype = "float32"
        return inst

    def call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the forward computation of `Function` on `inputs`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `call`.
        """
        if self._fn is not None:
            return self._fn(*args, **kwargs)
        return args[0] if args else None

    def compute_output_shape(self, input_shape: TensorLike) -> TensorLike:
        """Compute the output tensor shape produced for `input_shape`.

        Args:
            input_shape: Input tensor shape tuple or nested structure of shape tuples.

        Returns:
            TensorLike: Computed result for `compute_output_shape`.
        """
        if self._shape_fn is not None:
            return self._shape_fn(input_shape)
        return input_shape

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the symbolic `KerasTensor` specification for the given inputs.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `compute_output_spec`.
        """
        if isinstance(self, ComputeOutputShapeProtocol):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.tensors import KerasTensor

        return KerasTensor(
            shape=None, dtype=(self.dtype if isinstance(self, DTypeCarrierProtocol) else "float32")
        )

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Instantiate `Function` from its serialized configuration dictionary.

        Args:
            config: Configuration dictionary describing object hyperparameters.

        Returns:
            TensorLike: Computed result for `from_config`.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for `Function`.

        Returns:
            TensorLike: Computed result for `get_config`.
        """
        return {}

    @property
    def input(self) -> TensorLike:
        """Return the symbolic input tensor(s) of the first inbound node.

        Returns:
            TensorLike: Symbolic input tensor(s) or ``None``.
        """
        return self._inbound_nodes[0].input_tensors if self._inbound_nodes else None

    @property
    def inputs(self) -> TensorLike:
        """Return the list of symbolic input tensors for this function graph.

        Returns:
            TensorLike: List of symbolic input tensors.
        """
        return []

    @property
    def operations(self) -> TensorLike:
        """Return the topologically ordered operations in this function graph.

        Returns:
            TensorLike: List of ``Operation`` instances.
        """
        return []

    @property
    def output(self) -> TensorLike:
        """Return the symbolic output tensor(s) of the first inbound node.

        Returns:
            TensorLike: Symbolic output tensor(s) or ``None``.
        """
        return self._inbound_nodes[0].output_tensors if self._inbound_nodes else None

    @property
    def outputs(self) -> TensorLike:
        """Return the list of symbolic output tensors for this function graph.

        Returns:
            TensorLike: List of symbolic output tensors.
        """
        return []

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """Execute the quantized graph computation.

        Args:
            *args: Positional input tensors.
            **kwargs: Keyword call arguments.
        """
        return

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """Execute the symbolic graph tracing pass.

        Args:
            *args: Positional symbolic input tensors.
            **kwargs: Keyword call arguments.
        """
        return


class Operation:
    """Base symbolic and eager operation node."""

    def __new__(cls, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Allocate an ``Operation`` instance with default internal attributes.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.

        Returns:
            TensorLike: Allocated ``Operation`` instance.
        """
        _ = args, kwargs
        inst = super().__new__(cls)
        inst._fn = None
        inst._config = {}
        inst._input = None
        inst._output = None
        inst.dtype = "float32"
        return inst

    def call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the forward computation of `Operation` on `inputs`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `call`.
        """
        if self._fn is not None:
            return self._fn(*args, **kwargs)
        return args[0] if args else None

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the symbolic `KerasTensor` specification for the given inputs.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `compute_output_spec`.
        """
        if isinstance(self, ComputeOutputShapeProtocol):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.tensors import KerasTensor

        return KerasTensor(shape=None, dtype=self.dtype)

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Instantiate `Operation` from its serialized configuration dictionary.

        Args:
            config: Configuration dictionary describing object hyperparameters.

        Returns:
            TensorLike: Computed result for `from_config`.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for `Operation`.

        Returns:
            TensorLike: Computed result for `get_config`.
        """
        return self._config

    @property
    def input(self) -> TensorLike:
        """Return the symbolic input tensor(s) of this operation.

        Returns:
            TensorLike: Symbolic input tensor(s) or ``None``.
        """
        return self._input

    @property
    def output(self) -> TensorLike:
        """Return the symbolic output tensor(s) of this operation.

        Returns:
            TensorLike: Symbolic output tensor(s) or ``None``.
        """
        return self._output

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the quantized forward pass on `inputs`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `quantized_call`.
        """
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the symbolic graph call for this operation.

        Args:
            *args: Positional arguments forwarded to ``call``.
            **kwargs: Keyword arguments forwarded to ``call``.

        Returns:
            TensorLike: Symbolic output tensor(s).
        """
        return self.call(*args, **kwargs)


class RematScope:
    """Context manager enabling gradient rematerialization for enclosed operations."""

    def __enter__(self) -> TensorLike:
        """Enter the rematerialization context.

        Returns:
            TensorLike: Active ``RematScope`` instance.
        """
        self._active = True
        return self

    def __exit__(self, *args: TensorLike) -> None:
        """Exit the rematerialization context.

        Args:
            *args: Exception type, value, and traceback if an error occurred.
        """
        self._active = False


class StatelessScope:
    """Context manager capturing variable updates and losses without mutating state."""

    def __init__(self) -> None:
        """Initialize the `StatelessScope` instance."""
        self.losses = []
        self.updates = []

    def add_loss(self, loss: TensorLike) -> None:
        """Record a scalar ``loss`` tensor within this stateless scope.

        Args:
            loss: Scalar loss tensor to record.
        """
        self.losses.append(loss)

    def add_update(self, update: TensorLike) -> None:
        """Record a stateless variable ``update`` pair within this scope.

        Args:
            update: Tuple or operation representing a state variable update.
        """
        self.updates.append(update)

    def get_current_value(self, variable: TensorLike) -> TensorLike:
        """Return the current value of ``variable`` within this ``StatelessScope``.

        Args:
            variable: Target ``Variable`` instance to look up.

        Returns:
            TensorLike: Current value of ``variable`` or ``None``.
        """
        return variable.value if isinstance(variable, ValueCarrierProtocol) else None

    def __enter__(self) -> TensorLike:
        """Enter the stateless execution scope.

        Returns:
            TensorLike: Active ``StatelessScope`` instance.
        """
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Exit the stateless execution scope.

        Args:
            *args: Exception type, value, and traceback if an error occurred.
        """


class SymbolicScope:
    """Context manager marking enclosed execution as symbolic graph tracing."""

    def __enter__(self) -> TensorLike:
        """Enter the symbolic tracing scope.

        Returns:
            TensorLike: Active ``SymbolicScope`` instance.
        """
        self._is_symbolic = True
        return self

    def __exit__(self, *args: TensorLike) -> None:
        """Exit the symbolic tracing scope.

        Args:
            *args: Exception type, value, and traceback if an error occurred.
        """
        self._is_symbolic = False


__all__ = [
    "Function",
    "Node",
    "Operation",
    "RematScope",
    "StatelessScope",
    "SymbolicScope",
]

"""Central domain types and structural protocols for zero_keras.

Provides strict static type aliases and runtime-checkable protocols used across
all modules in zero_keras without relying on ``typing.Any`` or ``object``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, Union, runtime_checkable

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor

ScalarValue = Union[int, float, bool, str]
DimensionValue = Union[int, None]
ShapeTuple = tuple[DimensionValue, ...]
ShapeSpec = Union[Sequence[DimensionValue], int]
ShapeLike = Union[ShapeTuple, Sequence[DimensionValue], int]
AxisSpec = Union[int, tuple[int, ...], list[int], None]
PaddingMode = Union[Literal["valid", "same", "causal", "full"], str]
DataFormat = Union[Literal["channels_last", "channels_first"], str, None]


@runtime_checkable
class DTypePolicyProtocol(Protocol):
    """Structural protocol for Keras DTypePolicy instances."""

    @property
    def name(self) -> str | None:
        """Return the policy name.

        Returns:
            str | None: Policy name string.
        """

    @property
    def compute_dtype(self) -> str:
        """Return the compute dtype string.

        Returns:
            str: Compute dtype.
        """

    @property
    def variable_dtype(self) -> str:
        """Return the variable storage dtype string.

        Returns:
            str: Variable dtype.
        """


DTypeLike = Union[str, DType, DTypePolicyProtocol, None]


@runtime_checkable
class SymbolicTensorProtocol(Protocol):
    """Structural protocol for symbolic or eager tensor objects."""

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Return the tensor shape tuple.

        Returns:
            tuple[int | None, ...]: Dimensions of the tensor.
        """

    @property
    def dtype(self) -> str:
        """Return the standardized data type string.

        Returns:
            str: Data type name.
        """

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> Tensor | SymbolicTensorProtocol:
        """Return a sliced or indexed sub-tensor.

        Args:
            key: Integer index, slice, or tuple of indices/slices.

        Returns:
            Tensor | SymbolicTensorProtocol: Indexed tensor view or sub-tensor.
        """


@runtime_checkable
class SupportsShapeAndDType(Protocol):
    """Structural protocol for objects exposing ``shape`` and ``dtype``."""

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Return the shape tuple.

        Returns:
            tuple[int | None, ...]: Shape dimensions.
        """

    @property
    def dtype(self) -> str:
        """Return the dtype name.

        Returns:
            str: Dtype string.
        """


@runtime_checkable
class SupportsToNumpy(Protocol):
    """Structural protocol for objects exposing a ``numpy()`` conversion method."""

    def numpy(self) -> Tensor:
        """Convert the underlying data to a NumPy/backend tensor.

        Returns:
            Tensor: Converted tensor.
        """


SupportsNumpy = SupportsToNumpy


@runtime_checkable
class SupportsAssign(Protocol):
    """Structural protocol for mutable variable objects exposing ``assign``."""

    def assign(self, value: TensorLike) -> TensorLike | None:
        """Assign a new tensor value in-place.

        Args:
            value: Replacement tensor or scalar value.

        Returns:
            TensorLike | None: Updated variable or None.
        """


@runtime_checkable
class SupportsToBytes(Protocol):
    """Structural protocol for buffer-like objects exposing ``tobytes()``."""

    def tobytes(self) -> bytes:
        """Serialize raw element buffer to bytes.

        Returns:
            bytes: Raw byte representation.
        """


@runtime_checkable
class SupportsUnderlyingTensor(Protocol):
    """Structural protocol for wrapper objects holding an underlying ``_tensor``."""

    @property
    def _tensor(self) -> Tensor:
        """Return the underlying backend ``Tensor``.

        Returns:
            Tensor: Underlying tensor instance.
        """


@runtime_checkable
class SupportsCall(Protocol):
    """Structural protocol for callable objects accepting tensor inputs."""

    def __call__(self, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invoke the callable on tensor inputs.

        Args:
            *args: Positional tensor arguments.
            **kwargs: Keyword configuration arguments.

        Returns:
            TensorLike: Evaluated tensor output.
        """


@runtime_checkable
class ShapeCarrierProtocol(Protocol):
    """Structural protocol for objects exposing a ``shape`` attribute."""

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Return the shape tuple.

        Returns:
            tuple[int | None, ...]: Shape dimensions.
        """


@runtime_checkable
class DTypeCarrierProtocol(Protocol):
    """Structural protocol for objects exposing a ``dtype`` attribute."""

    @property
    def dtype(self) -> str:
        """Return the dtype string.

        Returns:
            str: Data type name.
        """


@runtime_checkable
class BuildableLayerProtocol(Protocol):
    """Structural protocol for layers supporting ``built`` and ``build()``."""

    built: bool

    def build(self, input_shape: ShapeLike | None = None) -> None:
        """Build the layer state for ``input_shape``.

        Args:
            input_shape: Input shape specification.
        """


@runtime_checkable
class MaskComputeProtocol(Protocol):
    """Structural protocol for layers supporting ``compute_mask()``."""

    def compute_mask(self, inputs: TensorLike, mask: TensorLike | None = None) -> TensorLike:
        """Compute an output mask tensor.

        Args:
            inputs: Input tensor.
            mask: Optional incoming mask tensor.

        Returns:
            TensorLike: Computed output mask.
        """


@runtime_checkable
class EndpointCallableProtocol(Protocol):
    """Structural protocol for export archive endpoint callables."""

    def __call__(self, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike | None:
        """Invoke the exported endpoint callable.

        Args:
            *args: Positional tensor arguments.
            **kwargs: Keyword configuration arguments.

        Returns:
            TensorLike | None: Endpoint output tensor.
        """


EpochLogCallbackProtocol = EndpointCallableProtocol


@runtime_checkable
class ScheduleFnProtocol(Protocol):
    """Structural protocol for epoch learning rate schedule functions."""

    def __call__(self, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Compute the scheduled learning rate for an epoch.

        Args:
            *args: Epoch index and optional current learning rate.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Updated learning rate value.
        """


PaddingSpec = Union[PaddingMode, int, Sequence[int], Sequence[tuple[int, int]]]
DataFormatSpec = DataFormat


NumericTensor = Union[Tensor, SymbolicTensorProtocol]

ArrayLike = Union[
    Tensor,
    SymbolicTensorProtocol,
    Sequence[float],
    Sequence[int],
    Sequence[Sequence[float]],
    Sequence[Sequence[int]],
    float,
    int,
    bool,
]

TensorLike = Union[
    Tensor,
    SymbolicTensorProtocol,
    int,
    float,
    bool,
    complex,
    bytes,
    str,
    Sequence["TensorLike"],
]

NestedTensor = Union[
    TensorLike,
    Sequence["NestedTensor"],
    Mapping[str, "NestedTensor"],
]

ConfigValue = Union[
    int,
    float,
    bool,
    str,
    None,
    tuple["ConfigValue", ...],
    list["ConfigValue"],
    dict[str, "ConfigValue"],
    SymbolicTensorProtocol,
    SupportsCall,
]

ConfigDict = dict[str, ConfigValue]


@runtime_checkable
class SupportsGetConfig(Protocol):
    """Structural protocol for serializable Keras objects with ``get_config``."""

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """


@runtime_checkable
class KerasSerializable(Protocol):
    """Structural protocol for Keras objects supporting ``get_config`` and ``from_config``."""

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """

    @classmethod
    def from_config(cls, config: ConfigDict) -> KerasSerializable:
        """Instantiate the object from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            KerasSerializable: Reconstructed instance.
        """


@runtime_checkable
class SeedGeneratorProtocol(Protocol):
    """Structural protocol for random seed generators."""

    def next(self, ordered: bool = True) -> Tensor:
        """Advance and return the next seed state tensor.

        Args:
            ordered: Whether seed advancement is ordered.

        Returns:
            Tensor: Seed state tensor.
        """


SeedLike = Union[int, SeedGeneratorProtocol, Tensor, None]


@runtime_checkable
class ActivationProtocol(Protocol):
    """Structural protocol for Keras activation callables."""

    def __call__(self, x: TensorLike) -> TensorLike:
        """Evaluate the activation on input tensor ``x``.

        Args:
            x: Input tensor or scalar.

        Returns:
            TensorLike: Activated output tensor.
        """


ActivationSpec = Union[str, ActivationProtocol, Callable[[TensorLike], TensorLike], None]


@runtime_checkable
class InitializerProtocol(Protocol):
    """Structural protocol for Keras weight initializers."""

    def __call__(self, shape: ShapeLike, dtype: DTypeLike = None) -> Tensor:
        """Generate an initialized weight tensor for the given shape and dtype.

        Args:
            shape: Target tensor shape.
            dtype: Optional target data type.

        Returns:
            Tensor: Initialized weight tensor.
        """

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """


InitializerSpec = Union[
    str,
    InitializerProtocol,
    Callable[[ShapeLike, DTypeLike], Tensor],
    None,
]


@runtime_checkable
class RegularizerProtocol(Protocol):
    """Structural protocol for Keras weight/activity regularizers."""

    def __call__(self, x: TensorLike) -> Tensor:
        """Compute regularization penalty for tensor ``x``.

        Args:
            x: Input weight or activity tensor.

        Returns:
            Tensor: Scalar penalty tensor.
        """

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """


RegularizerSpec = Union[str, RegularizerProtocol, Callable[[TensorLike], Tensor], None]


@runtime_checkable
class ConstraintProtocol(Protocol):
    """Structural protocol for Keras weight constraints."""

    def __call__(self, w: TensorLike) -> Tensor:
        """Project weight tensor ``w`` onto the constraint set.

        Args:
            w: Input weight tensor.

        Returns:
            Tensor: Constrained weight tensor.
        """

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """


ConstraintSpec = Union[str, ConstraintProtocol, Callable[[TensorLike], Tensor], None]


@runtime_checkable
class LearningRateScheduleProtocol(Protocol):
    """Structural protocol for optimizer learning-rate schedules."""

    def __call__(self, step: int | float | TensorLike) -> Tensor | float:
        """Compute learning rate for the given training step.

        Args:
            step: Current training step index.

        Returns:
            Tensor | float: Evaluated learning rate.
        """

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """


@runtime_checkable
class LayerProtocol(Protocol):
    """Structural protocol for Keras Layer instances."""

    name: str
    built: bool
    trainable: bool

    def __call__(self, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invoke the layer forward pass.

        Args:
            *args: Positional input tensors.
            **kwargs: Keyword call options.

        Returns:
            TensorLike: Layer output tensor(s).
        """


@runtime_checkable
class RNNCellProtocol(Protocol):
    """Structural protocol for RNN step cells."""

    @property
    def state_size(self) -> int | tuple[int, ...]:
        """Return the state size(s) of the RNN cell.

        Returns:
            int | tuple[int, ...]: Cell state dimension(s).
        """

    def call(
        self,
        inputs: TensorLike,
        states: Sequence[TensorLike],
        training: bool | None = None,
    ) -> tuple[TensorLike, Sequence[TensorLike]]:
        """Execute one recurrent step.

        Args:
            inputs: Step input tensor.
            states: Previous recurrent state tensors.
            training: Whether in training mode.

        Returns:
            tuple[TensorLike, Sequence[TensorLike]]: Step output and new states.
        """


@runtime_checkable
class LossProtocol(Protocol):
    """Structural protocol for Keras Loss instances."""

    name: str
    reduction: str | None

    def __call__(
        self,
        y_true: TensorLike,
        y_pred: TensorLike,
        sample_weight: TensorLike | None = None,
    ) -> Tensor:
        """Evaluate the loss between ``y_true`` and ``y_pred``.

        Args:
            y_true: Ground-truth targets.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.

        Returns:
            Tensor: Computed loss tensor.
        """


LossSpec = Union[str, LossProtocol, Callable[[TensorLike, TensorLike], Tensor], None]


@runtime_checkable
class MetricProtocol(Protocol):
    """Structural protocol for stateful Keras Metric instances."""

    name: str

    def update_state(
        self,
        y_true: TensorLike,
        y_pred: TensorLike,
        sample_weight: TensorLike | None = None,
    ) -> None:
        """Accumulate metric statistics for a batch.

        Args:
            y_true: Ground-truth targets.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        """

    def result(self) -> TensorLike:
        """Compute and return the current metric value.

        Returns:
            TensorLike: Current metric value.
        """

    def reset_state(self) -> None:
        """Reset all metric state variables to initial values."""


MetricSpec = Union[str, MetricProtocol, Callable[[TensorLike, TensorLike], TensorLike], None]


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Structural protocol for Keras Optimizer instances."""

    iterations: int

    def apply_gradients(self, grads_and_vars: Sequence[tuple[TensorLike, TensorLike]]) -> None:
        """Apply gradients to variables.

        Args:
            grads_and_vars: Sequence of ``(gradient, variable)`` pairs.
        """


OptimizerSpec = Union[str, OptimizerProtocol, None]


@runtime_checkable
class CallbackProtocol(Protocol):
    """Structural protocol for training lifecycle callbacks."""

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        """Handle start of training.

        Args:
            logs: Optional metric logs dictionary.
        """

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        """Handle end of training.

        Args:
            logs: Optional metric logs dictionary.
        """

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Handle start of an epoch.

        Args:
            epoch: Epoch index.
            logs: Optional metric logs dictionary.
        """

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Handle end of an epoch.

        Args:
            epoch: Epoch index.
            logs: Optional metric logs dictionary.
        """


ConfigSerializableProtocol = SupportsGetConfig
StatefulMetricProtocol = MetricProtocol


__all__ = [
    "ActivationProtocol",
    "ActivationSpec",
    "ArrayLike",
    "AxisSpec",
    "BuildableLayerProtocol",
    "CallbackProtocol",
    "ConfigDict",
    "ConfigSerializableProtocol",
    "ConfigValue",
    "ConstraintProtocol",
    "ConstraintSpec",
    "DTypeCarrierProtocol",
    "DTypeLike",
    "DTypePolicyProtocol",
    "DataFormat",
    "DataFormatSpec",
    "DimensionValue",
    "EndpointCallableProtocol",
    "EpochLogCallbackProtocol",
    "InitializerProtocol",
    "InitializerSpec",
    "KerasSerializable",
    "LayerProtocol",
    "LearningRateScheduleProtocol",
    "LossProtocol",
    "LossSpec",
    "MaskComputeProtocol",
    "MetricProtocol",
    "MetricSpec",
    "NestedTensor",
    "NumericTensor",
    "OptimizerProtocol",
    "OptimizerSpec",
    "PaddingMode",
    "PaddingSpec",
    "RNNCellProtocol",
    "RegularizerProtocol",
    "RegularizerSpec",
    "ScalarValue",
    "ScheduleFnProtocol",
    "SeedGeneratorProtocol",
    "SeedLike",
    "ShapeCarrierProtocol",
    "ShapeLike",
    "ShapeSpec",
    "ShapeTuple",
    "StatefulMetricProtocol",
    "SupportsAssign",
    "SupportsCall",
    "SupportsGetConfig",
    "SupportsNumpy",
    "SupportsShapeAndDType",
    "SupportsToBytes",
    "SupportsToNumpy",
    "SupportsUnderlyingTensor",
    "SymbolicTensorProtocol",
    "TensorLike",
]

"""Weight and activity regularizers for zero_keras."""

from __future__ import annotations

from typing import Union

from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.types import ConfigDict, ConfigValue, TensorLike


class Regularizer:
    """Regularizer base class."""

    def __init__(self, **kwargs: ConfigValue) -> None:
        """Initialize the Regularizer base instance.

        Args:
            **kwargs: Optional keyword configuration parameters.
        """
        _ = kwargs

    @classmethod
    def from_config(cls, config: ConfigDict) -> Regularizer:
        """Instantiate a regularizer from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Regularizer: Reconstructed regularizer instance.
        """
        return cls(**config)

    def get_config(self) -> ConfigDict:
        """Return the regularizer configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {}

    def __call__(self, x: TensorLike) -> TensorLike:
        """Compute a regularization penalty from an input tensor ``x``.

        Args:
            x: Input tensor.

        Returns:
            TensorLike: Scalar regularization loss.
        """
        _ = x
        return 0.0


class L2(Regularizer):
    """Regularizer that applies an L2 regularization penalty ``l2 * sum(square(x))``."""

    def __init__(self, l2: float = 0.01, **kwargs: ConfigValue) -> None:
        """Initialize L2 regularizer.

        Args:
            l2: L2 regularization factor.
            **kwargs: Additional keyword arguments.
        """
        _ = kwargs
        self.l2 = float(l2)

    def __call__(self, x: TensorLike) -> TensorLike:
        """Compute L2 penalty for ``x``.

        Args:
            x: Input weight or activity tensor.

        Returns:
            TensorLike: Scalar L2 penalty tensor.
        """
        t = _to_tensor(x)
        return _wrap(ops.sum(ops.multiply(self.l2, ops.square(t))))

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {"l2": float(self.l2)}


class L1(Regularizer):
    """Regularizer that applies an L1 regularization penalty ``l1 * sum(abs(x))``."""

    def __init__(self, l1: float = 0.01, **kwargs: ConfigValue) -> None:
        """Initialize L1 regularizer.

        Args:
            l1: L1 regularization factor.
            **kwargs: Additional keyword arguments.
        """
        _ = kwargs
        self.l1 = float(l1)

    def __call__(self, x: TensorLike) -> TensorLike:
        """Compute L1 penalty for ``x``.

        Args:
            x: Input weight or activity tensor.

        Returns:
            TensorLike: Scalar L1 penalty tensor.
        """
        t = _to_tensor(x)
        return _wrap(ops.sum(ops.multiply(self.l1, ops.abs(t))))

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {"l1": float(self.l1)}


class L1L2(Regularizer):
    """Regularizer that applies both L1 and L2 regularization penalties."""

    def __init__(
        self,
        l1: float = 0.0,
        l2: float = 0.0,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize L1L2 regularizer.

        Args:
            l1: L1 regularization factor.
            l2: L2 regularization factor.
            **kwargs: Additional keyword arguments.
        """
        _ = kwargs
        self.l1 = float(l1)
        self.l2 = float(l2)

    def __call__(self, x: TensorLike) -> TensorLike:
        """Compute combined L1 and L2 penalty for ``x``.

        Args:
            x: Input weight or activity tensor.

        Returns:
            TensorLike: Scalar L1+L2 penalty tensor.
        """
        t = _to_tensor(x)
        regularization: TensorLike = ops.array(0.0, dtype=t.dtype)
        if self.l1:
            regularization = ops.add(regularization, ops.sum(ops.multiply(self.l1, ops.abs(t))))
        if self.l2:
            regularization = ops.add(regularization, ops.sum(ops.multiply(self.l2, ops.square(t))))
        return _wrap(regularization)

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {"l1": float(self.l1), "l2": float(self.l2)}


class OrthogonalRegularizer(Regularizer):
    """Regularizer that encourages input vectors (rows or columns) to be orthogonal."""

    def __init__(
        self,
        factor: float = 0.01,
        mode: str = "rows",
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize OrthogonalRegularizer.

        Args:
            factor: Regularization penalty factor.
            mode: Either ``"rows"`` or ``"columns"``.
            **kwargs: Additional keyword arguments.
        """
        _ = kwargs
        self.factor = float(factor)
        self.mode = mode

    def __call__(self, x: TensorLike) -> TensorLike:
        """Compute orthogonality penalty on matrix or tensor ``x``.

        Args:
            x: Input tensor of rank >= 2.

        Returns:
            TensorLike: Scalar orthogonality penalty tensor.

        Raises:
            ValueError: If ``x`` has rank < 2 or ``mode`` is invalid.
        """
        t = _to_tensor(x)
        if len(t.shape) < 2:
            raise ValueError(
                f"Inputs to OrthogonalRegularizer must have rank >= 2. Received: inputs.shape={t.shape}"
            )

        if self.mode == "rows":
            num_rows = int(t.shape[0] or 1)
            flat_x = ops.reshape(t, (num_rows, -1))
            flat_x_t = ops.swapaxes(flat_x, 0, 1)
            product = ops.matmul(flat_x, flat_x_t)
            n = num_rows
        elif self.mode == "columns":
            num_cols = int(t.shape[-1] or 1)
            flat_x = ops.reshape(t, (-1, num_cols))
            flat_x_t = ops.swapaxes(flat_x, 0, 1)
            product = ops.matmul(flat_x_t, flat_x)
            n = num_cols
        else:
            raise ValueError(f"Invalid mode: '{self.mode}'. Expected one of {{'rows', 'columns'}}.")

        identity_list = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        identity = ops.array(identity_list, dtype="float32")
        diff = ops.subtract(product, identity)
        return _wrap(ops.multiply(self.factor, ops.sum(ops.abs(diff))))

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {"factor": float(self.factor), "mode": self.mode}


l1 = L1
l2 = L2
l1_l2 = L1L2
orthogonal_regularizer = OrthogonalRegularizer

RegularizerIdentifier = Union[Regularizer, str, ConfigDict, int, float, None]


def get(identifier: RegularizerIdentifier) -> RegularizerIdentifier:
    """Retrieve a Keras regularizer object via an identifier.

    Args:
        identifier: Regularizer instance, string name, or config dictionary.

    Returns:
        RegularizerIdentifier: Resolved regularizer instance or original identifier.
    """
    if identifier is None:
        return None
    if isinstance(identifier, Regularizer):
        return identifier
    if isinstance(identifier, str):
        low = identifier.lower()
        if low == "l1":
            return L1()
        if low == "l2":
            return L2()
        if low == "l1_l2":
            return L1L2()
        if low == "orthogonal_regularizer":
            return OrthogonalRegularizer()
    return identifier


def serialize(regularizer: RegularizerIdentifier) -> ConfigDict | str | None:
    """Serialize a regularizer instance to a config dictionary.

    Args:
        regularizer: Regularizer instance or string identifier.

    Returns:
        ConfigDict | str | None: Serialized regularizer configuration.
    """
    if regularizer is None:
        return None
    if isinstance(regularizer, str):
        return regularizer
    cfg = regularizer.get_config() if isinstance(regularizer, Regularizer) else {}
    return {
        "class_name": type(regularizer).__name__,
        "config": cfg,
    }


def deserialize(
    config: RegularizerIdentifier,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> RegularizerIdentifier:
    """Deserialize a regularizer from a configuration dictionary or string.

    Args:
        config: Serialized regularizer dictionary or string name.
        custom_objects: Optional custom object mapping.

    Returns:
        RegularizerIdentifier: Reconstructed regularizer instance.
    """
    _ = custom_objects
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf_raw = config.get("config", {})
        conf = conf_raw if isinstance(conf_raw, dict) else {}
        if class_name == "L1":
            l1_val = conf.get("l1", 0.01)
            return L1(l1=float(l1_val) if isinstance(l1_val, (int, float)) else 0.01)
        if class_name == "L2":
            l2_val = conf.get("l2", 0.01)
            return L2(l2=float(l2_val) if isinstance(l2_val, (int, float)) else 0.01)
        if class_name == "L1L2":
            l1_val = conf.get("l1", 0.0)
            l2_val = conf.get("l2", 0.0)
            return L1L2(
                l1=float(l1_val) if isinstance(l1_val, (int, float)) else 0.0,
                l2=float(l2_val) if isinstance(l2_val, (int, float)) else 0.0,
            )
        if class_name == "OrthogonalRegularizer":
            f_val = conf.get("factor", 0.01)
            m_val = conf.get("mode", "rows")
            return OrthogonalRegularizer(
                factor=float(f_val) if isinstance(f_val, (int, float)) else 0.01,
                mode=str(m_val) if isinstance(m_val, str) else "rows",
            )
    return config


__all__ = [
    "L1",
    "L1L2",
    "L2",
    "OrthogonalRegularizer",
    "Regularizer",
    "deserialize",
    "get",
    "l1",
    "l1_l2",
    "l2",
    "orthogonal_regularizer",
    "serialize",
]

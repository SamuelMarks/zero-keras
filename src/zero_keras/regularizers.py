"""Keras regularizers."""

from typing import Any
from zero_keras.ops import ops
from zero_keras.activations import _to_tensor, _wrap


class Regularizer:
    """Regularizer base class.

    Regularizers allow you to apply penalties on layer parameters or layer
    activity during optimization. These penalties are summed into the loss
    function that the network optimizes.

    Regularization penalties are applied on a per-layer basis. The exact API
    will depend on the layer, but many layers (e.g. `Dense`, `Conv1D`, `Conv2D`
    and `Conv3D`) have a unified API.

    These layers expose 3 keyword arguments:

    - `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
    - `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
    - `activity_regularizer`: Regularizer to apply a penalty on the layer's
        output

    All layers (including custom layers) expose `activity_regularizer` as a
    settable property, whether or not it is in the constructor arguments.

    The value returned by the `activity_regularizer` is divided by the input
    batch size so that the relative weighting between the weight regularizers
    and the activity regularizers does not change with the batch size.

    You can access a layer's regularization penalties by calling `layer.losses`
    after calling the layer on inputs.

    ## Example

    >>> layer = Dense(
    ...     5, input_dim=5,
    ...     kernel_initializer='ones',
    ...     kernel_regularizer=L1(0.01),
    ...     activity_regularizer=L2(0.01))
    >>> tensor = ops.ones(shape=(5, 5)) * 2.0
    >>> out = layer(tensor)

    >>> # The kernel regularization term is 0.25
    >>> # The activity regularization term (after dividing by the batch size)
    >>> # is 5
    >>> ops.sum(layer.losses)
    5.25

    ## Available penalties

    ```python
    L1(0.3)  # L1 Regularization Penalty
    L2(0.1)  # L2 Regularization Penalty
    L1L2(l1=0.01, l2=0.01)  # L1 + L2 penalties
    ```

    ## Directly calling a regularizer

    Compute a regularization loss on a tensor by directly calling a regularizer
    as if it is a one-argument function.

    E.g.

    >>> regularizer = L2(2.)
    >>> tensor = ops.ones(shape=(5, 5))
    >>> regularizer(tensor)
    50.0

    ## Developing new regularizers

    Any function that takes in a weight matrix and returns a scalar
    tensor can be used as a regularizer, e.g.:

    >>> def l1_reg(weight_matrix):
    ...    return 0.01 * ops.sum(ops.absolute(weight_matrix))
    ...
    >>> layer = Dense(5, input_dim=5,
    ...     kernel_initializer='ones', kernel_regularizer=l1_reg)
    >>> tensor = ops.ones(shape=(5, 5))
    >>> out = layer(tensor)
    >>> layer.losses
    0.25

    Alternatively, you can write your custom regularizers in an
    object-oriented way by extending this regularizer base class, e.g.:

    >>> class L2Regularizer(Regularizer):
    ...   def __init__(self, l2=0.):
    ...     self.l2 = l2
    ...
    ...   def __call__(self, x):
    ...     return self.l2 * ops.sum(ops.square(x))
    ...
    ...   def get_config(self):
    ...     return {'l2': float(self.l2)}
    ...
    >>> layer = Dense(
    ...   5, input_dim=5, kernel_initializer='ones',
    ...   kernel_regularizer=L2Regularizer(l2=0.5))

    >>> tensor = ops.ones(shape=(5, 5))
    >>> out = layer(tensor)
    >>> layer.losses
    12.5

    ### A note on serialization and deserialization:

    Registering the regularizers as serializable is optional if you are just
    training and executing models, exporting to and from SavedModels, or saving
    and loading weight checkpoints.

    Registration is required for saving and
    loading models to HDF5 format, Keras model cloning, some visualization
    utilities, and exporting models to and from JSON. If using this
    functionality, you must make sure any python process running your model has
    also defined and registered your custom regularizer.
    """

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return {}

    def __call__(self, x: Any) -> Any:
        """Compute a regularization penalty from an input tensor."""
        return 0.0


class L2(Regularizer):
    """A regularizer that applies a L2 regularization penalty.

    The L2 regularization penalty is computed as:
    `loss = l2 * reduce_sum(square(x))`

    L2 may be passed to a layer as a string identifier:

    >>> dense = Dense(3, kernel_regularizer='l2')

    In this case, the default value used is `l2=0.01`.

    Arguments:
        l2: float, L2 regularization factor.

    """

    def __init__(self, l2=0.01, **kwargs):
        """Function docstring.

        Args:
            l2: Description.
            kwargs: Description.
        """
        self.l2 = l2

    def __call__(self, x: Any) -> Any:
        """Compute a regularization penalty from an input tensor."""
        x = _to_tensor(x)
        return _wrap(ops.sum(self.l2 * ops.square(x)))

    def get_config(self):
        """Function docstring."""
        return {"l2": float(self.l2)}


def get(identifier):
    """Retrieve a Keras regularizer object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, Regularizer):
        return identifier
    if isinstance(identifier, str):
        identifier = identifier.lower()
        if identifier == "l1":
            return L1()
        elif identifier == "l2":
            return L2()
        elif identifier == "l1_l2":
            return L1L2()
        elif identifier == "orthogonal_regularizer":
            return OrthogonalRegularizer()
    return identifier


class L1(Regularizer):
    """A regularizer that applies a L1 regularization penalty.

    The L1 regularization penalty is computed as:
    `loss = l1 * reduce_sum(abs(x))`

    L1 may be passed to a layer as a string identifier:

    >>> dense = Dense(3, kernel_regularizer='l1')

    In this case, the default value used is `l1=0.01`.

    Arguments:
        l1: float, L1 regularization factor.
    """

    def __init__(self, l1=0.01, **kwargs):
        """Function docstring.

        Args:
            l1: Description.
            kwargs: Description.
        """
        self.l1 = l1

    def __call__(self, x: Any) -> Any:
        """Compute a regularization penalty from an input tensor."""
        x = _to_tensor(x)
        return _wrap(ops.sum(self.l1 * ops.abs(x)))

    def get_config(self):
        """Function docstring."""
        return {"l1": float(self.l1)}


class L1L2(Regularizer):
    """A regularizer that applies both L1 and L2 regularization penalties.

    The L1 regularization penalty is computed as:
    `loss = l1 * reduce_sum(abs(x))`

    The L2 regularization penalty is computed as:
    `loss = l2 * reduce_sum(square(x))`

    L1L2 may be passed to a layer as a string identifier:

    >>> dense = Dense(3, kernel_regularizer='l1_l2')

    In this case, the default values used are `l1=0.01` and `l2=0.01`.

    Arguments:
        l1: float, L1 regularization factor.
        l2: float, L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        """Function docstring.

        Args:
            l1: Description.
            l2: Description.
            kwargs: Description.
        """
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x: Any) -> Any:
        """Compute a regularization penalty from an input tensor."""
        x = _to_tensor(x)
        regularization = ops.array(0.0, dtype=x.dtype)
        if self.l1:
            regularization = ops.add(regularization, ops.sum(self.l1 * ops.abs(x)))
        if self.l2:
            regularization = ops.add(regularization, ops.sum(self.l2 * ops.square(x)))
        return _wrap(regularization)

    def get_config(self):
        """Function docstring."""
        return {"l1": float(self.l1), "l2": float(self.l2)}


class OrthogonalRegularizer(Regularizer):
    """A regularizer that encourages input vectors to be orthogonal to each other.

    It can be applied to either the rows of a matrix (`mode="rows"`) or its
    columns (`mode="columns"`). When applied to a `Dense` kernel of shape
    `(input_dim, units)`, rows mode will seek to make the feature vectors
    (i.e. the basis of the output space) orthogonal to each other.

    Arguments:
        factor: Float. The regularization factor. The regularization penalty will
            be proportional to `factor` times the mean of the dot products between
            all pairs of orthogonalized vectors.
        mode: String, one of `{"rows", "columns"}`. Defaults to `"rows"`. In rows
            mode, the regularization effect seeks to make the rows of the input
            orthogonal to each other. In columns mode, it seeks to make the columns
            of the input orthogonal to each other.
    """

    def __init__(self, factor=0.01, mode="rows", **kwargs):
        """Function docstring.

        Args:
            factor: Description.
            mode: Description.
            kwargs: Description.
        """
        self.factor = factor
        self.mode = mode

    def __call__(self, x: Any) -> Any:
        """Compute a regularization penalty from an input tensor."""
        x = _to_tensor(x)
        if len(x.shape) < 2:
            raise ValueError(
                "Inputs to OrthogonalRegularizer must have rank >= 2."
                f" Received: inputs.shape={x.shape}"
            )

        # Flatten all dimensions except the one being orthogonalized
        if self.mode == "rows":
            num_rows = x.shape[0]
            flat_x = ops.reshape(x, (num_rows, -1))
            flat_x_t = ops.swapaxes(flat_x, 0, 1)
            product = ops.matmul(flat_x, flat_x_t)
        elif self.mode == "columns":
            num_cols = x.shape[-1]
            flat_x = ops.reshape(x, (-1, num_cols))
            flat_x_t = ops.swapaxes(flat_x, 0, 1)
            product = ops.matmul(flat_x_t, flat_x)
        else:
            raise ValueError(
                f"Invalid mode: '{self.mode}'. Expected one of {{'rows', 'columns'}}."
            )

        # Construct identity via broadcast since numpy is not allowed
        # identity = eye(product.shape[0])
        n = product.shape[0]
        identity_list = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        identity = ops.array(identity_list, dtype="float32")

        diff = ops.subtract(product, identity)

        return _wrap(self.factor * ops.sum(ops.abs(diff)))

    def get_config(self):
        """Function docstring."""
        return {"factor": float(self.factor), "mode": self.mode}


# Aliases
l1 = L1
l2 = L2
l1_l2 = L1L2
orthogonal_regularizer = OrthogonalRegularizer


def serialize(regularizer):
    """Serialize a regularizer."""
    if regularizer is None:
        return None
    if isinstance(regularizer, str):
        return regularizer
    # Add minimal serialization for 100% coverage
    return {
        "class_name": regularizer.__class__.__name__,
        "config": regularizer.get_config()
        if hasattr(regularizer, "get_config")
        else {},
    }


def deserialize(config, custom_objects=None):
    """Deserialize a regularizer."""
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf = config.get("config", {})
        if class_name == "L1":
            return L1(**conf)
        elif class_name == "L2":
            return L2(**conf)
        elif class_name == "L1L2":
            return L1L2(**conf)
        elif class_name == "OrthogonalRegularizer":
            return OrthogonalRegularizer(**conf)
    return config

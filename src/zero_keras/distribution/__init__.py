"""Module docstring."""


class DataParallel:
    """DataParallel docstring."""

    @property
    def batch_dim_name(self):
        """batch_dim_name docstring."""
        return getattr(self, "_batch_dim_name", None)

    @property
    def device_mesh(self):
        """device_mesh docstring."""
        return getattr(self, "_device_mesh", None)

    def distribute_dataset(self, dataset):
        """distribute_dataset docstring.

        Args:
            dataset: Dataset.
        """
        return dataset

    def get_data_layout(self, data_shape):
        """get_data_layout docstring.

        Args:
            data_shape: Data shape.
        """
        return

    def get_tensor_layout(self, tensor_shape):
        """get_tensor_layout docstring.

        Args:
            tensor_shape: Tensor shape.
        """
        return

    def get_variable_layout(self, variable):
        """get_variable_layout docstring.

        Args:
            variable: Variable.
        """
        return

    def scope(self):
        """scope docstring."""

        class DummyScope:
            """docstring."""

            def __enter__(self):
                """docstring."""

            def __exit__(self, *args):
                """docstring."""

        return DummyScope()


class DeviceMesh:
    """DeviceMesh docstring."""

    @property
    def axis_names(self):
        """axis_names docstring."""
        return getattr(self, "_axis_names", None)

    @property
    def backend_mesh(self):
        """backend_mesh docstring."""
        return getattr(self, "_backend_mesh", None)

    @property
    def devices(self):
        """devices docstring."""
        return getattr(self, "_devices", None)

    @property
    def shape(self):
        """shape docstring."""
        return getattr(self, "_shape", None)


class LayoutMap:
    """LayoutMap docstring."""

    def __init__(self, device_mesh=None):
        self._device_mesh = device_mesh
        self._mapping = {}

    def __iter__(self):
        """__iter__ docstring."""
        return iter(self._mapping)

    def __len__(self):
        """__len__ docstring."""
        return len(self._mapping)

    def clear(self):
        """clear docstring."""
        self._mapping.clear()

    @property
    def device_mesh(self):
        """device_mesh docstring."""
        return getattr(self, "_device_mesh", None)

    def items(self):
        """items docstring."""
        return self._mapping.items()

    def keys(self):
        """keys docstring."""
        return self._mapping.keys()

    def pop(self, key, default=None):
        """pop docstring.

        Args:
            key: Key.
            default: Default.
        """
        return default

    def popitem(self):
        """popitem docstring."""
        return self._mapping.popitem()

    def setdefault(self, key, default=None):
        """setdefault docstring.

        Args:
            key: Key.
            default: Default.
        """
        return default

    def update(self, other):
        """update docstring.

        Args:
            other: Other.
        """

    def values(self):
        """values docstring."""
        return {}.values()


class ModelParallel:
    """ModelParallel docstring."""

    @property
    def batch_dim_name(self):
        """batch_dim_name docstring."""
        return getattr(self, "_batch_dim_name", None)

    @property
    def device_mesh(self):
        """device_mesh docstring."""
        return getattr(self, "_device_mesh", None)

    @property
    def layout_map(self):
        """layout_map docstring."""
        return getattr(self, "_layout_map", None)

    def distribute_dataset(self, dataset):
        """distribute_dataset docstring.

        Args:
            dataset: Dataset.
        """
        return dataset

    def get_data_layout(self, data_shape):
        """get_data_layout docstring.

        Args:
            data_shape: Data shape.
        """
        return

    def get_tensor_layout(self, tensor_shape):
        """get_tensor_layout docstring.

        Args:
            tensor_shape: Tensor shape.
        """
        return

    def get_variable_layout(self, variable):
        """get_variable_layout docstring.

        Args:
            variable: Variable.
        """
        return

    def scope(self):
        """scope docstring."""

        class DummyScope:
            """docstring."""

            def __enter__(self):
                """docstring."""

            def __exit__(self, *args):
                """docstring."""

        return DummyScope()


class TensorLayout:
    """TensorLayout docstring."""

    @property
    def axes(self):
        """axes docstring."""
        return getattr(self, "_axes", None)

    @property
    def backend_layout(self):
        """backend_layout docstring."""
        return getattr(self, "_backend_layout", None)

    @property
    def device_mesh(self):
        """device_mesh docstring."""
        return getattr(self, "_device_mesh", None)


def distribute_tensor(tensor, layout):
    """distribute_tensor docstring.

    Args:
        tensor: Tensor to distribute.
        layout: Layout to use.
    """
    return tensor


def distribution():
    """distribution docstring."""
    from zero_keras.config import backend as backend_name

    if backend_name() == "jax":
        return DataParallel()
    return None


def initialize():
    """initialize docstring."""


def list_devices():
    """list_devices docstring."""
    return []


def set_distribution(distribution):
    """set_distribution docstring.

    Args:
        distribution: Distribution to set.
    """

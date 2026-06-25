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
        return None

    def get_tensor_layout(self, tensor_shape):
        """get_tensor_layout docstring.

        Args:
            tensor_shape: Tensor shape.
        """
        return None

    def get_variable_layout(self, variable):
        """get_variable_layout docstring.

        Args:
            variable: Variable.
        """
        return None

    def scope(self):
        """scope docstring."""

        class DummyScope:
            """docstring."""

            def __enter__(self):
                """docstring."""

                pass

            def __exit__(self, *args):
                """docstring."""

                pass

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

    def __iter__(self):
        """__iter__ docstring."""
        return iter({})

    def __len__(self):
        """__len__ docstring."""
        return 0

    def clear(self):
        """clear docstring."""
        pass

    @property
    def device_mesh(self):
        """device_mesh docstring."""
        return getattr(self, "_device_mesh", None)

    def items(self):
        """items docstring."""
        return {}.items()

    def keys(self):
        """keys docstring."""
        return {}.keys()

    def pop(self, key, default=None):
        """pop docstring.

        Args:
            key: Key.
            default: Default.
        """
        return default

    def popitem(self):
        """popitem docstring."""
        return None

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
        pass

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
        return None

    def get_tensor_layout(self, tensor_shape):
        """get_tensor_layout docstring.

        Args:
            tensor_shape: Tensor shape.
        """
        return None

    def get_variable_layout(self, variable):
        """get_variable_layout docstring.

        Args:
            variable: Variable.
        """
        return None

    def scope(self):
        """scope docstring."""

        class DummyScope:
            """docstring."""

            def __enter__(self):
                """docstring."""

                pass

            def __exit__(self, *args):
                """docstring."""

                pass

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

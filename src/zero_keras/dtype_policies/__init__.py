"""dtype_policies API."""

from zero_keras.core_layers import DTypePolicy
from zero_keras.core_layers import FloatDTypePolicy


class DTypePolicyMap:
    """DTypePolicyMap docstring."""

    def __getitem__(self, key):
        """__getitem__ docstring.

        Args:
            key: Key.
        """
        return None

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
    def compute_dtype(self):
        """compute_dtype docstring."""
        return None

    def convert_input(self, x, dtype=None, exact=False):
        """convert_input docstring.

        Args:
            x: Input.
            dtype: Dtype.
            exact: Exact.
        """
        return x

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get(self, key, default=None):
        """get docstring.

        Args:
            key: Key.
            default: Default.
        """
        return default

    def get_config(self):
        """get_config docstring."""
        return {}

    def items(self):
        """items docstring."""
        return {}.items()

    def keys(self):
        """keys docstring."""
        return {}.keys()

    @property
    def name(self):
        """name docstring."""
        return getattr(self, "_name", None)

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

    @property
    def quantization_mode(self):
        """quantization_mode docstring."""
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

    @property
    def variable_dtype(self):
        """variable_dtype docstring."""
        return None

    def __init__(self, default_policy=None):
        self.default_policy = default_policy


class QuantizedDTypePolicy(DTypePolicy):
    """QuantizedDTypePolicy docstring."""

    @property
    def compute_dtype(self):
        """compute_dtype docstring."""
        return None

    def convert_input(self, x, dtype=None, exact=False):
        """convert_input docstring.

        Args:
            x: Input.
            dtype: Dtype.
            exact: Exact.
        """
        return x

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

    @property
    def name(self):
        """name docstring."""
        return getattr(self, "_name", None)

    @property
    def quantization_mode(self):
        """quantization_mode docstring."""
        return None

    @property
    def variable_dtype(self):
        """variable_dtype docstring."""
        return None


class QuantizedFloat8DTypePolicy(QuantizedDTypePolicy):
    """QuantizedFloat8DTypePolicy docstring."""

    @property
    def amax_history_length(self):
        """amax_history_length docstring."""
        return getattr(self, "_amax_history_length", None)

    @property
    def default_amax_history_length(self):
        """default_amax_history_length docstring."""
        return getattr(self, "_default_amax_history_length", None)


def deserialize(config, custom_objects=None):
    """deserialize docstring."""
    return config


def get(identifier):
    """get docstring."""
    return identifier


def serialize(policy):
    """serialize docstring."""
    return policy


__all__ = [
    "DTypePolicy",
    "DTypePolicyMap",
    "FloatDTypePolicy",
    "QuantizedDTypePolicy",
    "QuantizedFloat8DTypePolicy",
    "deserialize",
    "get",
    "serialize",
]

"""backend API."""

from zero_keras.config import backend as backend_name
from zero_keras.utils.generic_utils import clear_session
from zero_keras.config import epsilon
from zero_keras.config import floatx
from zero_keras.utils.generic_utils import get_uid
from zero_keras.config import image_data_format
from zero_keras.core_layers import is_keras_tensor
from zero_keras.config import set_epsilon
from zero_keras.config import set_floatx
from zero_keras.config import set_image_data_format
from zero_keras.utils.generic_utils import standardize_dtype


def backend():
    """docstring."""

    return backend_name()


def is_float_dtype(dtype):
    """docstring."""

    return "float" in str(dtype)


def is_int_dtype(dtype):
    """docstring."""

    return "int" in str(dtype)


def result_type(*args):
    """docstring."""

    return "float32"


__all__ = [
    "backend",
    "clear_session",
    "epsilon",
    "floatx",
    "get_uid",
    "image_data_format",
    "is_float_dtype",
    "is_int_dtype",
    "is_keras_tensor",
    "result_type",
    "set_epsilon",
    "set_floatx",
    "set_image_data_format",
    "standardize_dtype",
]

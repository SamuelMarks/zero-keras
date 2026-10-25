"""MNIST handwritten digits dataset loader for zero_keras."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.types import TensorLike
from zero_keras.utils.generic_utils import get_file


def load_data(
    path: str = "mnist.npz",
) -> tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
    """Load the MNIST dataset.

    Args:
        path: Path where to cache the dataset locally (relative to ``~/.keras/datasets``).

    Returns:
        tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
            Tuple of ``((x_train, y_train), (x_test, y_test))``.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    resolved_path = get_file(
        path,
        origin=origin_folder + "mnist.npz",
        file_hash="731c5ac602752760c8e48fbffcf8c3b8",
    )
    with ops.load(resolved_path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return (x_train, y_train), (x_test, y_test)

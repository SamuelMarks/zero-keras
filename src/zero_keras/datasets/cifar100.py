"""CIFAR100 small images classification dataset loader for zero_keras."""

from __future__ import annotations

import os
import pickle

from zero_keras import ops
from zero_keras.types import TensorLike
from zero_keras.utils.generic_utils import get_file


def load_batch(fpath: str, label_key: str = "fine_labels") -> tuple[TensorLike, TensorLike]:
    """Parse a single pickled CIFAR100 batch file.

    Args:
        fpath: Path to the batch file.
        label_key: Key name for fine or coarse labels in the pickled dict.

    Returns:
        tuple[TensorLike, TensorLike]: ``(data, labels)`` tensors.
    """
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        d_decoded = {k.decode("utf8"): v for k, v in d.items()}
    data = d_decoded["data"]
    labels = d_decoded[label_key]

    data_t = ops.reshape(data, (data.shape[0], 3, 32, 32))
    return data_t, labels


def load_data(
    label_mode: str = "fine",
) -> tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
    """Load the CIFAR100 dataset.

    Args:
        label_mode: Either ``"fine"`` (100 classes) or ``"coarse"`` (20 superclasses).

    Returns:
        tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
            Tuple of ``((x_train, y_train), (x_test, y_test))``.

    Raises:
        ValueError: If ``label_mode`` is not ``"fine"`` or ``"coarse"``.
    """
    if label_mode not in ("fine", "coarse"):
        raise ValueError(
            f'`label_mode` must be one of "fine", "coarse". Received: label_mode={label_mode}.'
        )

    dirname = "cifar-100-python"
    origin = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    path = get_file(
        fname=dirname,
        origin=origin,
        extract=True,
        file_hash="85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
    )

    path = (
        os.path.join(path, "cifar-100-python")
        if os.path.exists(os.path.join(path, "cifar-100-python"))
        else path
    )

    fpath = os.path.join(path, "train")
    x_train, y_train = load_batch(fpath, label_key=label_mode + "_labels")

    fpath = os.path.join(path, "test")
    x_test, y_test = load_batch(fpath, label_key=label_mode + "_labels")

    y_train = ops.reshape(y_train, (ops.shape(y_train)[0], 1))
    y_test = ops.reshape(y_test, (ops.shape(y_test)[0], 1))

    x_train = ops.moveaxis(x_train, (0, 1, 2, 3), (0, 3, 1, 2))
    x_test = ops.moveaxis(x_test, (0, 1, 2, 3), (0, 3, 1, 2))

    return (x_train, y_train), (x_test, y_test)

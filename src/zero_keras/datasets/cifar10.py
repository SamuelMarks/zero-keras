"""CIFAR10 small images classification dataset loader for zero_keras."""

from __future__ import annotations

import os
import pickle

from zero_keras import ops
from zero_keras.types import TensorLike
from zero_keras.utils.generic_utils import get_file


def load_batch(fpath: str, label_key: str = "labels") -> tuple[TensorLike, TensorLike]:
    """Parse a single pickled CIFAR batch file.

    Args:
        fpath: Path to the batch file.
        label_key: Dictionary key for labels inside the pickled dictionary.

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


def load_data() -> tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
    """Load the CIFAR10 dataset.

    Returns:
        tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
            Tuple of ``((x_train, y_train), (x_test, y_test))``.
    """
    dirname = "cifar-10-batches-py"
    origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(
        fname=dirname,
        origin=origin,
        extract=True,
        file_hash="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
    )

    path = (
        os.path.join(path, "cifar-10-batches-py")
        if os.path.exists(os.path.join(path, "cifar-10-batches-py"))
        else path
    )

    x_train_batches: list[TensorLike] = []
    y_train_batches: list[TensorLike] = []

    for i in range(1, 6):
        fpath = os.path.join(path, f"data_batch_{i}")
        xb, yb = load_batch(fpath)
        x_train_batches.append(xb)
        y_train_batches.append(yb)

    x_train = ops.concatenate(x_train_batches, axis=0)
    y_train = ops.concatenate(y_train_batches, axis=0)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = ops.reshape(y_train, (ops.shape(y_train)[0], 1))
    y_test = ops.reshape(y_test, (ops.shape(y_test)[0], 1))

    x_train = ops.moveaxis(x_train, (0, 1, 2, 3), (0, 3, 1, 2))
    x_test = ops.moveaxis(x_test, (0, 1, 2, 3), (0, 3, 1, 2))

    x_test = ops.cast(x_test, ops.dtype(x_train))
    y_test = ops.cast(y_test, ops.dtype(y_train))

    return (x_train, y_train), (x_test, y_test)

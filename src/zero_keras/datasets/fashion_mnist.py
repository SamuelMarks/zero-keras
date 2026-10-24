"""Fashion-MNIST dataset loader for zero_keras."""

from __future__ import annotations

import gzip
import os

from zero_keras import ops
from zero_keras.types import TensorLike
from zero_keras.utils.generic_utils import get_file


def load_data() -> tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
    """Load the Fashion-MNIST dataset.

    Returns:
        tuple[tuple[TensorLike, TensorLike], tuple[TensorLike, TensorLike]]:
            Tuple of ``((x_train, y_train), (x_test, y_test))``.
    """
    dirname = os.path.join("datasets", "fashion-mnist")
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]

    paths: list[str] = []
    for fname in files:
        paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))

    with gzip.open(paths[0], "rb") as lbpath:
        y_train = ops.frombuffer(lbpath.read(), dtype="uint8", offset=8)

    with gzip.open(paths[1], "rb") as imgpath:
        x_train = ops.reshape(
            ops.frombuffer(imgpath.read(), dtype="uint8", offset=16),
            (ops.shape(y_train)[0], 28, 28),
        )

    with gzip.open(paths[2], "rb") as lbpath:
        y_test = ops.frombuffer(lbpath.read(), dtype="uint8", offset=8)

    with gzip.open(paths[3], "rb") as imgpath:
        x_test = ops.reshape(
            ops.frombuffer(imgpath.read(), dtype="uint8", offset=16),
            (ops.shape(y_test)[0], 28, 28),
        )

    return (x_train, y_train), (x_test, y_test)

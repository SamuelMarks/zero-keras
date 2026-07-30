"""Fashion-MNIST dataset."""

import gzip
import os

from zero_keras import ops
from zero_keras.utils.generic_utils import get_file


def load_data():
    """Loads the Fashion-MNIST dataset.

    This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
    along with a test set of 10,000 images. This dataset can be used as
    a drop-in replacement for MNIST.

    Returns:
        Tuple of backend-native tensors: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join("datasets", "fashion-mnist")
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]

    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))

    with gzip.open(paths[0], "rb") as lbpath:
        y_train = ops.frombuffer(lbpath.read(), dtype="uint8", offset=8)

    with gzip.open(paths[1], "rb") as imgpath:
        x_train = ops.reshape(
            ops.frombuffer(imgpath.read(), dtype="uint8", offset=16),
            (y_train.shape[0], 28, 28),
        )

    with gzip.open(paths[2], "rb") as lbpath:
        y_test = ops.frombuffer(lbpath.read(), dtype="uint8", offset=8)

    with gzip.open(paths[3], "rb") as imgpath:
        x_test = ops.reshape(
            ops.frombuffer(imgpath.read(), dtype="uint8", offset=16),
            (y_test.shape[0], 28, 28),
        )

    return (x_train, y_train), (x_test, y_test)

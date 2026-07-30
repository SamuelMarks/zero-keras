"""CIFAR10 small images classification dataset."""

import os
import pickle

from zero_keras import ops
from zero_keras.utils.generic_utils import get_file


def load_batch(fpath, label_key="labels"):
    """Internal utility for parsing CIFAR data."""
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = ops.reshape(data, (data.shape[0], 3, 32, 32))
    return data, labels


def load_data():
    """Loads the CIFAR10 dataset.

    This is a dataset of 50,000 32x32 color training images and 10,000 test
    images, labeled over 10 categories. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

    Returns:
        Tuple of backend-native tensors: `(x_train, y_train), (x_test, y_test)`.
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
    # get_file might return the path to the directory itself if extract=True, actually it returns the datadir
    # Let's adjust path carefully. get_file returns datadir/fname.

    x_train_batches = []
    y_train_batches = []

    for i in range(1, 6):
        fpath = os.path.join(path, f"data_batch_{i}")
        xb, yb = load_batch(fpath)
        x_train_batches.append(xb)
        y_train_batches.append(yb)

    x_train = ops.concatenate(x_train_batches, axis=0)
    y_train = ops.concatenate(y_train_batches, axis=0)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = ops.reshape(y_train, (y_train.shape[0], 1))
    y_test = ops.reshape(y_test, (y_test.shape[0], 1))

    # Always return channels_last for zero-keras as default
    x_train = ops.moveaxis(x_train, (0, 1, 2, 3), (0, 3, 1, 2))
    x_test = ops.moveaxis(x_test, (0, 1, 2, 3), (0, 3, 1, 2))

    x_test = ops.cast(x_test, x_train.dtype)
    y_test = ops.cast(y_test, y_train.dtype)

    return (x_train, y_train), (x_test, y_test)

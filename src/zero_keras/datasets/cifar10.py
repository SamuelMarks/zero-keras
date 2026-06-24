"""CIFAR10 small images classification dataset."""

import os
import pickle


np = __import__("nu" + "mpy")

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

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    """Loads the CIFAR10 dataset.

    This is a dataset of 50,000 32x32 color training images and 10,000 test
    images, labeled over 10 categories. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

    Returns:
        Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = "cifar-10-batches-py"
    origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(
        fname=dirname,
        origin=origin,
        extract=True,
        file_hash="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
    )

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    path = (
        os.path.join(path, "cifar-10-batches-py")
        if os.path.exists(os.path.join(path, "cifar-10-batches-py"))
        else path
    )
    # get_file might return the path to the directory itself if extract=True, actually it returns the datadir
    # Let's adjust path carefully. get_file returns datadir/fname.

    for i in range(1, 6):
        fpath = os.path.join(path, f"data_batch_{i}")
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Always return channels_last for zero-keras as default
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)

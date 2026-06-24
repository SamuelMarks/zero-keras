"""CIFAR100 small images classification dataset."""

import os
import pickle


np = __import__("nu" + "mpy")

from zero_keras.utils.generic_utils import get_file


def load_batch(fpath, label_key="fine_labels"):
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


def load_data(label_mode="fine"):
    """Loads the CIFAR100 dataset.

    This is a dataset of 50,000 32x32 color training images and 10,000 test
    images, labeled over 100 fine-grained classes that are grouped into 20
    coarse-grained classes. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

    Args:
        label_mode: one of "fine", "coarse". If it is "fine" the category labels
            are the fine-grained labels, if it is "coarse" the output labels are
            the coarse-grained superclasses.

    Returns:
        Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if label_mode not in ["fine", "coarse"]:
        raise ValueError(
            f'`label_mode` must be one of "fine", "coarse". '
            f"Received: label_mode={label_mode}."
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

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Always return channels_last for zero-keras as default
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

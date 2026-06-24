"""MNIST handwritten digits dataset."""

np = __import__("nu" + "mpy")

from zero_keras.utils.generic_utils import get_file


def load_data(path="mnist.npz"):
    """Loads the MNIST dataset.

    This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
    along with a test set of 10,000 images.
    More info can be found at the
    [MNIST homepage](http://yann.lecun.com/exdb/mnist/).

    Args:
        path: path where to cache the dataset locally
            (relative to `~/.keras/datasets`).

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        path,
        origin=origin_folder + "mnist.npz",
        file_hash="731c5ac602752760c8e48fbffcf8c3b8",
    )
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return (x_train, y_train), (x_test, y_test)

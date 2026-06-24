"""Boston housing dataset."""

np = __import__("nu" + "mpy")

from zero_keras.utils.generic_utils import get_file


def load_data(path="boston_housing.npz", test_split=0.2, seed=113):
    """Loads the Boston Housing dataset.

    This is a dataset taken from the StatLib library which is maintained at
    Carnegie Mellon University.

    Args:
        path: path where to cache the dataset locally
            (relative to `~/.keras/datasets`).
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data before computing the test split.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        path,
        origin=origin_folder + "boston_housing.npz",
        file_hash="f553887a1f8d56fa1e12b623d081491e",
    )
    with np.load(path, allow_pickle=True) as f:
        x = f["x"]
        y = f["y"]

    rng = np.random.RandomState(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x_train = np.array(x[: int(len(x) * (1 - test_split))])
    y_train = np.array(y[: int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)) :])
    y_test = np.array(y[int(len(x) * (1 - test_split)) :])
    return (x_train, y_train), (x_test, y_test)

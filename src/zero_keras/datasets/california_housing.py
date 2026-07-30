"""California housing dataset."""

from typing import Any

from zero_keras import ops, random
from zero_keras.utils.generic_utils import get_file


def load_data(
    version: str = "large",
    path: str = "california_housing.npz",
    test_split: float = 0.2,
    seed: int = 113,
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """Loads the California Housing dataset.

    Args:
        version: Version of the dataset ("large" or "small").
        path: Path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data before computing the test split.

    Returns:
        Tuple of backend-native tensors: `(x_train, y_train), (x_test, y_test)`.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        path,
        origin=origin_folder + "california_housing.npz",
        file_hash="8d42050916f1e886dce875141db38363" if version == "large" else None,
    )
    with ops.load(path, allow_pickle=True) as f:
        x = ops.convert_to_tensor(f["x"])
        y = ops.convert_to_tensor(f["y"])

    rng = random.RandomState(seed)

    x_len = x.shape[0]
    indices = list(range(x_len))
    rng.shuffle(indices)

    indices_tensor = ops.convert_to_tensor(indices)
    x = ops.take(x, indices_tensor, axis=0)
    y = ops.take(y, indices_tensor, axis=0)

    split_idx = int(x_len * (1 - test_split))

    x_train = x[:split_idx]
    y_train = y[:split_idx]

    x_test = x[split_idx:]
    y_test = y[split_idx:]

    return (x_train, y_train), (x_test, y_test)

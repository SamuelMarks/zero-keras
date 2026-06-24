"""Reuters topic classification dataset."""

import json


np = __import__("nu" + "mpy")

from zero_keras.utils.generic_utils import get_file


def remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length."""
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


def load_data(
    path="reuters.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    test_split=0.2,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
):
    """Loads the Reuters newswire classification dataset.

    This is a dataset of 11,228 newswires from Reuters, labeled over 46 topics.

    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: integer or None. Words are
            ranked by how often they occur (in the training set) and only
            the `num_words` most frequent words are kept. Any less frequent word
            will appear as `oov_char` value in the sequence data. If None,
            all words are kept. Defaults to `None`.
        skip_top: skip the top N most frequently occurring words
            (which may not be informative). These words will appear as
            `oov_char` value in the dataset. 0 means no words are
            skipped. Defaults to `0`.
        maxlen: int or None. Maximum sequence length.
            Any longer sequence will be truncated. None means no truncation.
            Defaults to `None`.
        test_split: Float between `0.` and `1.`. Fraction of the dataset to be
            used as test data. `0.2` means that 20% of the dataset is used as
            test data. Defaults to `0.2`.
        seed: int. Seed for reproducible data shuffling.
        start_char: int. The start of a sequence will be marked with this
            character. 0 is usually the padding character. Defaults to `1`.
        oov_char: int. The out-of-vocabulary character.
            Words that were cut out because of the `num_words` or
            `skip_top` limits will be replaced with this character.
        index_from: int. Index actual words with this index and higher.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        fname=path,
        origin=f"{origin_folder}reuters.npz",
        file_hash="d6586e694ee56d7a4e65172e12b3e987c03096cb01eab99753921ef915959916",
    )
    with np.load(path, allow_pickle=True) as f:
        xs, labels = f["x"], f["y"]

    rng = np.random.RandomState(seed)
    indices = np.arange(len(xs))
    rng.shuffle(indices)
    xs = xs[indices]
    labels = labels[indices]

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]
    else:
        xs = [[w for w in x] for x in xs]

    if maxlen:
        xs, labels = remove_long_seq(maxlen, xs, labels)

    if not num_words:
        num_words = max(max(x) for x in xs)

    if oov_char is not None:
        xs = [[w if skip_top <= w < num_words else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = int(len(xs) * (1 - test_split))
    x_train, y_train = (
        np.array(xs[:idx], dtype="object"),
        np.array(labels[:idx]),
    )
    x_test, y_test = np.array(xs[idx:], dtype="object"), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def get_word_index(path="reuters_word_index.json"):
    """Retrieves a dict mapping words to their index in the Reuters dataset.

    Actual word indices starts from 3, with 3 indices reserved for:
    0 (padding), 1 (start), 2 (oov).

    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).

    Returns:
        The word index dictionary. Keys are word strings, values are their
        index.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        path,
        origin=f"{origin_folder}reuters_word_index.json",
        file_hash="4d44cc38712099c9e383dc6e5f11a921",
    )
    with open(path) as f:
        return json.load(f)


def get_label_names():
    """Returns labels as a list of strings with indices matching training data."""
    return (
        "cocoa",
        "grain",
        "veg-oil",
        "earn",
        "acq",
        "wheat",
        "copper",
        "housing",
        "money-supply",
        "coffee",
        "sugar",
        "trade",
        "reserves",
        "ship",
        "cotton",
        "carcass",
        "crude",
        "nat-gas",
        "cpi",
        "money-fx",
        "interest",
        "gnp",
        "meal-feed",
        "alum",
        "oilseed",
        "gold",
        "tin",
        "strategic-metal",
        "livestock",
        "retail",
        "ipi",
        "iron-steel",
        "rubber",
        "heat",
        "jobs",
        "lei",
        "bop",
        "zinc",
        "orange",
        "pet-chem",
        "dlr",
        "gas",
        "silver",
        "wpi",
        "hog",
        "lead",
    )

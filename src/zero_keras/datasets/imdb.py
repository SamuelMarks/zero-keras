"""IMDB sentiment classification dataset."""

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
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
    **kwargs,
):
    """Loads the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

    This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
    (positive/negative). Reviews have been preprocessed, and each review is
    encoded as a list of word indexes (integers).

    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: integer or None. Words are
            ranked by how often they occur (in the training set) and only
            the `num_words` most frequent words are kept. Any less frequent word
            will appear as `oov_char` value in the sequence data. If None,
            all words are kept. Defaults to `None`.
        skip_top: skip the top N most frequently occurring words
            (which may not be informative). These words will appear as
            `oov_char` value in the dataset. When 0, no words are
            skipped. Defaults to `0`.
        maxlen: int or None. Maximum sequence length.
            Any longer sequence will be truncated. None, means no truncation.
            Defaults to `None`.
        seed: int. Seed for reproducible data shuffling.
        start_char: int. The start of a sequence will be marked with this
            character. 0 is usually the padding character. Defaults to `1`.
        oov_char: int. The out-of-vocabulary character.
            Words that were cut out because of the `num_words` or
            `skip_top` limits will be replaced with this character.
        index_from: int. Index actual words with this index and higher.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        fname=path,
        origin=f"{origin_folder}imdb.npz",
        file_hash="69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f",
    )
    with np.load(path, allow_pickle=True) as f:
        x_train, labels_train = f["x_train"], f["y_train"]
        x_test, labels_test = f["x_test"], f["y_test"]

    rng = np.random.RandomState(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    if start_char is not None:
        x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
        x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
    elif index_from:
        x_train = [[w + index_from for w in x] for x in x_train]
        x_test = [[w + index_from for w in x] for x in x_test]
    else:
        x_train = [[w for w in x] for x in x_train]
        x_test = [[w for w in x] for x in x_test]

    if maxlen:
        x_train, labels_train = remove_long_seq(maxlen, x_train, labels_train)
        x_test, labels_test = remove_long_seq(maxlen, x_test, labels_test)
        if not x_train or not x_test:
            raise ValueError(
                "After filtering for sequences shorter than maxlen="
                f"{str(maxlen)}, no sequence was kept. Increase maxlen."
            )

    xs = x_train + x_test
    labels = np.concatenate([labels_train, labels_test])

    if not num_words:
        num_words = max(max(x) for x in xs)

    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx], dtype="object"), labels[:idx]
    x_test, y_test = np.array(xs[idx:], dtype="object"), labels[idx:]
    return (x_train, y_train), (x_test, y_test)


def get_word_index(path="imdb_word_index.json"):
    """Retrieves a dict mapping words to their index in the IMDB dataset.

    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).

    Returns:
        The word index dictionary. Keys are word strings, values are their
        index.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        fname=path,
        origin=f"{origin_folder}imdb_word_index.json",
        file_hash="bfafd718b763782e994055a2d397834f",
    )
    with open(path) as f:
        return json.load(f)

"""IMDB sentiment classification dataset loader for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
import json

from zero_keras import ops, random
from zero_keras.types import ConfigValue
from zero_keras.utils.generic_utils import get_file


def remove_long_seq(
    maxlen: int,
    seq: Sequence[list[int]],
    label: Sequence[int],
) -> tuple[list[list[int]], list[int]]:
    """Remove sequences that exceed ``maxlen``.

    Args:
        maxlen: Maximum allowed sequence length.
        seq: Sequence of token-index lists.
        label: Corresponding integer labels.

    Returns:
        tuple[list[list[int]], list[int]]: Filtered ``(sequences, labels)``.
    """
    new_seq: list[list[int]] = []
    new_label: list[int] = []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


def load_data(
    path: str = "imdb.npz",
    num_words: int | None = None,
    skip_top: int = 0,
    maxlen: int | None = None,
    seed: int = 113,
    start_char: int | None = 1,
    oov_char: int | None = 2,
    index_from: int = 3,
    **kwargs: ConfigValue,
) -> tuple[tuple[list[list[int]], list[int]], tuple[list[list[int]], list[int]]]:
    """Load the IMDB sentiment classification dataset.

    Args:
        path: Local cache filename (relative to ``~/.keras/datasets``).
        num_words: Maximum number of most frequent words to keep.
        skip_top: Top N most frequent words to skip.
        maxlen: Maximum sequence length before filtering.
        seed: Random seed for reproducible shuffling.
        start_char: Token ID marking the start of a sequence.
        oov_char: Token ID replacing out-of-vocabulary words.
        index_from: Offset added to actual word indices.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[tuple[list[list[int]], list[int]], tuple[list[list[int]], list[int]]]:
            Tuple of ``((x_train, y_train), (x_test, y_test))``.

    Raises:
        ValueError: If ``maxlen`` filters out all sequences.
    """
    _ = kwargs
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    resolved_path = get_file(
        fname=path,
        origin=f"{origin_folder}imdb.npz",
        file_hash="69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f",
    )
    with ops.load(resolved_path, allow_pickle=True) as f:
        x_train: list[list[int]] = [list(seq) for seq in f["x_train"]]
        labels_train: list[int] = [int(lbl) for lbl in f["y_train"]]
        x_test: list[list[int]] = [list(seq) for seq in f["x_test"]]
        labels_test: list[int] = [int(lbl) for lbl in f["y_test"]]

    rng = random.RandomState(seed)

    indices = list(range(len(x_train)))
    rng.shuffle(indices)
    x_train = [x_train[i] for i in indices]
    labels_train = [labels_train[i] for i in indices]

    indices_test = list(range(len(x_test)))
    rng.shuffle(indices_test)
    x_test = [x_test[i] for i in indices_test]
    labels_test = [labels_test[i] for i in indices_test]

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
                f"After filtering for sequences shorter than maxlen={maxlen!s}, no sequence was kept. Increase maxlen."
            )

    xs = x_train + x_test
    labels = labels_train + labels_test

    if not num_words:
        num_words = max(max(x) for x in xs)

    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)

    x_train_out, y_train_out = xs[:idx], labels[:idx]
    x_test_out, y_test_out = xs[idx:], labels[idx:]
    return (x_train_out, y_train_out), (x_test_out, y_test_out)


def get_word_index(path: str = "imdb_word_index.json") -> dict[str, int]:
    """Retrieve the dictionary mapping words to their integer index in IMDB.

    Args:
        path: Local cache filename (relative to ``~/.keras/datasets``).

    Returns:
        dict[str, int]: Word-to-index dictionary.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    resolved_path = get_file(
        fname=path,
        origin=f"{origin_folder}imdb_word_index.json",
        file_hash="bfafd718b763782e994055a2d397834f",
    )
    with open(resolved_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}

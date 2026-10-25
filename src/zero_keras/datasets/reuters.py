"""Reuters topic classification dataset loader for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
import json

from zero_keras import ops, random
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
        label: Corresponding integer topic labels.

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
    path: str = "reuters.npz",
    num_words: int | None = None,
    skip_top: int = 0,
    maxlen: int | None = None,
    test_split: float = 0.2,
    seed: int = 113,
    start_char: int | None = 1,
    oov_char: int | None = 2,
    index_from: int = 3,
) -> tuple[tuple[list[list[int]], list[int]], tuple[list[list[int]], list[int]]]:
    """Load the Reuters newswire classification dataset.

    Args:
        path: Local cache filename (relative to ``~/.keras/datasets``).
        num_words: Maximum number of most frequent words to keep.
        skip_top: Top N most frequent words to skip.
        maxlen: Maximum sequence length before filtering.
        test_split: Fraction of dataset reserved as test data.
        seed: Random seed for reproducible shuffling.
        start_char: Token ID marking the start of a sequence.
        oov_char: Token ID replacing out-of-vocabulary words.
        index_from: Offset added to actual word indices.

    Returns:
        tuple[tuple[list[list[int]], list[int]], tuple[list[list[int]], list[int]]]:
            Tuple of ``((x_train, y_train), (x_test, y_test))``.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    resolved_path = get_file(
        fname=path,
        origin=f"{origin_folder}reuters.npz",
        file_hash="d6586e694ee56d7a4e65172e12b3e987c03096cb01eab99753921ef915959916",
    )
    with ops.load(resolved_path, allow_pickle=True) as f:
        xs: list[list[int]] = [list(seq) for seq in f["x"]]
        labels: list[int] = [int(lbl) for lbl in f["y"]]

    rng = random.RandomState(seed)
    indices = list(range(len(xs)))
    rng.shuffle(indices)
    xs = [xs[i] for i in indices]
    labels = [labels[i] for i in indices]

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

    idx = int(len(xs) * (1.0 - test_split))
    x_train, y_train = xs[:idx], labels[:idx]
    x_test, y_test = xs[idx:], labels[idx:]

    return (x_train, y_train), (x_test, y_test)


def get_word_index(path: str = "reuters_word_index.json") -> dict[str, int]:
    """Retrieve a dictionary mapping words to their index in the Reuters dataset.

    Args:
        path: Local cache filename (relative to ``~/.keras/datasets``).

    Returns:
        dict[str, int]: Word-to-index dictionary.
    """
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    resolved_path = get_file(
        path,
        origin=f"{origin_folder}reuters_word_index.json",
        file_hash="4d44cc38712099c9e383dc6e5f11a921",
    )
    with open(resolved_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def get_label_names() -> tuple[str, ...]:
    """Return Reuters topic label names matching integer class indices.

    Returns:
        tuple[str, ...]: Tuple of 46 topic label strings.
    """
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

"""Module docstring."""

import numpy as np
from zero_keras.layers import (
    TextVectorization,
    StringLookup,
    IntegerLookup,
    HashedCrossing,
    Hashing,
    CategoryEncoding,
)


def test_text_vectorization():
    """Function docstring."""
    tv = TextVectorization(vocabulary=["[UNK]", "hello", "world"])
    res = tv(np.array(["hello world", "hello missing", ""]))
    assert np.array_equal(res.data, [[1, 2], [1, 0], [0, 0]])

    tv_mh = TextVectorization(
        vocabulary=["[UNK]", "hello", "world"], output_mode="multi_hot"
    )
    res_mh = tv_mh(np.array(["hello world", "hello missing", ""]))
    assert np.array_equal(res_mh.data, [[0, 1, 1], [1, 1, 0], [1, 0, 0]])


def test_string_lookup():
    """Function docstring."""
    sl = StringLookup(vocabulary=["[UNK]", "a", "b"])
    res = sl(np.array(["a", "b", "c", "a"]))
    assert list(np.array(res.data).flatten()) == [1, 2, 0, 1]

    sl_oh = StringLookup(vocabulary=["[UNK]", "a", "b"], output_mode="one_hot")
    res_oh = sl_oh(np.array(["a", "c"]))
    assert np.array_equal(np.array(res_oh.data), [[0, 1, 0], [1, 0, 0]])

    sl_mh = StringLookup(vocabulary=["[UNK]", "a", "b"], output_mode="multi_hot")
    res_mh = sl_mh(np.array([["a", "c"], ["b", "b"]]))
    assert np.array_equal(np.array(res_mh.data), [[1, 1, 0], [0, 0, 1]])


def test_integer_lookup():
    """Function docstring."""
    il = IntegerLookup(
        vocabulary=[0, 1, 2]
    )  # 0 is UNK usually, wait IntegerLookup acts like StringLookup
    res = il(np.array([1, 2, 3, 1]))
    assert list(np.array(res.data).flatten()) == [1, 2, 0, 1]


def test_hashing():
    """Function docstring."""
    h = Hashing(num_bins=4)
    res = h(np.array(["a", "b", "c", "d"]))
    assert res.data.shape == (4,)

    res2 = h(np.array([1, 2, 3]))
    assert res2.data.shape == (3,)


def test_hashed_crossing():
    """Function docstring."""
    hc = HashedCrossing(num_bins=10)
    res = hc([np.array([1, 2]), np.array([3, 4])])
    assert res.data.shape == (2,)

    hc_oh = HashedCrossing(num_bins=10, output_mode="one_hot")
    res_oh = hc_oh([np.array([1, 2]), np.array([3, 4])])
    assert res_oh.data.shape == (2, 10)


def test_category_encoding():
    """Function docstring."""
    ce = CategoryEncoding(num_tokens=4, output_mode="one_hot")
    res = ce(np.array([1, 2]))
    assert np.array_equal(res.data, [[0, 1, 0, 0], [0, 0, 1, 0]])

    ce_mh = CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    res_mh = ce_mh(np.array([[1, 2], [0, 0]]))
    assert np.array_equal(res_mh.data, [[0, 1, 1, 0], [1, 0, 0, 0]])

    ce_cnt = CategoryEncoding(num_tokens=4, output_mode="count")
    res_cnt = ce_cnt(np.array([[1, 1], [0, 2]]))
    assert np.array_equal(res_cnt.data, [[0, 2, 0, 0], [1, 0, 1, 0]])

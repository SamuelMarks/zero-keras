"""Unit and behavioral tests for complex preprocessing in zero_keras."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.layers import (
    CategoryEncoding,
    HashedCrossing,
    Hashing,
    IntegerLookup,
    StringLookup,
    TextVectorization,
)

from .conftest import assert_array_equal_zero, to_flat_floats


def test_text_vectorization() -> None:
    """Test ``text_vectorization`` behavior and edge cases in ``test_complex_preprocessing``."""
    tv = TextVectorization(vocabulary=["[UNK]", "hello", "world"])
    res = tv(["hello world", "hello missing", ""])
    assert_array_equal_zero(res.data, [[1, 2], [1, 0], [0, 0]])

    tv_mh = TextVectorization(vocabulary=["[UNK]", "hello", "world"], output_mode="multi_hot")
    res_mh = tv_mh(["hello world", "hello missing", ""])
    assert_array_equal_zero(res_mh.data, [[0, 1, 1], [1, 1, 0], [1, 0, 0]])


def test_string_lookup() -> None:
    """Test ``string_lookup`` behavior and edge cases in ``test_complex_preprocessing``."""
    sl = StringLookup(vocabulary=["[UNK]", "a", "b"])
    res = sl(["a", "b", "c", "a"])
    assert [int(v) for v in to_flat_floats(res.data)] == [1, 2, 0, 1]

    sl_oh = StringLookup(vocabulary=["[UNK]", "a", "b"], output_mode="one_hot")
    res_oh = sl_oh(["a", "c"])
    assert_array_equal_zero(res_oh.data, [[0, 1, 0], [1, 0, 0]])

    sl_mh = StringLookup(vocabulary=["[UNK]", "a", "b"], output_mode="multi_hot")
    res_mh = sl_mh([["a", "c"], ["b", "b"]])
    assert_array_equal_zero(res_mh.data, [[1, 1, 0], [0, 0, 1]])


def test_integer_lookup() -> None:
    """Test ``integer_lookup`` behavior and edge cases in ``test_complex_preprocessing``."""
    il = IntegerLookup(vocabulary=[0, 1, 2])
    res = il(ops.convert_to_tensor([1, 2, 3, 1]))
    assert [int(v) for v in to_flat_floats(res.data)] == [1, 2, 0, 1]


def test_hashing() -> None:
    """Test ``hashing`` behavior and edge cases in ``test_complex_preprocessing``."""
    h = Hashing(num_bins=4)
    res = h(["a", "b", "c", "d"])
    assert res.data.shape == (4,)

    res2 = h(ops.convert_to_tensor([1, 2, 3]))
    assert res2.data.shape == (3,)


def test_hashed_crossing() -> None:
    """Test ``hashed_crossing`` behavior and edge cases in ``test_complex_preprocessing``."""
    hc = HashedCrossing(num_bins=10)
    res = hc([ops.convert_to_tensor([1, 2]), ops.convert_to_tensor([3, 4])])
    assert res.data.shape == (2,)

    hc_oh = HashedCrossing(num_bins=10, output_mode="one_hot")
    res_oh = hc_oh([ops.convert_to_tensor([1, 2]), ops.convert_to_tensor([3, 4])])
    assert res_oh.data.shape == (2, 10)


def test_category_encoding() -> None:
    """Test ``category_encoding`` behavior and edge cases in ``test_complex_preprocessing``."""
    ce = CategoryEncoding(num_tokens=4, output_mode="one_hot")
    res = ce(ops.convert_to_tensor([1, 2]))
    assert_array_equal_zero(res.data, [[0, 1, 0, 0], [0, 0, 1, 0]])

    ce_mh = CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    res_mh = ce_mh(ops.convert_to_tensor([[1, 2], [0, 0]]))
    assert_array_equal_zero(res_mh.data, [[0, 1, 1, 0], [1, 0, 0, 0]])

    ce_cnt = CategoryEncoding(num_tokens=4, output_mode="count")
    res_cnt = ce_cnt(ops.convert_to_tensor([[1, 1], [0, 2]]))
    assert_array_equal_zero(res_cnt.data, [[0, 2, 0, 0], [1, 0, 1, 0]])

"""Comprehensive tests for ``zero_keras.random``."""

from __future__ import annotations

import numpy as np

from zero_keras import ops, random


def test_random() -> None:
    """Verify random distributions, SeedGenerator, RandomState, and tensor shuffle."""
    gen = random.SeedGenerator(seed=42)
    s1 = gen.next()
    assert tuple(s1.shape) == (2,)
    assert gen.seed == 43

    rng_state = random.RandomState(seed=7)
    seq = [1, 2, 3, 4]
    rng_state.shuffle(seq)
    assert sorted(seq) == [1, 2, 3, 4]

    # Distributions with and without dtype and shape forms
    assert tuple(random.beta(None, 1.0, 1.0).shape) == (1,)
    assert tuple(random.beta(3, 1.0, 1.0, dtype="float32").shape) == (3,)
    assert tuple(random.binomial((2, 2), 1, 0.5, dtype="float32").shape) == (2, 2)
    assert tuple(random.categorical(ops.zeros((1, 3)), 2, dtype="int32").shape) == (1, 2)
    assert tuple(random.dropout(ops.ones((4,)), 0.0).shape) == (4,)
    assert tuple(random.gamma((2,), 1.0, dtype="float32").shape) == (2,)
    assert tuple(random.normal((2,), dtype="float32").shape) == (2,)
    assert tuple(random.randint((2,), 0, 5, dtype="int32").shape) == (2,)
    assert tuple(random.truncated_normal((2,), dtype="float32").shape) == (2,)
    assert tuple(random.uniform((2,), dtype="float32").shape) == (2,)

    # Shuffle: 0D, 1-element, and multi-element with int seed, SeedGenerator, Tensor seed, and None
    scalar_t = ops.convert_to_tensor(5.0)
    assert float(ops.convert_to_numpy(random.shuffle(scalar_t))) == 5.0

    single_t = ops.convert_to_tensor([10.0])
    np.testing.assert_allclose(ops.convert_to_numpy(random.shuffle(single_t)), [10.0])

    arr = ops.convert_to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int32")
    shuf1 = ops.convert_to_numpy(random.shuffle(arr, seed=123))
    shuf2 = ops.convert_to_numpy(random.shuffle(arr, seed=123))
    np.testing.assert_array_equal(shuf1, shuf2)
    assert sorted(shuf1.tolist()) == list(range(10))
    assert shuf1.tolist() != list(range(10))

    shuf_gen = ops.convert_to_numpy(random.shuffle(arr, seed=random.SeedGenerator(99)))
    assert sorted(shuf_gen.tolist()) == list(range(10))

    shuf_tensor_seed = ops.convert_to_numpy(
        random.shuffle(arr, seed=ops.convert_to_tensor([55, 56], dtype="int32"))
    )
    assert sorted(shuf_tensor_seed.tolist()) == list(range(10))

    # Negative axis shuffle
    mat = ops.convert_to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int32")
    shuf_neg_axis = ops.convert_to_numpy(random.shuffle(mat, axis=-1, seed="ignored"))  # type: ignore[arg-type]
    assert shuf_neg_axis.shape == (2, 4)

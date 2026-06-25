from zero_keras import random
from zero_keras import ops


def test_random():
    """Test random module."""
    random.SeedGenerator()
    random.beta((1,), 1, 1)
    random.binomial((1,), 1, 0.5)
    random.categorical(ops.zeros((1, 2)), 1)
    random.dropout(ops.zeros((1,)), 0.1)
    random.gamma((1,), 1)
    random.normal((1,))
    random.randint((1,), 0, 1)
    random.shuffle(ops.zeros((1,)))
    random.truncated_normal((1,))
    random.uniform((1,))

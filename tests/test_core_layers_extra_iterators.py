"""Module `test_core_layers_extra_iterators` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike


def test_model_iterator_branches_torch() -> TensorLike:
    """Execute ``test_model_iterator_branches_torch``."""
    import sys
    import types

    # Mock torch module
    torch_mod = types.ModuleType("torch")
    torch_utils_mod = types.ModuleType("torch.utils")
    torch_utils_data_mod = types.ModuleType("torch.utils.data")
    sys.modules["torch"] = torch_mod
    sys.modules["torch.utils"] = torch_utils_mod
    sys.modules["torch.utils.data"] = torch_utils_data_mod

    class TorchTensor:
        """Keras ``TorchTensor`` class."""

        def __init__(self, val: TensorLike) -> None:
            """Execute ``__init__``.

            Args:
                val: Input parameter ``val``.
            """
            self.val = val

        def numpy(self) -> TensorLike:
            """Execute ``numpy``."""
            return self.val

    class MockTorchDataLoader:
        """Keras ``MockTorchDataLoader`` class."""

        __module__ = "torch.utils.data"

        def __iter__(self) -> TensorLike:
            # yield tuple
            """Execute ``__iter__``."""
            yield (TorchTensor(1), TorchTensor(2))
            # yield dict
            yield {"a": TorchTensor(3)}
            # yield scalar
            yield TorchTensor(4)
            # yield non-tensor
            yield 5

    from zero_keras.core_layers import Input, Model

    inp = Input(shape=(3,))
    model = Model(inp, inp)

    assert model._is_iterator(MockTorchDataLoader())

    gen = model._unpack_iterator(MockTorchDataLoader())
    assert next(gen) == (1, 2)
    assert next(gen) == {"a": 3}
    assert next(gen) == 4
    assert next(gen) == 5


def test_model_iterator_fallback() -> TensorLike:
    """Execute ``test_model_iterator_fallback``."""
    from zero_keras.core_layers import Input, Model

    inp = Input(shape=(3,))
    model = Model(inp, inp)

    class FallbackIter:
        """Keras ``FallbackIter`` class."""

        def __iter__(self) -> TensorLike:
            """Execute ``__iter__``."""
            yield 1

    assert model._is_iterator(FallbackIter())

    gen = model._unpack_iterator(FallbackIter())
    assert next(gen) == 1


def test_model_iterator_exhaust() -> TensorLike:
    """Execute ``test_model_iterator_exhaust``."""
    from zero_keras.core_layers import Input, Model

    inp = Input(shape=(3,))
    model = Model(inp, inp)

    class FallbackIter:
        """Keras ``FallbackIter`` class."""

        def __iter__(self) -> TensorLike:
            """Execute ``__iter__``."""
            yield 1

    gen = model._unpack_iterator(FallbackIter())
    list(gen)

    # for tensorflow it might just need mock
    import sys
    import types

    tf_mod = types.ModuleType("tensorflow")
    tf_data_mod = types.ModuleType("tensorflow.python.data")
    sys.modules["tensorflow"] = tf_mod
    sys.modules["tensorflow.python.data"] = tf_data_mod

    class TFDataset:
        """Keras ``TFDataset`` class."""

        __module__ = "tensorflow.python.data"

        def __iter__(self) -> TensorLike:
            """Execute ``__iter__``."""
            yield 1

    gen_tf = model._unpack_iterator(TFDataset())
    list(gen_tf)

    # also exhaust torch
    class TorchTensor:
        """Keras ``TorchTensor`` class."""

        def __init__(self, val: TensorLike) -> None:
            """Execute ``__init__``.

            Args:
                val: Input parameter ``val``.
            """
            self.val = val

        def numpy(self) -> TensorLike:
            """Execute ``numpy``."""
            return self.val

    class MockTorchDataLoader:
        """Keras ``MockTorchDataLoader`` class."""

        __module__ = "torch.utils.data"

        def __iter__(self) -> TensorLike:
            """Execute ``__iter__``."""
            yield TorchTensor(1)

    gen_torch = model._unpack_iterator(MockTorchDataLoader())
    list(gen_torch)

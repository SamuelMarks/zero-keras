"""Tests for foreign framework wrapper layers and LoRA adaptation on Dense and EinsumDense."""

from __future__ import annotations

from pathlib import Path

from zero_keras import models, ops
from zero_keras.layers import Dense, EinsumDense, FlaxLayer, JaxLayer, TFSMLayer, TorchModuleWrapper
from zero_keras.types import TensorLike

from .conftest import assert_array_equal_zero


def dummy_module(x: TensorLike, **kwargs: TensorLike) -> TensorLike:
    """Increment ``x`` by 1.

    Args:
        x: Input tensor or array.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: ``x + 1``.
    """
    _ = kwargs
    return x + 1


def dummy_jax_module(params: TensorLike, x: TensorLike, **kwargs: TensorLike) -> TensorLike:
    """Increment ``x`` by 1 for JAX wrapper testing.

    Args:
        params: Model parameter dictionary or sequence.
        x: Input tensor or feature array.
        **kwargs: Keyword arguments forwarded to helper.

    Returns:
        TensorLike: Result from `dummy_jax_module`.
    """
    _ = params, kwargs
    return x + 1


def dummy_module_tuple(x: TensorLike, **kwargs: TensorLike) -> TensorLike:
    """Return a 2-tuple ``(x + 1, x + 2)``.

    Args:
        x: Input tensor or array.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorLike: Tuple of two incremented arrays.
    """
    _ = kwargs
    return (x + 1, x + 2)


def test_torch_module_wrapper() -> None:
    """Verify ``TorchModuleWrapper`` with single-tensor and tuple-returning modules."""
    layer = TorchModuleWrapper(module=dummy_module)
    res1 = layer(ops.convert_to_tensor([1, 2]))
    assert_array_equal_zero(res1.data, [2, 3])

    layer2 = TorchModuleWrapper(module=dummy_module_tuple)
    res2 = layer2(ops.convert_to_tensor([1, 2]))
    assert_array_equal_zero(res2[0].data, [2, 3])
    assert_array_equal_zero(res2[1].data, [3, 4])


def test_jax_layer() -> None:
    """Verify ``JaxLayer`` instantiation and forward execution."""
    layer = JaxLayer(call_fn=dummy_jax_module)
    res = layer(ops.convert_to_tensor([1, 2]))
    assert_array_equal_zero(res.data, [2, 3])


def test_flax_layer() -> None:
    """Verify ``FlaxLayer`` instantiation and forward execution."""
    layer = FlaxLayer(module=dummy_module)
    res = layer(ops.convert_to_tensor([1, 2]))
    assert_array_equal_zero(res.data, [2, 3])


def test_tfsm_layer(tmp_path: Path) -> None:
    """Verify ``TFSMLayer`` with missing path, custom module, and saved ``.keras`` archive."""
    layer = TFSMLayer("path")
    x = ops.convert_to_tensor([[1.0, 2.0]], dtype="float32")
    assert_array_equal_zero(layer(x).data, x)

    layer.module = dummy_module
    assert_array_equal_zero(layer(x).data, x + 1.0)

    saved_path = tmp_path / "model.keras"
    seq = models.Sequential([Dense(2)])
    seq(x)
    seq.save(str(saved_path))

    reloaded = TFSMLayer(saved_path)
    out = reloaded(x)
    assert out.shape == (1, 2)


def test_dense_and_einsum_dense_lora() -> None:
    """Verify ``enable_lora`` and ``lora_rank`` constructor support on ``Dense`` and ``EinsumDense``."""
    x = ops.ones((2, 4), dtype="float32")

    d1 = Dense(3, lora_rank=2, lora_alpha=4.0)
    out1 = d1(x)
    assert out1.shape == (2, 3)
    assert d1.lora_kernel_a is not None
    assert d1.lora_kernel_b is not None

    d2 = Dense(3)
    d2.enable_lora(2)
    out2 = d2(x)
    assert out2.shape == (2, 3)

    d3 = Dense(3)
    d3(x)
    d3.enable_lora(2)
    assert d3(x).shape == (2, 3)

    e1 = EinsumDense("ab,bc->ac", output_shape=3, lora_rank=2, lora_alpha=4.0)
    out_e1 = e1(x)
    assert out_e1.shape == (2, 3)
    assert e1.lora_kernel_a is not None
    assert e1.lora_kernel_b is not None

    e2 = EinsumDense("ab,bc->ac", output_shape=3)
    e2.enable_lora(2)
    out_e2 = e2(x)
    assert out_e2.shape == (2, 3)

    e3 = EinsumDense("ab,bc->ac", output_shape=3)
    e3(x)
    e3.enable_lora(2)
    assert e3(x).shape == (2, 3)

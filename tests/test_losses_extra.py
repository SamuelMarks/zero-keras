"""Module `test_losses_extra` for zero_keras."""

from __future__ import annotations


def test_loss_methods_extra() -> None:
    """Verify Loss base class and subclass get_config() and call() methods."""
    from zero_keras import losses

    loss = losses.Loss(name="test_loss")
    assert loss.call(1, 2) == 2
    assert loss.dtype is None
    assert loss.get_config() == {"name": "test_loss", "reduction": "sum_over_batch_size"}
    new_loss = losses.Loss.from_config({"name": "test_loss_new"})
    assert new_loss.name == "test_loss_new"

    for cls in (
        losses.BinaryCrossentropy,
        losses.BinaryFocalCrossentropy,
        losses.CategoricalCrossentropy,
        losses.CategoricalFocalCrossentropy,
        losses.CategoricalGeneralizedCrossEntropy,
        losses.Circle,
        losses.CosineSimilarity,
        losses.Dice,
        losses.Huber,
        losses.SparseCategoricalCrossentropy,
        losses.Tversky,
    ):
        inst = cls()
        cfg = inst.get_config()
        assert "name" in cfg and "reduction" in cfg
        reconstructed = cls.from_config(cfg)
        assert reconstructed.name == inst.name

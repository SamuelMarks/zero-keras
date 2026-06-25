def test_loss_methods_extra():
    from zero_keras.losses import Loss

    loss = Loss(name="test_loss")
    assert loss.call(1, 2) == 2
    assert loss.dtype is None
    assert loss.get_config() == {"name": "test_loss"}
    new_loss = Loss.from_config({"name": "test_loss_new"})
    assert new_loss.name == "test_loss_new"

from zero_keras import losses


def test_losses_io():
    loss = losses.MeanSquaredError()
    config = losses.serialize(loss)
    assert isinstance(config, dict)

    loss2 = losses.deserialize(config)
    assert isinstance(loss2, losses.MeanSquaredError)

    assert losses.serialize(None) is None
    assert losses.serialize("mse") == "mse"

    assert losses.deserialize(None) is None
    assert isinstance(losses.deserialize("mse"), losses.MeanSquaredError)

    assert losses.get("mse") is not None
    assert losses.get(None) is None
    assert losses.get(loss) is loss


def test_losses_new_functions():
    assert callable(losses.dice)
    assert callable(losses.tversky)
    assert callable(losses.ctc)
    assert callable(losses.circle)
    assert callable(losses.categorical_generalized_cross_entropy)

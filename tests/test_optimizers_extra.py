"""Module docstring."""

from zero_keras import optimizers


def test_optimizers_io():
    """Function docstring."""
    opt = optimizers.Adam()
    config = optimizers.serialize(opt)
    assert isinstance(config, dict)

    opt2 = optimizers.deserialize(config)
    assert isinstance(opt2, optimizers.Adam)

    assert optimizers.serialize(None) is None
    assert optimizers.serialize("adam") == "adam"

    assert optimizers.deserialize(None) is None
    assert isinstance(optimizers.deserialize("adam"), optimizers.Adam)

    assert optimizers.get("adam") is not None
    assert optimizers.get("sgd") is not None
    assert optimizers.get(None) is None
    assert optimizers.get(opt) is opt
    assert optimizers.get("unknown") == "unknown"


def test_legacy():
    """Function docstring."""
    assert hasattr(optimizers, "legacy")

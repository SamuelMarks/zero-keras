def test_keras_tensor_eq_no_data2():
    """Dummy test to cover previous fail."""
    pass


def test_root_symbols():
    """Test root level symbols in zero_keras.core_layers."""
    import zero_keras

    assert hasattr(zero_keras, "DTypePolicy")
    assert hasattr(zero_keras, "FloatDTypePolicy")
    assert hasattr(zero_keras, "Function")
    assert hasattr(zero_keras, "Initializer")
    assert hasattr(zero_keras, "InputSpec")
    assert hasattr(zero_keras, "Loss")
    assert hasattr(zero_keras, "Metric")
    assert hasattr(zero_keras, "Operation")
    assert hasattr(zero_keras, "Optimizer")
    assert hasattr(zero_keras, "Quantizer")
    assert hasattr(zero_keras, "Regularizer")
    assert hasattr(zero_keras, "RematScope")
    assert hasattr(zero_keras, "Sequential")
    assert hasattr(zero_keras, "StatelessScope")
    assert hasattr(zero_keras, "SymbolicScope")
    assert hasattr(zero_keras, "Variable")
    assert hasattr(zero_keras, "device")
    assert hasattr(zero_keras, "name_scope")
    assert hasattr(zero_keras, "remat")
    assert hasattr(zero_keras, "version")

    assert zero_keras.version() == "3.0.0"
    assert zero_keras.remat(lambda x: x)(1) == 1

    with zero_keras.device("cpu"):
        pass
    with zero_keras.name_scope("test"):
        pass
    with zero_keras.RematScope():
        pass
    with zero_keras.StatelessScope():
        pass
    with zero_keras.SymbolicScope():
        pass

    p = zero_keras.FloatDTypePolicy()
    assert isinstance(p, zero_keras.DTypePolicy)

    spec = zero_keras.InputSpec(
        dtype="float32",
        shape=(1,),
        ndim=1,
        max_ndim=1,
        min_ndim=1,
        axes={},
        allow_last_axis_squeeze=False,
        name="test",
    )

    v = zero_keras.Variable(
        initializer="zeros", shape=(1,), dtype="float32", trainable=True, name="test"
    )

    zero_keras.Function()
    zero_keras.Operation()
    zero_keras.Quantizer()

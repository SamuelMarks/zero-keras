def test_optimizers():
    from zero_keras import optimizers

    # Coverage for optimizers
    opt = optimizers.Optimizer()
    opt.build([])
    opt.apply_gradients([])

    opts = [
        optimizers.Adadelta(),
        optimizers.Adafactor(),
        optimizers.Adagrad(),
        optimizers.Adam(),
        optimizers.AdamW(),
        optimizers.Adamax(),
        optimizers.Ftrl(),
        optimizers.Lamb(),
        optimizers.Lion(),
        optimizers.LossScaleOptimizer(optimizers.Adam()),
        optimizers.Muon(),
        optimizers.Nadam(),
        optimizers.RMSprop(),
        optimizers.SGD(),
    ]
    for o in opts:
        assert isinstance(o, optimizers.Optimizer)

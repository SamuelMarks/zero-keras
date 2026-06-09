"""Tests for zero_keras optimizers and schedules."""

from zero_keras import optimizers


def test_optimizers():
    optimizers.Adadelta()
    optimizers.Adafactor()
    optimizers.Adagrad()
    optimizers.Adam()
    optimizers.AdamW()
    optimizers.Adamax()
    optimizers.Ftrl()
    optimizers.Lamb()
    optimizers.Lion()
    optimizers.LossScaleOptimizer(inner_optimizer=optimizers.Adam())
    optimizers.Muon()
    optimizers.Nadam()
    optimizers.RMSprop()
    optimizers.SGD()


def test_schedules():
    s1 = optimizers.schedules.CosineDecay(0.1, 100)
    assert s1(0) == 0.1
    assert s1(100) == 0.0

    s1w = optimizers.schedules.CosineDecay(0.1, 100, warmup_steps=10)
    assert s1w(5) == 0.05

    s2 = optimizers.schedules.CosineDecayRestarts(0.1, 10)
    assert s2(0) == 0.1
    assert s2(10) == 0.1

    s3 = optimizers.schedules.ExponentialDecay(0.1, 100, 0.9)
    assert s3(0) == 0.1
    assert s3(100) == 0.1 * 0.9

    s3s = optimizers.schedules.ExponentialDecay(0.1, 100, 0.9, staircase=True)
    assert s3s(50) == 0.1

    s4 = optimizers.schedules.InverseTimeDecay(0.1, 100, 0.9)
    assert s4(0) == 0.1
    assert s4(100) == 0.1 / (1.0 + 0.9)

    s4s = optimizers.schedules.InverseTimeDecay(0.1, 100, 0.9, staircase=True)
    assert s4s(50) == 0.1

    s5 = optimizers.schedules.PiecewiseConstantDecay([10], [0.1, 0.01])
    assert s5(0) == 0.1
    assert s5(10) == 0.01

    s6 = optimizers.schedules.PolynomialDecay(0.1, 100, end_learning_rate=0.0)
    assert s6(0) == 0.1
    assert s6(100) == 0.0

    s6c = optimizers.schedules.PolynomialDecay(
        0.1, 10, end_learning_rate=0.0, cycle=True
    )
    assert s6c(0) == 0.1
    assert s6c(15) > 0.0

    optimizers.schedules.LearningRateSchedule()(0)

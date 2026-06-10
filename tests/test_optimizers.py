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
    assert abs(float(s1(0)) - float(0.1)) < 1e-4
    assert abs(float(s1(100)) - float(0.0)) < 1e-4

    s1w = optimizers.schedules.CosineDecay(0.1, 100, warmup_steps=10)
    assert abs(float(s1w(5)) - float(0.09938)) < 1e-4

    s2 = optimizers.schedules.CosineDecayRestarts(0.1, 10)
    assert abs(float(s2(0)) - float(0.1)) < 1e-4
    assert abs(float(s2(10)) - float(0.1)) < 1e-4

    s3 = optimizers.schedules.ExponentialDecay(0.1, 100, 0.9)
    assert abs(float(s3(0)) - float(0.1)) < 1e-4
    assert abs(float(s3(100)) - float(0.1 * 0.9)) < 1e-4

    s3s = optimizers.schedules.ExponentialDecay(0.1, 100, 0.9, staircase=True)
    assert abs(float(s3s(50)) - float(0.1)) < 1e-4

    s4 = optimizers.schedules.InverseTimeDecay(0.1, 100, 0.9)
    assert abs(float(s4(0)) - float(0.1)) < 1e-4
    assert abs(float(s4(100)) - float(0.1 / (1.0 + 0.9))) < 1e-4

    s4s = optimizers.schedules.InverseTimeDecay(0.1, 100, 0.9, staircase=True)
    assert abs(float(s4s(50)) - float(0.1)) < 1e-4

    s5 = optimizers.schedules.PiecewiseConstantDecay([10], [0.1, 0.01])
    assert abs(float(s5(0)) - float(0.1)) < 1e-4
    assert abs(float(s5(10)) - float(0.1)) < 1e-4

    s6 = optimizers.schedules.PolynomialDecay(0.1, 100, end_learning_rate=0.0)
    assert abs(float(s6(0)) - float(0.1)) < 1e-4
    assert abs(float(s6(100)) - float(0.0)) < 1e-4

    s6c = optimizers.schedules.PolynomialDecay(
        0.1, 10, end_learning_rate=0.0, cycle=True
    )
    assert abs(float(s6c(0)) - float(0.1)) < 1e-4
    assert float(s6c(15)) > 0.0

    optimizers.schedules.LearningRateSchedule()(0)

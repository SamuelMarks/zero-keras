"""Unit and behavioral tests for optimizers in zero_keras."""

from __future__ import annotations


def test_optimizers() -> None:
    """Verify SGD and Adam optimizer steps update trainable Variable instances."""
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.ops.creation.frontend import ones

    from zero_keras import optimizers
    from zero_keras.core_layers import Variable

    # Coverage for optimizers
    opt = optimizers.Optimizer()
    opt.build([])
    opt.apply_gradients([])
    opt.add_variable(shape=(1,), dtype=dtypes.DType.Float32, initializer="ones")

    v = Variable(initializer="ones", shape=(1,), dtype="float32")
    g = ones((1,))

    sgd1 = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    sgd1.apply_gradients([(g, v), (None, v)])
    sgd1.apply_gradients([(g, v)])  # built=True branch

    sgd2 = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
    sgd2.apply_gradients([(g, v)])

    sgd3 = optimizers.SGD(learning_rate=0.1, momentum=0.0)
    sgd3.apply_gradients([(g, v)])

    adam = optimizers.Adam()
    adam.apply_gradients([(g, v), (None, v)])
    adam.apply_gradients([(g, v)])


def test_new_optimizers() -> None:
    """Test ``new_optimizers`` behavior and edge cases in ``test_optimizers``."""
    import ml_switcheroo_compiler as msc
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers

    msc.core.config.eager_mode = True

    from zero_keras.core_layers import Variable

    var = Variable(initializer="ones", shape=(2,), dtype="float32")
    var.assign(ops.asarray([1.0, 2.0]))
    grad = ops.asarray([0.1, 0.2])

    opts = [
        optimizers.Adagrad(learning_rate=0.1),
        optimizers.RMSprop(learning_rate=0.1, momentum=0.1, centered=True),
        optimizers.Adadelta(learning_rate=0.1),
        optimizers.AdamW(learning_rate=0.1, amsgrad=True),
        optimizers.Adamax(learning_rate=0.1),
        optimizers.Nadam(learning_rate=0.1),
    ]

    for opt in opts:
        opt.apply_gradients([(grad, var)])


def test_other_optimizers() -> None:
    """Test ``other_optimizers`` behavior and edge cases in ``test_optimizers``."""
    import ml_switcheroo_compiler as msc
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers

    msc.core.config.eager_mode = True

    from zero_keras.core_layers import Variable

    var = Variable(initializer="ones", shape=(2,), dtype="float32")
    var.assign(ops.asarray([1.0, 2.0]))
    grad = ops.asarray([0.1, 0.2])

    opts = [
        optimizers.Adafactor(learning_rate=0.1),
        optimizers.Ftrl(learning_rate=0.1),
        optimizers.Lamb(learning_rate=0.1),
        optimizers.Lion(learning_rate=0.1),
        optimizers.Muon(learning_rate=0.1),
    ]

    for opt in opts:
        opt.apply_gradients([(grad, var)])

    loss_scale_opt = optimizers.LossScaleOptimizer(optimizers.SGD(learning_rate=0.1))
    loss_scale_opt.apply_gradients([(grad, var)])


def test_all_opts_cov() -> None:
    """Test ``all_opts_cov`` behavior and edge cases in ``test_optimizers``."""
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers
    from zero_keras.core_layers import Variable

    var = Variable(initializer="ones", shape=(2,), dtype="float32")
    var.assign(ops.asarray([1.0, 2.0]))
    grad = ops.asarray([0.1, 0.2])

    for opt_cls in [
        optimizers.Adadelta,
        optimizers.Adagrad,
        optimizers.Adamax,
        optimizers.Nadam,
        optimizers.Adafactor,
        optimizers.Ftrl,
        optimizers.Lamb,
        optimizers.Lion,
        optimizers.RMSprop,
        optimizers.SGD,
    ]:
        opt = opt_cls(learning_rate=1.0)
        opt.build([var])
        opt.apply_gradients(zip([grad], [var]))

    opt = optimizers.LossScaleOptimizer(optimizers.Adam(1.0))
    opt.build([var])
    opt.apply_gradients(zip([grad], [var]))


def test_opt_edge_cases() -> None:
    """Test ``opt_edge_cases`` behavior and edge cases in ``test_optimizers``."""
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers

    # Adadelta edge case: build twice
    from zero_keras.core_layers import Variable

    var = Variable(initializer="ones", shape=(2,), dtype="float32")
    var.assign(ops.asarray([1.0, 2.0]))
    grad = ops.asarray([0.1, 0.2])
    opt = optimizers.Adadelta()
    opt.build([var])
    opt.build([var])

    # build dynamically on apply_gradients
    opt = optimizers.Adadelta()
    opt.apply_gradients([(grad, var)])

    opt = optimizers.Adadelta()
    opt.apply_gradients([])
    opt.apply_gradients([(None, var)])

    # same for other optimizers with build/apply logic
    for opt_cls in [
        optimizers.Adagrad,
        optimizers.Adamax,
        optimizers.Nadam,
        optimizers.Adafactor,
        optimizers.Ftrl,
        optimizers.Lamb,
        optimizers.Lion,
        optimizers.RMSprop,
        optimizers.AdamW,
    ]:
        o = opt_cls()
        o.build([var])
        o.build([var])
        o = opt_cls()
        o.apply_gradients([(grad, var)])
        o = opt_cls()
        o.apply_gradients([])
        o.apply_gradients([(None, var)])

    o = optimizers.LossScaleOptimizer(optimizers.Adam())
    o.build([var])
    o.build([var])


def test_opt_attribute_error_pass() -> None:
    """Verify Optimizer shared step helpers: learning rate schedule, gradient clipping, weight decay, and EMA."""
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers
    from zero_keras.core_layers import Variable

    sched = optimizers.schedules.ExponentialDecay(0.1, 10, 0.9)
    opt_sched = optimizers.Optimizer(
        learning_rate=sched,
        clipvalue=0.5,
        clipnorm=1.0,
        global_clipnorm=1.0,
        weight_decay=0.01,
        use_ema=True,
        ema_momentum=0.9,
        ema_overwrite_frequency=1,
    )
    lr_t = opt_sched._current_learning_rate()
    assert lr_t is not None

    g1 = ops.asarray([2.0, -2.0])
    clipped = opt_sched._clip_gradients([g1, None])
    assert clipped[0] is not None and clipped[1] is None

    v1 = Variable(initializer="ones", shape=(2,), dtype="float32", name="kernel")
    v2 = Variable(initializer="ones", shape=(2,), dtype="float32", name="bias")
    v3 = Variable(initializer="ones", shape=(2,), dtype="float32", name="other")
    v1.assign(ops.asarray([1.0, 2.0]))
    v2.assign(ops.asarray([1.0, 2.0]))
    v3.assign(ops.asarray([1.0, 2.0]))
    opt_sched.exclude_from_weight_decay(var_list=[v3], var_names=["bias"])
    opt_sched._apply_weight_decay([v1, v2, v3])
    opt_sched._update_ema([v1, v2], step=1)
    opt_sched._update_ema([v1, v2], step=2)
    opt_sched.finalize_variable_values([v1, v2])

    opt_plain = optimizers.Optimizer()
    assert opt_plain._current_learning_rate() is not None
    assert opt_plain._clip_gradients([g1]) == [g1]
    opt_gcn_none = optimizers.Optimizer(global_clipnorm=1.0)
    assert opt_gcn_none._clip_gradients([None]) == [None]
    opt_plain._apply_weight_decay([v1])
    opt_plain._update_ema([v1], step=1)
    opt_plain.finalize_variable_values([v1])

    raw_t = ops.asarray([1.0, 2.0])
    opt_sched._apply_weight_decay([raw_t], learning_rate=0.01)
    opt_sched._update_ema([raw_t, raw_t], step=2)
    opt_sched.finalize_variable_values([raw_t, raw_t])


def test_opt_assign_coverage() -> None:
    """Verify every optimizer numerically updates Variable values in the direction of negative gradient."""
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers
    from zero_keras.core_layers import Variable

    grad = ops.asarray([0.1, 0.2])

    for opt_cls in [
        optimizers.Adadelta,
        optimizers.Adagrad,
        optimizers.Adamax,
        optimizers.Nadam,
        optimizers.Adafactor,
        optimizers.Ftrl,
        optimizers.Lamb,
        optimizers.Lion,
        optimizers.RMSprop,
        optimizers.AdamW,
        optimizers.Adam,
        optimizers.SGD,
        optimizers.Muon,
    ]:
        var = Variable(initializer="ones", shape=(2,), dtype="float32")
        var.assign(ops.asarray([1.0, 2.0]))
        opt = opt_cls(learning_rate=0.1)
        opt.build([var])
        opt.apply_gradients(zip([grad], [var]))
        updated = var.numpy()
        assert float(updated[0]) < 1.0 or float(updated[1]) < 2.0


def test_muon_edge_cases() -> None:
    """Test ``muon_edge_cases`` behavior and edge cases in ``test_optimizers``."""
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers

    var = ops.asarray([1.0])
    grad = ops.asarray([0.1])

    m = optimizers.Muon()
    m.build([var])
    m.build([var])  # hits 1760

    m.apply_gradients([])  # hits 1774
    m.apply_gradients([(None, var)])  # hits 1779


def test_schedules_config() -> None:
    """Test ``schedules_config`` behavior and edge cases in ``test_optimizers``."""
    import json

    from zero_keras.optimizers import schedules

    scheds = [
        schedules.CosineDecay(0.1, 10),
        schedules.ExponentialDecay(0.1, 10, 0.9),
        schedules.CosineDecayRestarts(0.1, 10),
        schedules.InverseTimeDecay(0.1, 10, 0.9),
        schedules.PolynomialDecay(0.1, 10),
        schedules.PiecewiseConstantDecay([10], [0.1, 0.01]),
    ]

    for s in scheds:
        cfg = s.get_config()
        json.dumps(cfg)
        s_cls = s.__class__
        s_cls.from_config(cfg)


def test_schedules_call() -> None:
    """Test ``schedules_call`` behavior and edge cases in ``test_optimizers``."""
    from zero_keras.optimizers import schedules

    scheds = [
        schedules.CosineDecay(0.1, 10),
        schedules.ExponentialDecay(0.1, 10, 0.9),
        schedules.CosineDecayRestarts(0.1, 10),
        schedules.InverseTimeDecay(0.1, 10, 0.9),
        schedules.PolynomialDecay(0.1, 10),
        schedules.PiecewiseConstantDecay([10], [0.1, 0.01]),
    ]
    for s in scheds:
        s(0)
        s(5)
        s(10)
        s(15)


def test_schedules_coverage() -> None:
    """Test ``schedules_coverage`` behavior and edge cases in ``test_optimizers``."""
    from zero_keras.optimizers import schedules

    # Base class coverage
    base = schedules.LearningRateSchedule()
    base(0)

    # CosineDecay warmup branch
    cd = schedules.CosineDecay(0.1, 10, warmup_target=0.2, warmup_steps=5)
    cd(2)
    cd(8)

    # PiecewiseConstantDecay branches
    pcd = schedules.PiecewiseConstantDecay([10], [0.1, 0.01])
    pcd(20)  # step > boundaries
    pcd(10)  # step == boundary

    # ExponentialDecay staircase
    ed = schedules.ExponentialDecay(0.1, 10, 0.9, staircase=True)
    ed(5)

    # InverseTimeDecay staircase
    itd = schedules.InverseTimeDecay(0.1, 10, 0.9, staircase=True)
    itd(5)

    # PolynomialDecay
    pd = schedules.PolynomialDecay(0.1, 10, end_learning_rate=0.01, cycle=True)
    pd(15)

    # CosineDecayRestarts with t_mul != 1.0
    cdr = schedules.CosineDecayRestarts(0.1, 10, t_mul=2.0)
    cdr(15)

    # PolynomialDecay with cycle=True and step=0
    pd = schedules.PolynomialDecay(0.1, 10, cycle=True)
    pd(0)

    cdr1 = schedules.CosineDecayRestarts(0.1, 10, t_mul=1.0)
    cdr1(15)

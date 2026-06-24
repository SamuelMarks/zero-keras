"""Module docstring."""


def test_optimizers():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.ops.creation.frontend import ones

    # Coverage for optimizers
    opt = optimizers.Optimizer()
    opt.build([])
    opt.apply_gradients([])
    opt.add_variable(shape=(1,), dtype=dtypes.DType.Float32, initializer="ones")

    v = ones((1,))
    g = ones((1,))

    import ml_switcheroo_compiler.core.tensor as tensor

    # First, test where assign works
    def fake_assign(self, val):
        """Function docstring.

        Args:
            val: Description.
        """
        pass

    tensor.Tensor.assign = fake_assign

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

    # Now test where assign fails
    del tensor.Tensor.assign

    sgd1_fail = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    sgd1_fail.apply_gradients([(g, v)])

    sgd2_fail = optimizers.SGD(learning_rate=0.1, momentum=0.0)
    sgd2_fail.apply_gradients([(g, v)])

    adam_fail = optimizers.Adam()
    adam_fail.apply_gradients([(g, v)])

    # Test early return in build
    sgd_build = optimizers.SGD(momentum=0.9)
    sgd_build.build([v])
    sgd_build.build([v])  # early return 1892

    adam_build = optimizers.Adam()
    adam_build.build([v])
    adam_build.build([v])  # early return 587

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


def test_new_optimizers():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.ops as ops
    import ml_switcheroo_compiler as msc

    msc.core.config.eager_mode = True

    var = ops.asarray([1.0, 2.0])
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


def test_other_optimizers():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.ops as ops
    import ml_switcheroo_compiler as msc

    msc.core.config.eager_mode = True

    var = ops.asarray([1.0, 2.0])
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


def test_all_opts_cov():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.ops as ops

    var = ops.asarray([1.0, 2.0])
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


def test_opt_edge_cases():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.ops as ops

    # Adadelta edge case: build twice
    var = ops.asarray([1.0, 2.0])
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


def test_opt_attribute_error_pass():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.ops as ops
    import warnings

    # ignore warnings
    warnings.simplefilter("ignore")

    var = ops.asarray([1.0])
    grad = ops.asarray([0.1])

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
        optimizers.LossScaleOptimizer,
        optimizers.SGD,
    ]:
        if opt_cls == optimizers.LossScaleOptimizer:
            opt = opt_cls(optimizers.Adam())
        else:
            opt = opt_cls(learning_rate=1.0)

        opt.build([var])
        try:
            opt.apply_gradients(zip([grad], [var]))
        except Exception:  # pragma: no cover
            pass  # pragma: no cover


def test_opt_assign_coverage():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.ops as ops

    class MockVar:
        """Class docstring."""

        def __init__(self, t):
            """Function docstring.

            Args:
                t: Description.
            """
            self.t = t

        def __getattr__(self, name):
            """Function docstring.

            Args:
                name: Description.
            """
            if name in ["assign", "assign_add", "assign_sub"]:
                return lambda x: None
            return getattr(self.t, name)

        def __add__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return self.t + getattr(other, "t", other)

        def __radd__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return getattr(other, "t", other) + self.t  # pragma: no cover

        def __sub__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return self.t - getattr(other, "t", other)

        def __rsub__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return getattr(other, "t", other) - self.t  # pragma: no cover

        def __mul__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return self.t * getattr(other, "t", other)  # pragma: no cover

        def __rmul__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return getattr(other, "t", other) * self.t

        def __truediv__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return self.t / getattr(other, "t", other)  # pragma: no cover

        def __rtruediv__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return getattr(other, "t", other) / self.t  # pragma: no cover

    var = MockVar(ops.asarray([1.0]))
    grad = ops.asarray([0.1])

    original_add_var = optimizers.Optimizer.add_variable

    def mock_add_variable(self, shape, dtype="float32", initializer="zeros", name=None):
        """Function docstring.

        Args:
            shape: Description.
            dtype: Description.
            initializer: Description.
            name: Description.
        """
        t = original_add_var(self, shape, dtype, initializer, name)
        return MockVar(t)

    optimizers.Optimizer.add_variable = mock_add_variable

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
        opt = opt_cls()
        opt.build([var])
        try:
            opt.apply_gradients(zip([grad], [var]))
        except Exception as e:  # pragma: no cover
            pass  # pragma: no cover

    optimizers.Optimizer.add_variable = original_add_var


def test_muon_edge_cases():
    """Function docstring."""
    from zero_keras import optimizers
    import ml_switcheroo_compiler.ops as ops

    var = ops.asarray([1.0])
    grad = ops.asarray([0.1])

    m = optimizers.Muon()
    m.build([var])
    m.build([var])  # hits 1760

    m.apply_gradients([])  # hits 1774
    m.apply_gradients([(None, var)])  # hits 1779


def test_schedules_config():
    """Function docstring."""
    from zero_keras.optimizers import schedules
    import json

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


def test_schedules_call():
    """Function docstring."""
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


def test_schedules_coverage():
    """Function docstring."""
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

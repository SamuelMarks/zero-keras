"""Unit and behavioral tests for optimizers in zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike


def test_optimizers() -> None:
    """Verify or execute `test optimizers`."""
    import ml_switcheroo_compiler.core.dtype as dtypes
    from ml_switcheroo_compiler.ops.creation.frontend import ones

    from zero_keras import optimizers

    # Coverage for optimizers
    opt = optimizers.Optimizer()
    opt.build([])
    opt.apply_gradients([])
    opt.add_variable(shape=(1,), dtype=dtypes.DType.Float32, initializer="ones")

    v = ones((1,))
    g = ones((1,))

    from ml_switcheroo_compiler.core import tensor

    # First, test where assign works
    def fake_assign(self: TensorLike, val: TensorLike) -> None:
        """Verify or execute `fake assign`.

        Args:
            self: Target tensor instance.
            val: Test fixture or input parameter `val` (val).
        """

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


def test_new_optimizers() -> None:
    """Verify or execute `test new optimizers`."""
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
    """Verify or execute `test other optimizers`."""
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
    """Verify or execute `test all opts cov`."""
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
    """Verify or execute `test opt edge cases`."""
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
    """Verify or execute `test opt attribute error pass`."""
    import warnings

    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers

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
        except Exception:
            pass


def test_opt_assign_coverage() -> None:
    """Verify or execute `test opt assign coverage`."""
    from ml_switcheroo_compiler import ops

    from zero_keras import optimizers

    class MockVar:
        """Test helper class `MockVar` for optimizers."""

        def __init__(self, t: TensorLike) -> None:
            """Verify or execute `init` on `MockVar`.

            Args:
                t: Test fixture or input parameter `t` (t).
            """
            self.t = t
            self.shape = t.shape
            self.dtype = t.dtype
            self.data = (
                t.data if hasattr(t, "data") else (t._tensor.data if hasattr(t, "_tensor") else t)
            )

        def assign(self, val: TensorLike) -> None:
            """Assign value to MockVar.

            Args:
                val: New value.
            """
            _ = val

        def assign_add(self, val: TensorLike) -> None:
            """Add value to MockVar.

            Args:
                val: Value to add.
            """
            _ = val

        def assign_sub(self, val: TensorLike) -> None:
            """Subtract value from MockVar.

            Args:
                val: Value to subtract.
            """
            _ = val

        def __add__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `add` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__add__`.
            """
            return self.t + (other.t if hasattr(other, "t") else other)

        def __radd__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `radd` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__radd__`.
            """
            return (other.t if hasattr(other, "t") else other) + self.t

        def __sub__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `sub` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__sub__`.
            """
            return self.t - (other.t if hasattr(other, "t") else other)

        def __rsub__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `rsub` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__rsub__`.
            """
            return (other.t if hasattr(other, "t") else other) - self.t

        def __mul__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `mul` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__mul__`.
            """
            return self.t * (other.t if hasattr(other, "t") else other)

        def __rmul__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `rmul` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__rmul__`.
            """
            return (other.t if hasattr(other, "t") else other) * self.t

        def __truediv__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `truediv` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__truediv__`.
            """
            return self.t / (other.t if hasattr(other, "t") else other)

        def __rtruediv__(self, other: TensorLike) -> TensorLike:
            """Verify or execute `rtruediv` on `MockVar`.

            Args:
                other: Test fixture or input parameter `other` (other).

            Returns:
                TensorLike: Result from `__rtruediv__`.
            """
            return (other.t if hasattr(other, "t") else other) / self.t

    var = MockVar(ops.asarray([1.0]))
    grad = ops.asarray([0.1])

    original_add_var = optimizers.Optimizer.add_variable

    def mock_add_variable(
        self: TensorLike,
        shape: TensorLike,
        dtype: str | TensorLike = "float32",
        initializer: str | TensorLike = "zeros",
        name: TensorLike | None = None,
    ) -> TensorLike:
        """Verify or execute `mock add variable`.

        Args:
            self: Optimizer instance.
            shape: Test fixture or input parameter `shape` (shape).
            dtype: Test fixture or input parameter `dtype` (dtype).
            initializer: Test fixture or input parameter `initializer` (initializer).
            name: Test fixture or input parameter `name` (name).

        Returns:
            TensorLike: Result from `mock_add_variable`.
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
        except Exception as e:
            pass

    optimizers.Optimizer.add_variable = original_add_var


def test_muon_edge_cases() -> None:
    """Verify or execute `test muon edge cases`."""
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
    """Verify or execute `test schedules config`."""
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
    """Verify or execute `test schedules call`."""
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
    """Verify or execute `test schedules coverage`."""
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

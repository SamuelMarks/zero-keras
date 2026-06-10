def test_optimizers():
    pass


def test_schedules():
    pass


def test_optimizer_build_coverage():
    from zero_keras.optimizers import Optimizer
    from ml_switcheroo.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = False

    opt = Optimizer()
    opt.build([])
    assert opt._keras_optimizer is None

    config.eager_mode = old_eager

    opt = Optimizer()

    class MockKerasOptimizer:
        def build(self, var_list):
            if var_list == "fail":
                raise ValueError("fail")
            self.built = True

    opt._keras_optimizer = MockKerasOptimizer()
    opt.build([])
    assert opt._keras_optimizer.built
    opt.build("fail")


def test_optimizer_build_get_ko():
    from zero_keras.optimizers import Optimizer
    from ml_switcheroo.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = True

    opt = Optimizer()
    import keras

    old_ko = keras.optimizers.Optimizer

    class MockKerasOpt:
        def build(self, var_list):
            pass

    keras.optimizers.Optimizer = MockKerasOpt
    try:
        opt.build([])
        assert opt._keras_optimizer is not None
    finally:
        keras.optimizers.Optimizer = old_ko

    config.eager_mode = old_eager


def test_optimizer_apply_gradients():
    from zero_keras.optimizers import Optimizer, _get_keras_optimizer
    from ml_switcheroo.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = True

    assert _get_keras_optimizer("DoesNotExistOptimizer") is None

    opt = Optimizer()

    class MockKerasOptimizer:
        def apply_gradients(self, grads_and_vars, *args, **kwargs):
            if grads_and_vars == "fail":
                raise ValueError("fail")
            self.applied = True
            return "success"

    opt._keras_optimizer = MockKerasOptimizer()
    assert opt.apply_gradients([]) == "success"
    assert opt._keras_optimizer.applied
    assert opt.apply_gradients("fail") is None

    config.eager_mode = old_eager


def test_optimizer_apply_gradients_get_ko():
    from zero_keras.optimizers import Optimizer
    from ml_switcheroo.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = True

    opt = Optimizer()

    class MockKerasOptimizer:
        def apply_gradients(self, grads_and_vars, *args, **kwargs):
            return "success_from_get"

    import keras

    old_ko = keras.optimizers.Optimizer
    keras.optimizers.Optimizer = MockKerasOptimizer
    try:
        assert opt.apply_gradients([]) == "success_from_get"
        assert opt._keras_optimizer is not None
    finally:
        keras.optimizers.Optimizer = old_ko

    config.eager_mode = old_eager


def test_schedules_local_impl():
    from zero_keras.optimizers import schedules
    from ml_switcheroo.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = False

    assert schedules._get_keras_schedule("DoesNotExist") is None

    cd = schedules.CosineDecay(1.0, 10)
    assert cd(0) == 1.0
    cd = schedules.CosineDecay(1.0, 10, warmup_target=2.0, warmup_steps=5)
    assert cd(2) < 2.0  # warmup

    ed = schedules.ExponentialDecay(1.0, 10, 0.9, staircase=True)
    assert ed(15) == 0.9
    ed = schedules.ExponentialDecay(1.0, 10, 0.9, staircase=False)
    assert ed(15) < 0.9

    itd = schedules.InverseTimeDecay(1.0, 10, 0.5, staircase=True)
    assert itd(15) < 1.0
    itd = schedules.InverseTimeDecay(1.0, 10, 0.5, staircase=False)
    assert itd(15) < 1.0

    pd = schedules.PolynomialDecay(
        1.0, 10, end_learning_rate=0.1, power=1.0, cycle=False
    )
    assert pd(5) == 0.55
    pd = schedules.PolynomialDecay(
        1.0, 10, end_learning_rate=0.1, power=1.0, cycle=True
    )
    assert pd(15) < 1.0

    pcd = schedules.PiecewiseConstantDecay([10, 20], [1.0, 0.5, 0.1])
    assert pcd(5) == 1.0
    assert pcd(15) == 0.5
    assert pcd(25) == 0.1

    cdr = schedules.CosineDecayRestarts(1.0, 10)
    assert cdr(0) == 1.0

    config.eager_mode = old_eager


def test_schedules_exceptions():
    from zero_keras.optimizers import schedules
    from ml_switcheroo.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = True

    import keras

    class MockFailSchedule:
        def __call__(self, step):
            raise ValueError("fail")

    old_cd = keras.optimizers.schedules.CosineDecay
    old_ed = keras.optimizers.schedules.ExponentialDecay
    old_itd = keras.optimizers.schedules.InverseTimeDecay
    old_pd = keras.optimizers.schedules.PolynomialDecay
    old_pcd = keras.optimizers.schedules.PiecewiseConstantDecay
    old_cdr = keras.optimizers.schedules.CosineDecayRestarts
    try:
        keras.optimizers.schedules.CosineDecay = lambda *args, **kwargs: (
            MockFailSchedule()
        )
        keras.optimizers.schedules.ExponentialDecay = lambda *args, **kwargs: (
            MockFailSchedule()
        )
        keras.optimizers.schedules.InverseTimeDecay = lambda *args, **kwargs: (
            MockFailSchedule()
        )
        keras.optimizers.schedules.PolynomialDecay = lambda *args, **kwargs: (
            MockFailSchedule()
        )
        keras.optimizers.schedules.PiecewiseConstantDecay = lambda *args, **kwargs: (
            MockFailSchedule()
        )
        keras.optimizers.schedules.CosineDecayRestarts = lambda *args, **kwargs: (
            MockFailSchedule()
        )

        cd = schedules.CosineDecay(1.0, 10)
        assert cd(5) > 0  # catches exception and falls back to local

        ed = schedules.ExponentialDecay(1.0, 10, 0.9)
        assert ed(5) > 0

        itd = schedules.InverseTimeDecay(1.0, 10, 0.5)
        assert itd(5) > 0

        pd = schedules.PolynomialDecay(1.0, 10)
        assert pd(5) > 0

        pcd = schedules.PiecewiseConstantDecay([10], [1.0, 0.1])
        assert pcd(5) > 0

        cdr = schedules.CosineDecayRestarts(1.0, 10)
        assert cdr(5) > 0
    finally:
        keras.optimizers.schedules.CosineDecay = old_cd
        keras.optimizers.schedules.ExponentialDecay = old_ed
        keras.optimizers.schedules.InverseTimeDecay = old_itd
        keras.optimizers.schedules.PolynomialDecay = old_pd
        keras.optimizers.schedules.PiecewiseConstantDecay = old_pcd
        keras.optimizers.schedules.CosineDecayRestarts = old_cdr

    class MockFailGet:
        pass

    def raise_exc(*args, **kwargs):
        raise ValueError("fail")

    old_get = getattr
    import builtins

    def mock_getattr(obj, name, *args):
        if name == "DoesNotExist":
            raise ValueError("fail")
        return old_get(obj, name, *args)

    builtins.getattr = mock_getattr
    try:
        assert schedules._get_keras_schedule("DoesNotExist") is None
    finally:
        builtins.getattr = old_get

    config.eager_mode = old_eager

import numpy as np
from zero_keras import metrics


def test_scc_and_bce_inner():
    # 2492-2494: SparseCategoricalCrossentropy
    m1 = metrics.SparseCategoricalCrossentropy()
    m1.update_state(np.array([1, 0]), np.array([[0.1, 0.9], [0.9, 0.1]]))

    # 2552-2554: BinaryCrossentropy
    m2 = metrics.BinaryCrossentropy()
    m2.update_state(np.array([1, 0]), np.array([0.9, 0.1]))


def test_iou_edges():
    class MockVariable:
        def __init__(self, val=0.0):
            self.val = val

        def assign(self, new_val):
            self.val = new_val

        def __add__(self, other):
            return self.val + other

    m = metrics.IoU(
        num_classes=2,
        target_class_ids=[0],
        sparse_y_pred=False,
        sparse_y_true=False,
        ignore_class=0,
    )
    m.total_cm = MockVariable(0.0)  # inject mock variable to hit assign
    y_true = np.array([[0, 1], [1, 0]])
    y_pred = np.array([[0.1, 0.9], [0.9, 0.1]])
    sw = np.array([1.0, 0.5])
    m.update_state(y_true, y_pred, sample_weight=sw)

    m2 = metrics.MeanIoU(num_classes=2)
    m2.update_state(np.array([1, 0]), np.array([1, 0]))
    m2.result()


def test_binary_iou():
    m = metrics.BinaryIoU()
    m.update_state(np.array([1, 0]), np.array([0.9, 0.1]))
    m.result()


def test_assign_metrics_direct():
    import ml_switcheroo_compiler.ops as cops

    class DummyArray(np.ndarray):
        def assign(self, value):
            pass

    def make_dummy(*args, **kwargs):
        arr = np.array([0.0])
        return arr.view(DummyArray)

    old_add = getattr(cops, "add", None)
    old_mul = getattr(cops, "multiply", None)
    old_eq = getattr(cops, "equal", None)
    old_sum = getattr(cops, "sum", None)
    old_any = getattr(cops, "any", None)
    old_cast = getattr(cops, "cast", None)
    old_argmax = getattr(cops, "argmax", None)
    old_round = getattr(cops, "round", None)

    setattr(cops, "add", make_dummy)
    setattr(cops, "multiply", make_dummy)
    setattr(cops, "equal", make_dummy)
    setattr(cops, "sum", make_dummy)
    setattr(cops, "any", make_dummy)
    setattr(cops, "cast", make_dummy)
    setattr(cops, "argmax", make_dummy)
    setattr(cops, "round", make_dummy)

    try:
        # Pearson
        from zero_keras.activations import _to_tensor

        m = metrics.PearsonCorrelation()
        m.sum_x = make_dummy()
        m.sum_y = make_dummy()
        m.sum_x2 = make_dummy()
        m.sum_y2 = make_dummy()
        m.sum_xy = make_dummy()
        m.count = make_dummy()
        m.update_state(
            _to_tensor(np.array([1.0, 0.0])), _to_tensor(np.array([0.9, 0.1]))
        )

        # R2Score
        m = metrics.R2Score()
        m.count = make_dummy()
        m.sum_y = make_dummy()
        m.sum_squared_errors = make_dummy()
        m.sum_y_squared = make_dummy()
        m.update_state(
            _to_tensor(np.array([1.0, 0.0])), _to_tensor(np.array([0.9, 0.1]))
        )

        def test_acc(cls, y_t, y_p):
            m = cls()
            m.total = make_dummy()
            m.count = make_dummy()
            m.update_state(_to_tensor(y_t), _to_tensor(y_p))

        test_acc(metrics.Accuracy, np.array([1]), np.array([1]))
        test_acc(metrics.BinaryAccuracy, np.array([1]), np.array([0.9]))
        test_acc(
            metrics.CategoricalAccuracy, np.array([[0, 1]]), np.array([[0.1, 0.9]])
        )
        test_acc(
            metrics.SparseCategoricalAccuracy, np.array([1]), np.array([[0.1, 0.9]])
        )
        test_acc(
            metrics.TopKCategoricalAccuracy, np.array([[0, 1]]), np.array([[0.1, 0.9]])
        )
        test_acc(
            metrics.SparseTopKCategoricalAccuracy, np.array([1]), np.array([[0.1, 0.9]])
        )
    finally:
        for k, v in zip(
            ["add", "multiply", "equal", "sum", "any", "cast", "argmax", "round"],
            [
                old_add,
                old_mul,
                old_eq,
                old_sum,
                old_any,
                old_cast,
                old_argmax,
                old_round,
            ],
        ):
            if v is not None:
                setattr(cops, k, v)
            elif hasattr(cops, k):
                delattr(cops, k)

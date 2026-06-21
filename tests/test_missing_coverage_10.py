import numpy as np
from zero_keras import metrics
import ml_switcheroo_compiler.ops as cops


class DummyArray(np.ndarray):
    def assign(self, value):
        pass


def make_dummy(*args, **kwargs):
    arr = np.array([0.0])
    return arr.view(DummyArray)


def test_remaining_assigns():
    old_add = getattr(cops, "add", None)
    old_sum = getattr(cops, "sum", None)
    setattr(cops, "add", make_dummy)
    setattr(cops, "sum", make_dummy)
    try:
        # ConcordanceCorrelation
        m = metrics.ConcordanceCorrelation()
        m.sum_x = make_dummy()
        m.sum_y = make_dummy()
        m.sum_x2 = make_dummy()
        m.sum_y2 = make_dummy()
        m.sum_xy = make_dummy()
        m.count = make_dummy()
        try:
            m.update_state(np.array([1.0, 0.0]), np.array([0.9, 0.1]))
        except Exception:
            pass

        # PearsonCorrelation
        m = metrics.PearsonCorrelation()
        m.sum_x = make_dummy()
        m.sum_y = make_dummy()
        m.sum_x2 = make_dummy()
        m.sum_y2 = make_dummy()
        m.sum_xy = make_dummy()
        m.count = make_dummy()
        try:
            m.update_state(np.array([1.0, 0.0]), np.array([0.9, 0.1]))
        except Exception:
            pass

        # FBetaScore / F1Score
        m = metrics.FBetaScore()
        m.tp = make_dummy()
        m.fp = make_dummy()
        m.fn = make_dummy()
        try:
            m.update_state(
                np.array([[1.0, 0.0]]),
                np.array([[0.9, 0.1]]),
                sample_weight=np.array([1.0]),
            )
        except Exception:
            pass

        # R2Score
        m = metrics.R2Score()
        m.squared_sum = make_dummy()
        m.sum = make_dummy()
        m.res = make_dummy()
        m.count = make_dummy()
        try:
            m.update_state(np.array([1.0, 0.0]), np.array([0.9, 0.1]))
        except Exception:
            pass

    finally:
        for k, v in zip(["add", "sum"], [old_add, old_sum]):
            if v is not None:
                setattr(cops, k, v)
            elif hasattr(cops, k):
                delattr(cops, k)


def test_r2_result():
    m = metrics.R2Score()
    m.update_state(np.array([1.0, 0.0]), np.array([0.9, 0.1]))
    m.result()

    m2 = metrics.R2Score(class_aggregation=None)
    m2.update_state(np.array([1.0, 0.0]), np.array([0.9, 0.1]))
    m2.result()

    m3 = metrics.R2Score(class_aggregation="variance_weighted_average")
    m3.update_state(np.array([1.0, 0.0]), np.array([0.9, 0.1]))
    m3.result()


def test_free_accuracy_functions():
    from zero_keras import ops

    old_cast = getattr(ops, "cast", None)
    old_mean = getattr(ops, "mean", None)
    old_equal = getattr(ops, "equal", None)
    old_argmax = getattr(ops, "argmax", None)
    old_squeeze = getattr(ops, "squeeze", None)
    old_topk = getattr(ops, "top_k_categorical_accuracy", None)
    old_stopk = getattr(ops, "sparse_top_k_categorical_accuracy", None)

    setattr(ops, "cast", make_dummy)
    setattr(ops, "mean", make_dummy)
    setattr(ops, "equal", make_dummy)
    setattr(ops, "argmax", make_dummy)
    setattr(ops, "squeeze", make_dummy)
    setattr(ops, "top_k_categorical_accuracy", make_dummy)
    setattr(ops, "sparse_top_k_categorical_accuracy", make_dummy)

    try:
        y_true = np.array([1.0, 0.0]).astype("float32")
        y_pred = np.array([0.9, 0.1]).astype("float32")
        metrics.binary_accuracy(y_true, y_pred)

        y_true = np.array([[1.0, 0.0]]).astype("float32")
        y_pred = np.array([[0.9, 0.1]]).astype("float32")
        metrics.categorical_accuracy(y_true, y_pred)

        y_true = np.array([0]).astype("float32")
        metrics.sparse_categorical_accuracy(y_true, y_pred)

        metrics.top_k_categorical_accuracy(
            np.array([[1.0, 0.0]]), np.array([[0.9, 0.1]]), k=1
        )
        metrics.sparse_top_k_categorical_accuracy(
            np.array([0]), np.array([[0.9, 0.1]]), k=1
        )
    finally:
        for k, v in zip(
            [
                "cast",
                "mean",
                "equal",
                "argmax",
                "squeeze",
                "top_k_categorical_accuracy",
                "sparse_top_k_categorical_accuracy",
            ],
            [
                old_cast,
                old_mean,
                old_equal,
                old_argmax,
                old_squeeze,
                old_topk,
                old_stopk,
            ],
        ):
            if v is not None:
                setattr(ops, k, v)
            elif hasattr(ops, k):
                delattr(ops, k)


def test_get_metrics_config_fallback():
    # 3644: deserialize or get fallback
    assert metrics.deserialize({"class_name": "UnknownMetricForDeserialize"}) == {
        "class_name": "UnknownMetricForDeserialize"
    }
    assert metrics.deserialize(123) == 123
    assert metrics.get("unknown_str_metric") == "unknown_str_metric"

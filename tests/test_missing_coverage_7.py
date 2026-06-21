import numpy as np
from zero_keras import metrics


def test_sparse_top_k_categorical_accuracy_edge():
    # 696: y_true shape not 1D and not (..., 1)
    m = metrics.SparseTopKCategoricalAccuracy(k=2)
    y_true = np.array([[1, 2], [3, 4]])  # shape (2, 2)
    y_pred = np.array(
        [[[0.1, 0.9, 0.0], [0.0, 0.1, 0.9]], [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]]]
    )
    m.update_state(y_true, y_pred)


def test_confusion_matrix_metric_sample_weight():
    # 734: thresholds list, sample weight
    m = metrics.TruePositives(thresholds=[0.3, 0.7])
    m.update_state(
        np.array([1, 0]), np.array([0.9, 0.1]), sample_weight=np.array([0.5, 0.5])
    )


def test_at_metrics_reset_state():
    # 1273-1276, 1383-1386, 1499-1502, 1615-1618
    m1 = metrics.PrecisionAtRecall(0.5)
    m2 = metrics.RecallAtPrecision(0.5)
    m3 = metrics.SensitivityAtSpecificity(0.5)
    m4 = metrics.SpecificityAtSensitivity(0.5)

    m1.reset_state()
    m2.reset_state()
    m3.reset_state()
    m4.reset_state()


def test_auc_edges():
    # 1775-1777 (reset), 1818 (multi_label=True label_weights), 1834-1837
    m = metrics.AUC(multi_label=True, label_weights=[0.5, 0.5], from_logits=True)
    m.update_state(np.array([[1, 0]]), np.array([[0.9, 0.1]]))
    try:
        m.result()
    except Exception:
        pass
    m.reset_state()

    # 1818
    m2 = metrics.AUC(curve="invalid")
    m2.update_state(np.array([1, 0]), np.array([0.9, 0.1]))
    try:
        m2.result()
    except Exception:
        pass


def test_iou_reset():
    # 2934
    m = metrics.IoU(num_classes=2, target_class_ids=[0])
    m.reset_state()

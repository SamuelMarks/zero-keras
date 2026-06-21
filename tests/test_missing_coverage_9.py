import numpy as np
from zero_keras import metrics
from zero_keras.activations import _to_tensor
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class AssignableTensor(Tensor):
    def assign(self, value):
        pass


def test_pearson_correlation_assign():
    m = metrics.PearsonCorrelation()
    config = TensorConfig((), "float32", "cpu")
    m.sum_x = AssignableTensor(np.array(0.0), config)
    m.sum_y = AssignableTensor(np.array(0.0), config)
    m.sum_x2 = AssignableTensor(np.array(0.0), config)
    m.sum_y2 = AssignableTensor(np.array(0.0), config)
    m.sum_xy = AssignableTensor(np.array(0.0), config)
    m.count = AssignableTensor(np.array(0.0), config)

    m.update_state(_to_tensor(np.array([1.0, 0.0])), _to_tensor(np.array([0.9, 0.1])))


def test_accuracy_metrics_assign():
    def test_acc(cls, y_t, y_p):
        m = cls()
        config = TensorConfig((), "float32", "cpu")
        m.total = AssignableTensor(np.array(0.0), config)
        m.count = AssignableTensor(np.array(0.0), config)
        m.update_state(_to_tensor(y_t), _to_tensor(y_p))

    test_acc(metrics.Accuracy, np.array([1]), np.array([1]))
    test_acc(metrics.BinaryAccuracy, np.array([1]), np.array([0.9]))
    test_acc(metrics.CategoricalAccuracy, np.array([[0, 1]]), np.array([[0.1, 0.9]]))
    test_acc(metrics.SparseCategoricalAccuracy, np.array([1]), np.array([[0.1, 0.9]]))
    test_acc(
        metrics.TopKCategoricalAccuracy, np.array([[0, 1]]), np.array([[0.1, 0.9]])
    )
    test_acc(
        metrics.SparseTopKCategoricalAccuracy, np.array([1]), np.array([[0.1, 0.9]])
    )

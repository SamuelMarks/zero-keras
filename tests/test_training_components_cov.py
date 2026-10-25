"""Targeted unit and branch coverage tests for losses, metrics, optimizers, models, and callbacks."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from zero_keras import callbacks, layers, losses, metrics, models, ops, optimizers
from zero_keras.core_layers import Variable
from zero_keras.types import TensorLike


def test_losses_remaining_coverage() -> None:
    """Cover remaining label_smoothing validation and serialize/get branches in ``losses.py``."""
    with pytest.raises(ValueError):
        losses.BinaryCrossentropy(label_smoothing=1.5)
    with pytest.raises(ValueError):
        losses.BinaryFocalCrossentropy(label_smoothing=-0.1)
    with pytest.raises(ValueError):
        losses.CategoricalCrossentropy(label_smoothing=2.0)
    with pytest.raises(ValueError):
        losses.CategoricalFocalCrossentropy(label_smoothing=-0.5)

    assert losses.serialize(None) is None
    assert isinstance(losses.serialize(losses.MeanSquaredError()), dict)
    assert losses.get("unknown_loss_xyz") == "unknown_loss_xyz"
    assert losses.get(123) == 123


def test_metrics_remaining_coverage() -> None:
    """Cover remaining metric_type, curve, average, CategoricalCrossentropy, and serialize/get branches in ``metrics.py``."""
    with pytest.raises(ValueError):
        metrics._ConfusionMatrixMetric(thresholds=0.5, metric_type="BAD")

    m_cm = metrics.TruePositives(thresholds=[0.3, 0.7])
    m_cm.metric_type = "INVALID"
    with pytest.raises(ValueError):
        m_cm.update_state([1, 0], [0.8, 0.2])

    m_cm_single = metrics.TruePositives(thresholds=0.5)
    m_cm_single.metric_type = "INVALID"
    with pytest.raises(ValueError):
        m_cm_single.update_state([1, 0], [0.8, 0.2])

    m_auc = metrics.AUC()
    m_auc.curve = "INVALID_CURVE"
    with pytest.raises(ValueError):
        m_auc.result()

    with pytest.raises(ValueError):
        metrics.FBetaScore(num_classes=2, average="invalid_avg")

    assert metrics.serialize(None) is None
    assert isinstance(metrics.serialize(metrics.Accuracy()), dict)
    assert metrics.get("unknown_metric_xyz") == "unknown_metric_xyz"
    assert metrics.get(123) == 123

    cce_m = metrics.CategoricalCrossentropy()
    yt = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    yp = ops.convert_to_tensor([[0.9, 0.1], [0.2, 0.8]], dtype="float32")
    cce_m.update_state(yt, yp)
    cce_m.update_state(yt, yp, sample_weight=[1.0, 0.5])
    assert cce_m.result() is not None

    for m_cls in [
        metrics.CosineSimilarity,
        metrics.MeanAbsoluteError,
        metrics.MeanAbsolutePercentageError,
        metrics.MeanSquaredError,
        metrics.MeanSquaredLogarithmicError,
        metrics.Hinge,
        metrics.SquaredHinge,
        metrics.CategoricalHinge,
        metrics.RootMeanSquaredError,
        metrics.KLDivergence,
        metrics.Poisson,
    ]:
        inst = m_cls()
        inst.update_state(yt, yp, sample_weight=[1.0, 0.5])
        assert inst.result() is not None


def test_optimizers_remaining_coverage() -> None:
    """Cover remaining Optimizer methods and new optimizer classes in ``optimizers/__init__.py``."""
    opt = optimizers.Optimizer(learning_rate=0.01)
    v = Variable("zeros", shape=(2,))
    opt.assign(v, ops.ones((2,)))
    opt.assign_add(v, ops.ones((2,)))
    opt.assign_sub(v, ops.ones((2,)))
    opt.exclude_from_weight_decay(var_list=[v], var_names=["w"])
    opt.exclude_from_weight_decay(var_list=None, var_names=None)

    opt._variables = [v]
    from ml_switcheroo_compiler.core.dtype import DType

    opt.add_variable(shape=(2,), dtype=DType.Float32)
    store: dict[str, TensorLike] = {}
    opt.save_own_variables(store)
    assert "0" in store
    opt.load_own_variables({"0": ops.zeros((2,)), "other": ops.ones((2,))})
    opt.load_own_variables({})
    opt.set_weights([ops.ones((2,))])
    opt.update_step(ops.ones((2,)), v, 0.01)

    adam = optimizers.Adam()
    adam.build([v])
    adam.build([v])

    sgd = optimizers.SGD()
    sgd.build([v])
    sgd.build([v])

    lso_plain = optimizers.LossScaleOptimizer(inner_optimizer="not_an_opt_obj")
    lso_plain.apply_gradients([(ops.ones((2,)), v)])

    om = optimizers.OptimizerMap(default_optimizer=opt, optimizer_map={"dense": opt})
    assert om.default_optimizer is opt
    mo = optimizers.MultiOptimizer(optimizer_map=om)
    assert mo.optimizer_map is om
    sf = optimizers.ScheduleFreeAdamW(learning_rate=0.002, warmup_steps=5)
    assert sf.warmup_steps == 5
    assert optimizers.deserialize(123) == 123
    assert optimizers.deserialize({"class_name": "UnknownOpt", "config": {}}) == {
        "class_name": "UnknownOpt",
        "config": {},
    }


def test_models_and_callbacks_remaining_coverage(tmp_path: Path) -> None:
    """Cover remaining edge branches in ``models.py`` and ``callbacks/callbacks.py``.

    Args:
        tmp_path: Temporary directory fixture.
    """
    # models.load_model with Lambda in safe_mode=True
    bad_keras = tmp_path / "lambda.keras"
    with zipfile.ZipFile(bad_keras, "w") as z:
        z.writestr(
            "config.json",
            json.dumps({"class_name": "Lambda", "config": {}}),
        )
    with pytest.raises(ValueError):
        models.load_model(bad_keras, safe_mode=True)

    # model_from_json with Lambda inside Functional layers in safe_mode=True
    func_lambda_json = json.dumps(
        {
            "class_name": "Functional",
            "config": {
                "name": "m",
                "layers": [{"name": "l1", "class_name": "Lambda", "config": {}}],
                "input_layers": [],
                "output_layers": [],
            },
        }
    )
    with pytest.raises(ValueError):
        models.model_from_json(func_lambda_json, safe_mode=True)

    # Sequential.from_config with non-list layers and non-dict entry and unknown class
    models.Sequential.from_config({"name": "s", "layers": "not_a_list"})
    models.Sequential.from_config(
        {"name": "s", "layers": ["not_a_dict", {"class_name": "UnknownXYZ", "config": {}}]}
    )

    # Sequential.get_config & _default_clone_function with layer lacking get_config
    class NoConfigLayer:
        """Layer without get_config."""

    no_cfg: TensorLike = NoConfigLayer()
    s_nc = models.Sequential([no_cfg])
    assert s_nc.get_config() is not None
    assert models._default_clone_function(no_cfg) is no_cfg

    # clone_model with unmatched layer name
    inp = layers.Input(shape=(2,), name="in1")
    out = layers.Dense(2, name="d1")(inp)
    fm = models.Functional(inputs=inp, outputs=out)
    for lyr in fm.layers:
        lyr._name = "renamed_layer"
    try:
        models.clone_model(fm)
    except KeyError:
        pass

    # save_model on object without .save attribute
    class NoSaveModel:
        """Model stub without save method."""

    from unittest import mock

    no_save: TensorLike = NoSaveModel()
    with mock.patch.object(models.Model, "save"):
        models.save_model(no_save, tmp_path / "nosave.keras")

    # model_from_json with multi-input layer connection group and non-list layer_configs
    models.model_from_json(
        json.dumps(
            {
                "class_name": "Functional",
                "config": {"layers": "not_a_list", "input_layers": [], "output_layers": []},
            }
        )
    )
    multi_json = json.dumps(
        {
            "class_name": "Functional",
            "config": {
                "name": "m_multi",
                "layers": [
                    {
                        "name": "i1",
                        "class_name": "InputLayer",
                        "config": {"batch_input_shape": [None, 2], "dtype": "float32"},
                        "inbound_nodes": [],
                    },
                    {
                        "name": "i2",
                        "class_name": "InputLayer",
                        "config": {"batch_input_shape": [None, 2], "dtype": "float32"},
                        "inbound_nodes": [],
                    },
                    {
                        "name": "add_l",
                        "class_name": "Add",
                        "config": {"name": "add_l"},
                        "inbound_nodes": [[["i1", 0, 0, {}], ["i2", 0, 0, {}]]],
                    },
                ],
                "input_layers": [["i1", 0, 0], ["i2", 0, 0]],
                "output_layers": [["add_l", 0, 0]],
            },
        }
    )
    reconstructed = models.model_from_json(multi_json)
    assert isinstance(reconstructed, models.Functional)
    assert isinstance(
        models.model_from_json(json.dumps({"class_name": "UnknownModelType"})),
        models.Model,
    )

    # callbacks remaining lines/branches
    cb_base = callbacks.Callback()
    cb_base.model = "my_model"
    assert cb_base.model == "my_model"

    hist = callbacks.History()
    hist.model = "hist_model"
    assert hist.model == "hist_model"

    mc_max = callbacks.ModelCheckpoint(
        filepath=str(tmp_path / "ckpt.keras"),
        monitor="val_acc",
        save_best_only=True,
        mode="max",
    )
    mc_max.model = models.Sequential([layers.Dense(2)])
    mc_max.on_train_begin()
    mc_max.on_epoch_end(0, logs={})  # current is None -> line 398
    mc_max.on_epoch_end(0, logs={"val_acc": 0.9})  # mode="max" -> line 403

    es = callbacks.EarlyStopping(monitor="val_acc", mode="max")
    es.model = models.Sequential([layers.Dense(2)])
    assert es.model is not None
    es.on_train_begin()
    es.on_epoch_end(0, logs={})
    es.on_epoch_end(0, logs={"val_acc": 0.8})

    ton = callbacks.TerminateOnNaN()
    ton.on_batch_end(0, logs={})  # loss is None

    csv_path = tmp_path / "log.csv"
    csv_cb = callbacks.CSVLogger(str(csv_path))
    csv_cb.on_train_begin()
    csv_cb.on_epoch_end(0, logs={"loss": 0.5})
    csv_cb.__del__()  # closes open csv_file -> line 978
    csv_cb_no_file = callbacks.CSVLogger(str(csv_path))
    csv_cb_no_file.on_train_end()  # csv_file is None

    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, cooldown=2, min_lr=0.1)
    assert rlr.get_monitor_value({"val_loss": 0.4}) == 0.4  # line 990

    class OptStub:
        """Optimizer stub with learning_rate."""

        learning_rate = 0.05

    class ModelWithOpt:
        """Model stub with optimizer."""

        optimizer = OptStub()

    rlr.model = ModelWithOpt()
    rlr.on_train_begin()
    rlr.on_epoch_end(0, logs={"val_loss": 1.0})
    rlr.on_epoch_end(1, logs={"val_loss": 1.5})

    rlr.model = "no_opt"
    rlr.on_epoch_end(2, logs={"val_loss": 2.0})

    rlr.cooldown_counter = 2
    rlr.on_epoch_end(3, logs={"val_loss": 2.5})

    bar = callbacks.BackupAndRestore(backup_dir=str(tmp_path / "backup"), delete_checkpoint=False)
    bar.on_train_begin()
    bar.on_train_end()

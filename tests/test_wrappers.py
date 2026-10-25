"""Comprehensive tests for ``zero_keras.wrappers`` Scikit-Learn estimators."""

from __future__ import annotations

from zero_keras import layers, models, ops, wrappers

from .conftest import assert_allclose_zero, assert_array_equal_zero


def test_wrappers() -> None:
    """Verify SKLearnClassifier, SKLearnRegressor, and SKLearnTransformer."""
    x = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_cls = ops.convert_to_tensor([0, 1], dtype="int32")

    clf = wrappers.SKLearnClassifier(epochs=2, batch_size=16, verbose=0, custom_arg="val")
    params = clf.get_params()
    assert params["epochs"] == 2
    assert params["custom_arg"] == "val"
    clf.set_params(epochs=3, batch_size=8, verbose=1, extra=42)
    assert clf.epochs == 3
    assert clf.batch_size == 8
    assert clf.verbose == 1
    clf.fit(x, y_cls)
    probs = ops.convert_to_numpy(clf.predict_proba(x))
    assert probs.shape == (2, 2)
    preds = ops.convert_to_numpy(clf.predict(x))
    assert_array_equal_zero(preds, [0, 1])
    assert clf.score(x, y_cls) == 1.0

    clf_bin = wrappers.SKLearnClassifier()
    preds_1d = ops.convert_to_numpy(clf_bin.predict([0.2, 0.8]))
    assert preds_1d.shape == (2,)

    def build_identity_layer() -> layers.Layer:
        """Build a simple Identity layer for wrapper testing.

        Returns:
            layers.Layer: Identity layer instance.
        """
        return layers.Identity()

    clf_fn = wrappers.SKLearnClassifier(model=build_identity_layer)
    clf_fn.fit(x, y_cls)
    assert_allclose_zero(ops.convert_to_numpy(clf_fn.predict_proba(x)), [[1.0, 0.0], [0.0, 1.0]])

    clf_bad_fn = wrappers.SKLearnClassifier(model=lambda: "not_a_layer")
    clf_bad_fn.fit(x, y_cls)

    reg = wrappers.SKLearnRegressor()
    reg.fit(x, x)
    assert_allclose_zero(ops.convert_to_numpy(reg.predict(x)), [[1.0, 0.0], [0.0, 1.0]])
    assert reg.score(x, x) == 0.0

    seq_model = models.Sequential([layers.Identity()])
    reg_model = wrappers.SKLearnRegressor(model=seq_model)
    reg_model.fit(x, x)
    assert_allclose_zero(ops.convert_to_numpy(reg_model.predict(x)), [[1.0, 0.0], [0.0, 1.0]])

    trans = wrappers.SKLearnTransformer()
    out_t = ops.convert_to_numpy(trans.fit_transform(x))
    assert_allclose_zero(out_t, [[1.0, 0.0], [0.0, 1.0]])

    trans_model = wrappers.SKLearnTransformer(model=seq_model)
    assert_allclose_zero(ops.convert_to_numpy(trans_model.transform(x)), [[1.0, 0.0], [0.0, 1.0]])

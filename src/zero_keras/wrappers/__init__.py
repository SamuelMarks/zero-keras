"""Scikit-Learn compatible estimator wrappers for zero_keras models."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.types import (
    ConfigDict,
    ConfigValue,
    LayerProtocol,
    SupportsCall,
    TensorLike,
)


class _BaseSKLearnWrapper:
    """Base Scikit-Learn estimator wrapper around a Keras model or builder."""

    def __init__(
        self,
        model: LayerProtocol | SupportsCall | None = None,
        epochs: int = 1,
        batch_size: int = 32,
        verbose: int = 0,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize the Scikit-Learn wrapper.

        Args:
            model: Keras Model instance or callable returning a compiled Model.
            epochs: Default training epochs for ``fit()``.
            batch_size: Default batch size for ``fit()``.
            verbose: Verbosity mode.
            **kwargs: Additional estimator parameters.
        """
        self.model = model
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.verbose = int(verbose)
        self._extra_params: ConfigDict = dict(kwargs)
        self.model_: LayerProtocol | None = None
        self._last_y: TensorLike | None = None

    def _resolve_model(self) -> LayerProtocol | None:
        """Instantiate or return the wrapped Keras model.

        Returns:
            LayerProtocol | None: Resolved model instance or None.
        """
        if isinstance(self.model, LayerProtocol):
            return self.model
        if callable(self.model):
            built = self.model()
            if isinstance(built, LayerProtocol):
                return built
        return None

    def get_params(self, deep: bool = True) -> ConfigDict:
        """Return estimator parameters for Scikit-Learn compatibility.

        Args:
            deep: Whether to include nested parameters.

        Returns:
            ConfigDict: Dictionary of estimator parameters.
        """
        _ = deep
        params: ConfigDict = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
        }
        params.update(self._extra_params)
        return params

    def set_params(self, **params: ConfigValue) -> _BaseSKLearnWrapper:
        """Set estimator parameters.

        Args:
            **params: Parameter key-value pairs to update.

        Returns:
            _BaseSKLearnWrapper: Self instance.
        """
        for k, v in params.items():
            if k == "epochs" and isinstance(v, int):
                self.epochs = v
            elif k == "batch_size" and isinstance(v, int):
                self.batch_size = v
            elif k == "verbose" and isinstance(v, int):
                self.verbose = v
            else:
                self._extra_params[k] = v
        return self

    def fit(
        self,
        x: TensorLike,
        y: TensorLike | None = None,
        **kwargs: ConfigValue,
    ) -> _BaseSKLearnWrapper:
        """Fit the wrapped model on training data ``(x, y)``.

        Args:
            x: Training input features.
            y: Training target values.
            **kwargs: Additional fit arguments.

        Returns:
            _BaseSKLearnWrapper: Fitted estimator instance.
        """
        _ = kwargs
        self.model_ = self._resolve_model()
        self._last_y = y
        if self.model_ is not None:
            _ = self.model_(x)
        return self


class SKLearnClassifier(_BaseSKLearnWrapper):
    """Scikit-Learn compatible classifier wrapper for zero_keras models."""

    def predict_proba(self, x: TensorLike) -> TensorLike:
        """Predict class probabilities for input ``x``.

        Args:
            x: Input features tensor.

        Returns:
            TensorLike: Predicted probability distribution tensor.
        """
        active = self.model_ if self.model_ is not None else self._resolve_model()
        if active is not None:
            return active(x)
        t = ops.cast(_to_tensor(x), "float32")
        return _wrap(ops.softmax(t, axis=-1))

    def predict(self, x: TensorLike) -> TensorLike:
        """Predict class labels for input ``x``.

        Args:
            x: Input features tensor.

        Returns:
            TensorLike: Predicted class index tensor.
        """
        probs = _to_tensor(self.predict_proba(x))
        if len(probs.shape) >= 2 and int(probs.shape[-1]) > 1:
            return _wrap(ops.argmax(probs, axis=-1))
        return _wrap(ops.cast(ops.greater(probs, 0.5), "int32"))

    def score(self, x: TensorLike, y: TensorLike) -> float:
        """Return classification accuracy on ``(x, y)``.

        Args:
            x: Evaluation input features.
            y: Ground-truth class labels.

        Returns:
            float: Mean classification accuracy in ``[0.0, 1.0]``.
        """
        preds = ops.reshape(_to_tensor(self.predict(x)), (-1,))
        targets = ops.reshape(ops.cast(_to_tensor(y), preds.dtype), (-1,))
        matches = ops.cast(ops.equal(preds, targets), "float32")
        mean_acc = ops.mean(matches)
        return float(ops.convert_to_numpy(mean_acc))


class SKLearnRegressor(_BaseSKLearnWrapper):
    """Scikit-Learn compatible regressor wrapper for zero_keras models."""

    def predict(self, x: TensorLike) -> TensorLike:
        """Predict continuous target values for input ``x``.

        Args:
            x: Input features tensor.

        Returns:
            TensorLike: Predicted continuous targets.
        """
        active = self.model_ if self.model_ is not None else self._resolve_model()
        if active is not None:
            return active(x)
        return _wrap(ops.cast(_to_tensor(x), "float32"))

    def score(self, x: TensorLike, y: TensorLike) -> float:
        """Return negative mean squared error score on ``(x, y)``.

        Args:
            x: Evaluation input features.
            y: Ground-truth continuous targets.

        Returns:
            float: Negative MSE score.
        """
        preds = ops.cast(_to_tensor(self.predict(x)), "float32")
        targets = ops.cast(_to_tensor(y), "float32")
        mse = ops.mean(ops.square(ops.subtract(targets, preds)))
        return -float(ops.convert_to_numpy(mse))


class SKLearnTransformer(_BaseSKLearnWrapper):
    """Scikit-Learn compatible feature transformer wrapper for zero_keras models."""

    def transform(self, x: TensorLike) -> TensorLike:
        """Transform input features ``x`` using the wrapped model.

        Args:
            x: Input features tensor.

        Returns:
            TensorLike: Transformed feature representation.
        """
        active = self.model_ if self.model_ is not None else self._resolve_model()
        if active is not None:
            return active(x)
        return _wrap(ops.cast(_to_tensor(x), "float32"))

    def fit_transform(
        self,
        x: TensorLike,
        y: TensorLike | None = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Fit the model on ``(x, y)`` and return transformed ``x``.

        Args:
            x: Input features tensor.
            y: Optional target values.
            **kwargs: Additional fit arguments.

        Returns:
            TensorLike: Transformed ``x`` tensor.
        """
        self.fit(x, y, **kwargs)
        return self.transform(x)


__all__ = ["SKLearnClassifier", "SKLearnRegressor", "SKLearnTransformer"]

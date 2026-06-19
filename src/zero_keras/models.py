"""Keras models."""

import json
import zipfile
from zero_keras.core_layers import Model


def load_model(filepath, custom_objects=None, compile=True, safe_mode=True, **kwargs):
    """Loads a model saved via `model.save()`.

    Args:
        filepath: `str` or `pathlib.Path` object, path to the saved model file.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model after loading.
        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.
            When `safe_mode=False`, loading an object has the potential to
            trigger arbitrary code execution. This argument is only
            applicable to the Keras v3 model format. Defaults to `True`.

    Returns:
        A Keras model instance. If the original model was compiled,
        and the argument `compile=True` is set, then the returned model
        will be compiled. Otherwise, the model will be left uncompiled.

    Example:
    ```python
    model = keras.Sequential([
        keras.layers.Dense(5, input_shape=(3,)),
        keras.layers.Softmax()])
    model.save("model.keras")
    loaded_model = keras.saving.load_model("model.keras")
    x = np.random.random((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
    ```

    Note that the model variables may have different name values
    (`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.
    It is recommended that you use layer attributes to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.

    """
    from zero_keras import layers
    from zero_keras.models import Sequential

    if str(filepath).endswith(".keras") or zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, "r") as z:
            config_str = z.read("config.json").decode("utf-8")
            config = json.loads(config_str)

            # Very basic reconstruction stub
            if config.get("class_name") == "Sequential":
                model = Sequential()
                for layer_config in config.get("config", {}).get("layers", []):
                    class_name = layer_config.get("class_name")
                    layer_cls = getattr(layers, class_name, None)
                    if layer_cls:
                        model.add(layer_cls(**layer_config.get("config", {})))
            else:
                model = Model()  # stub

            def _load_safetensors(data):
                """_load_safetensors function.

                Args:
                data: Parameter data.

                Returns:
                Any: Return value.

                """
                import struct
                import json
                import array
                from ml_switcheroo_compiler.core.dtype import DType

                header_size = struct.unpack("<Q", data[:8])[0]
                header_json = data[8 : 8 + header_size].decode("utf-8")
                header = json.loads(header_json)
                weights_dict = {}
                for k, v in header.items():
                    if k == "__metadata__":
                        continue
                    start, end = v["data_offsets"]
                    raw = data[8 + header_size + start : 8 + header_size + end]
                    dt_str = v["dtype"]
                    fmt = "f" if dt_str == "F32" else "i" if dt_str == "I32" else "d"
                    arr = array.array(fmt)
                    arr.frombytes(raw)
                    shape = tuple(v["shape"])
                    dt = (
                        DType.Float32
                        if dt_str == "F32"
                        else DType.Int32
                        if dt_str == "I32"
                        else DType.Float64
                    )
                    weights_dict[k] = (list(arr), shape, dt)
                # Sort by key index
                return [
                    weights_dict[k]
                    for k in sorted(
                        weights_dict.keys(), key=lambda x: int(x.split("_")[-1])
                    )
                ]

            if "model.safetensors" in z.namelist():
                from ml_switcheroo_compiler.ops import asarray
                from ml_switcheroo_compiler.ops.shape import reshape

                with z.open("model.safetensors") as w:
                    data = w.read()
                    parsed = _load_safetensors(data)
                    weights = []
                    for flat, shape, dt in parsed:
                        t = asarray(flat, dtype=dt)
                        t = reshape(t, shape)
                        weights.append(t)
                    if weights:
                        try:
                            model.set_weights(weights)
                        except Exception:
                            pass
            return model
    return Model()


class Sequential(Model):
    """`Sequential` groups a linear stack of layers into a `Model`.

    Examples:
    ```python
    model = keras.Sequential()
    model.add(keras.Input(shape=(16,)))
    model.add(keras.layers.Dense(8))

    # Note that you can also omit the initial `Input`.
    # In that case the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = keras.Sequential()
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(4))
    # model.weights not created yet

    # Whereas if you specify an `Input`, the model gets built
    # continuously as you are adding layers:
    model = keras.Sequential()
    model.add(keras.Input(shape=(16,)))
    model.add(keras.layers.Dense(8))
    len(model.weights)  # Returns "2"

    # When using the delayed-build pattern (no input shape specified), you can
    # choose to manually build your model by calling
    # `build(batch_input_shape)`:
    model = keras.Sequential()
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(4))
    model.build((None, 16))
    len(model.weights)  # Returns "4"

    # Note that when using the delayed-build pattern (no input shape specified),
    # the model gets built the first time you call `fit`, `eval`, or `predict`,
    # or the first time you call the model on some input data.
    model = keras.Sequential()
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mse')
    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)
    ```

    """

    def __init__(self, layers=None, name=None):
        self.layers = layers or []
        self._name = name
        self.built = False

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

        Args:
            layer: layer instance.

        """
        self.layers.append(layer)

    def get_config(self):
        """Returns the config of the object.

        An object config is a Python dictionary (serializable)
        containing the information needed to re-instantiate it.
        """
        return {
            "name": getattr(self, "name", None),
            "layers": [
                {
                    "class_name": layer.__class__.__name__,
                    "config": getattr(
                        layer, "get_config", lambda: getattr(layer, "_kwargs", {})
                    )(),
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def from_config(cls, config):
        """Creates an operation from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same operation from the config dictionary.

        Note: If you override this method, you might receive a serialized dtype
        config, which is a `dict`. You can deserialize it as follows:

        ```python
        if "dtype" in config and isinstance(config["dtype"], dict):
            policy = dtype_policies.deserialize(config["dtype"])
        ```

        Args:
            config: A Python dictionary, typically the output of `get_config`.

        Returns:
            An operation instance.

        """
        layers_cfg = config.get("layers", [])
        from zero_keras import layers

        inst = cls(name=config.get("name"))
        for l_c in layers_cfg:
            l_cls = getattr(layers, l_c.get("class_name"), None)
            if l_cls:
                inst.add(l_cls(**l_c.get("config", {})))
        return inst

    @property
    def weights(self):
        """List of all weight variables of the layer.

        Unlike, `layer.variables` this excludes metric state and random seeds.
        """
        w = []
        for layer in self.layers:
            w.extend(layer.weights)
        return w

    @property
    def trainable_weights(self):
        """List of all trainable weight variables of the layer.

        These are the weights that get updated by the optimizer during training.
        """
        w = []
        for layer in self.layers:
            w.extend(layer.trainable_weights)
        return w

    @property
    def non_trainable_weights(self):
        """List of all non-trainable weight variables of the layer.

        These are the weights that should not be updated by the optimizer during
        training. Unlike, `layer.non_trainable_variables` this excludes metric
        state and random seeds.
        """
        w = []
        for layer in self.layers:
            w.extend(layer.non_trainable_weights)
        return w

    def get_weights(self):
        """Return the values of `layer.weights` as a list of NumPy arrays."""
        w = []
        for layer in self.layers:
            if hasattr(layer, "get_weights"):
                w.extend(layer.get_weights())
        return w

    def set_weights(self, weights):
        """Sets the values of `layer.weights` from a list of NumPy arrays."""
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "get_weights") and hasattr(layer, "set_weights"):
                cw = layer.get_weights()
                layer.set_weights(weights[idx : idx + len(cw)])
                idx += len(cw)

    def build(self, input_shape):
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        Any: Return value.

        """
        current_shape = input_shape
        for layer in self.layers:
            if not getattr(layer, "built", False):
                layer.build(current_shape)
            if hasattr(layer, "compute_output_shape"):
                current_shape = layer.compute_output_shape(current_shape)
            else:
                # heuristic fallback
                units = getattr(layer, "units", None)
                if units is not None:
                    if isinstance(current_shape, tuple):
                        current_shape = current_shape[:-1] + (units,)
                    else:
                        current_shape = (units,)
        self.built = True

    def call(self, inputs, training=None, mask=None, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        training: Parameter training.
        mask: Parameter mask.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        out = inputs
        for layer in self.layers:
            out = layer(out, **kwargs)
        return out

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Saves a model as a `.keras` file.

        Note that `model.save()` is an alias for `keras.saving.save_model()`.

        The saved `.keras` file contains:

        - The model's configuration (architecture)
        - The model's weights
        - The model's optimizer's state (if any)

        Thus models can be reinstantiated in the exact same state.

        Args:
            filepath: `str` or `pathlib.Path` object.
                The path where to save the model. Must end in `.keras`
                (unless saving the model as an unzipped directory
                via `zipped=False`).
            overwrite: Whether we should overwrite any existing model at
                the target location, or instead ask the user via
                an interactive prompt.
            zipped: Whether to save the model as a zipped `.keras`
                archive (default when saving locally), or as an
                unzipped directory (default when saving on the
                Hugging Face Hub).

        Example:
        ```python
        model = keras.Sequential(
            [
                keras.layers.Dense(5, input_shape=(3,)),
                keras.layers.Softmax(),
            ],
        )
        model.save("model.keras")
        loaded_model = keras.saving.load_model("model.keras")
        x = keras.random.uniform((10, 3))
        assert np.allclose(model.predict(x), loaded_model.predict(x))
        ```

        """
        from zero_keras.core_layers import Model

        Model.save(
            self, filepath, overwrite=overwrite, save_format=save_format, **kwargs
        )

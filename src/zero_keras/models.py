"""Model classes (`Sequential`, `Model`, `Functional`), cloning, and serialization for zero_keras."""

from __future__ import annotations

import array
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import struct
import zipfile

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops import asarray as _msc_asarray

from zero_keras.core_layers import Functional, Input, Layer, Model, Variable, deserialize
from zero_keras.ops import reshape
from zero_keras.saving.registry import get_object_class_name
from zero_keras.types import (
    ConfigDict,
    ConfigValue,
    ShapeLike,
    ShapeTuple,
    SupportsGetConfig,
    TensorLike,
)


def _load_safetensors(data: bytes) -> list[tuple[list[float | int], tuple[int, ...], DType]]:
    """Parse raw ``.safetensors`` bytes into ``(flat_values, shape, dtype)`` tuples.

    Args:
        data: Raw bytes of a ``.safetensors`` file.

    Returns:
        list[tuple[list[float | int], tuple[int, ...], DType]]: Ordered weight entries.
    """
    header_size = int(struct.unpack("<Q", data[:8])[0])
    header_json = data[8 : 8 + header_size].decode("utf-8")
    header = json.loads(header_json)
    weights_dict: dict[str, tuple[list[float | int], tuple[int, ...], DType]] = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        start, end = int(v["data_offsets"][0]), int(v["data_offsets"][1])
        raw = data[8 + header_size + start : 8 + header_size + end]
        dt_str = str(v["dtype"])
        fmt = "f" if dt_str == "F32" else ("i" if dt_str == "I32" else "d")
        arr = array.array(fmt)
        arr.frombytes(raw)
        shape = tuple(int(dim) for dim in v["shape"])
        dt = (
            DType.Float32
            if dt_str == "F32"
            else (DType.Int32 if dt_str == "I32" else DType.Float64)
        )
        weights_dict[str(k)] = (list(arr), shape, dt)

    return [
        weights_dict[k] for k in sorted(weights_dict.keys(), key=lambda x: int(x.split("_")[-1]))
    ]


def load_model(
    filepath: str | Path,
    custom_objects: dict[str, ConfigValue] | None = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: ConfigValue,
) -> Model:
    """Load a model saved via ``model.save()``.

    Args:
        filepath: String or ``Path`` to the saved ``.keras`` model file.
        custom_objects: Optional dictionary mapping names to custom classes or functions.
        compile: Whether to compile the model after loading.
        safe_mode: Whether to disallow unsafe ``Lambda`` deserialization.
        **kwargs: Additional keyword options.

    Returns:
        Model: Loaded Keras model instance.

    Raises:
        ValueError: If ``safe_mode=True`` and a ``Lambda`` is encountered, or if format is unsupported.
    """
    _ = compile, kwargs
    path_str = str(filepath)
    if path_str.endswith(".keras") or zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, "r") as z:
            config_str = z.read("config.json").decode("utf-8")
            config_dict = json.loads(config_str)
            if safe_mode and config_dict.get("class_name") == "Lambda":
                raise ValueError("safe_mode is True, cannot deserialize Lambda")

            model = model_from_json(config_str, custom_objects=custom_objects, safe_mode=safe_mode)

            if "model.safetensors" in z.namelist():
                with z.open("model.safetensors") as w:
                    data = w.read()
                    parsed = _load_safetensors(data)
                    weights: list[TensorLike] = []
                    for flat, shape, dt in parsed:
                        t = _msc_asarray(flat, dtype=dt)
                        t = reshape(t, shape)
                        weights.append(t)
                    if weights:
                        try:
                            model.set_weights(weights)
                        except Exception:
                            pass
            return model
    if path_str.endswith((".h5", ".hdf5")):
        raise ValueError(
            "Legacy HDF5 format is not supported in zero_keras. Please use .keras format."
        )
    raise ValueError(f"File format not supported for {filepath}")


class Sequential(Model):
    """``Sequential`` groups a linear stack of layers into a ``Model``."""

    inputs: TensorLike | None = None
    outputs: TensorLike | None = None

    def __init__(
        self,
        layers: Sequence[Layer | TensorLike] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a Sequential model.

        Args:
            layers: Optional sequence of layers to add to the model.
            name: Optional name for the model.
        """
        super().__init__(name=name)
        self.layers: list[Layer | TensorLike] = list(layers) if layers is not None else []
        self._name = name
        self.built = False
        self._build_input_shape: ShapeTuple | None = None

    def add(self, layer: Layer | TensorLike) -> None:
        """Add a layer instance on top of the layer stack.

        Args:
            layer: Layer instance to append.
        """
        self.layers.append(layer)

    def pop(self, rebuild: bool = True) -> Layer | TensorLike:
        """Remove the last layer in the model.

        Args:
            rebuild: Whether to rebuild the model after popping the layer.

        Returns:
            Layer | TensorLike: The removed layer instance.

        Raises:
            TypeError: If there are no layers in the model.
        """
        _ = rebuild
        if not self.layers:
            raise TypeError("There are no layers in the model.")
        return self.layers.pop()

    @property
    def input_shape(self) -> ShapeTuple | None:
        """Return the input shape tuple of the Sequential model.

        Returns:
            ShapeTuple | None: Input shape tuple if built, else ``None``.
        """
        return self._build_input_shape if self.built else None

    @property
    def output_shape(self) -> ShapeTuple | None:
        """Return the output shape tuple of the Sequential model.

        Returns:
            ShapeTuple | None: Output shape tuple if built, else ``None``.
        """
        if not self.built or self._build_input_shape is None:
            return None
        current_shape: ShapeTuple = self._build_input_shape
        for layer in self.layers:
            if isinstance(layer, Layer):
                current_shape = layer.compute_output_shape(current_shape)
        return current_shape

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration of the Sequential model.

        Returns:
            ConfigDict: Configuration dictionary containing ``name`` and ``layers``.
        """
        layer_configs: list[ConfigValue] = []
        for layer in self.layers:
            l_cfg = layer.get_config() if isinstance(layer, SupportsGetConfig) else {}
            layer_configs.append(
                {
                    "class_name": get_object_class_name(layer),
                    "config": l_cfg,
                }
            )
        return {
            "name": self._name,
            "layers": layer_configs,
        }

    @classmethod
    def from_config(cls, config: ConfigDict) -> Sequential:
        """Reconstruct a Sequential model from its configuration dictionary.

        Args:
            config: Configuration dictionary produced by ``get_config()``.

        Returns:
            Sequential: Reconstructed Sequential model.
        """
        from zero_keras.layers import LAYER_REGISTRY

        layers_cfg = config.get("layers", [])
        name_val = config.get("name")
        inst = cls(name=str(name_val) if isinstance(name_val, str) else None)
        if isinstance(layers_cfg, list):
            for l_c in layers_cfg:
                if isinstance(l_c, dict):
                    l_cls_name = l_c.get("class_name")
                    l_cls = (
                        LAYER_REGISTRY.get(str(l_cls_name)) if isinstance(l_cls_name, str) else None
                    )
                    if l_cls is not None:
                        sub_cfg = l_c.get("config", {})
                        inst.add(l_cls(**(sub_cfg if isinstance(sub_cfg, dict) else {})))
        return inst

    @property
    def weights(self) -> list[Variable]:
        """Return list of all weight variables across child layers.

        Returns:
            list[Variable]: All model weights.
        """
        w: list[Variable] = []
        for layer in self.layers:
            w.extend(layer.weights)
        return w

    @property
    def trainable_weights(self) -> list[Variable]:
        """Return list of all trainable weight variables across child layers.

        Returns:
            list[Variable]: Trainable model weights.
        """
        w: list[Variable] = []
        for layer in self.layers:
            w.extend(layer.trainable_weights)
        return w

    @property
    def non_trainable_weights(self) -> list[Variable]:
        """Return list of all non-trainable weight variables across child layers.

        Returns:
            list[Variable]: Non-trainable model weights.
        """
        w: list[Variable] = []
        for layer in self.layers:
            w.extend(layer.non_trainable_weights)
        return w

    def get_weights(self) -> list[TensorLike]:
        """Return the values of all layer weights as a flat list.

        Returns:
            list[TensorLike]: List of weight arrays/tensors.
        """
        w: list[TensorLike] = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                w.extend(layer.get_weights())
        return w

    def set_weights(self, weights: Sequence[TensorLike]) -> None:
        """Set the values of all layer weights from a flat sequence of arrays.

        Args:
            weights: Sequence of weight values matching ``get_weights()`` order.
        """
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                cw = layer.get_weights()
                layer.set_weights(list(weights[idx : idx + len(cw)]))
                idx += len(cw)

    def build(self, input_shape: ShapeLike | None = None) -> None:
        """Build child layers sequentially for ``input_shape``.

        Args:
            input_shape: Input shape tuple.
        """
        if isinstance(input_shape, tuple):
            self._build_input_shape = input_shape
        current_shape = input_shape
        for layer in self.layers:
            if isinstance(layer, Layer):
                if not layer.built:
                    layer.build(current_shape)
                current_shape = layer.compute_output_shape(current_shape)
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        training: bool | None = None,
        mask: TensorLike | None = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Execute all child layers sequentially on ``inputs``.

        Args:
            inputs: Input tensor.
            training: Optional boolean indicating training vs inference mode.
            mask: Optional input mask tensor.
            **kwargs: Additional keyword arguments passed to child layers.

        Returns:
            TensorLike: Final output tensor.
        """
        _ = training, mask
        out = inputs
        for layer in self.layers:
            out = layer(out, **kwargs)
        return out

    def save(
        self,
        filepath: str | Path,
        overwrite: bool = True,
        save_format: str | None = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Save the Sequential model to ``filepath``.

        Args:
            filepath: Target ``.keras`` file path.
            overwrite: Whether to overwrite existing files.
            save_format: Optional format string.
            **kwargs: Additional save options.
        """
        Model.save(self, filepath, overwrite=overwrite, save_format=save_format, **kwargs)


def _default_clone_function(layer: Layer | TensorLike) -> Layer | TensorLike:
    """Clone a single layer via ``from_config(layer.get_config())``.

    Args:
        layer: Layer instance to clone.

    Returns:
        Layer | TensorLike: Cloned layer instance.
    """
    if isinstance(layer, Layer):
        return type(layer).from_config(layer.get_config())
    return layer


def clone_model(
    model: Model,
    input_tensors: Sequence[TensorLike] | TensorLike | None = None,
    clone_function: Callable[[Layer | TensorLike], Layer | TensorLike] | None = None,
) -> Model:
    """Clone a ``Functional`` or ``Sequential`` ``Model`` instance with fresh weights.

    Args:
        model: Instance of ``Model`` or ``Sequential`` to clone.
        input_tensors: Optional input tensors to build the cloned model upon.
        clone_function: Optional callable applied to clone each layer.

    Returns:
        Model: Cloned model instance.

    Raises:
        ValueError: If ``model`` is not a ``Model`` instance.
    """
    _ = input_tensors
    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument to be a `Model` instance.")

    resolved_clone_fn = clone_function if clone_function is not None else _default_clone_function

    if isinstance(model, Sequential):
        cloned_seq = Sequential(name=model._name)
        for layer in model.layers:
            cloned_seq.add(resolved_clone_fn(layer))
        return cloned_seq

    config = model.get_config()
    layer_configs = config.get("layers", [])

    if not isinstance(layer_configs, list) or not layer_configs:
        return model

    created_layers: dict[str, Layer | TensorLike] = {}
    tensor_map: dict[str, TensorLike] = {}

    for lc in layer_configs:
        lc_dict = lc if isinstance(lc, dict) else {}
        lname = str(lc_dict.get("name", ""))
        cname = str(lc_dict.get("class_name", ""))
        cconfig_raw = lc_dict.get("config", {})
        cconfig = cconfig_raw if isinstance(cconfig_raw, dict) else {}

        if cname == "InputLayer":
            shape_raw = cconfig.get("batch_input_shape", (None,))
            raw_seq = list(shape_raw) if isinstance(shape_raw, (list, tuple)) else [None]
            shape_items = raw_seq[1:] if len(raw_seq) > 0 and raw_seq[0] is None else raw_seq
            shape: ShapeTuple = tuple(
                int(d) if isinstance(d, (int, float)) else None for d in shape_items
            )
            dt_val = cconfig.get("dtype")
            inp_tensor = Input(
                shape=shape,
                name=lname,
                dtype=str(dt_val) if isinstance(dt_val, str) else None,
            )
            created_layers[lname] = inp_tensor
            tensor_map[lname] = inp_tensor
        else:
            orig_by_name = {lyr.name: lyr for lyr in model.layers}
            created_layers[lname] = resolved_clone_fn(orig_by_name[lname])

    for lc in layer_configs:
        lc_dict = lc if isinstance(lc, dict) else {}
        lname = str(lc_dict.get("name", ""))
        inbound = lc_dict.get("inbound_nodes", [])
        if isinstance(inbound, list) and inbound:
            layer_obj = created_layers[lname]
            for connection_group in inbound:
                cg = list(connection_group) if isinstance(connection_group, (list, tuple)) else []
                if len(cg) == 1:
                    conn0 = list(cg[0]) if isinstance(cg[0], (list, tuple)) else []
                    src_layer, call_kwargs = conn0[0], conn0[3]
                    inp_t = tensor_map[str(src_layer)]
                    kw = dict(call_kwargs) if isinstance(call_kwargs, dict) else {}
                    tensor_map[lname] = layer_obj(inp_t, **kw) if callable(layer_obj) else inp_t
                else:
                    inp_ts = [tensor_map[str(c[0])] for c in cg if isinstance(c, (list, tuple))]
                    tensor_map[lname] = layer_obj(inp_ts) if callable(layer_obj) else inp_ts

    input_layers_raw = config.get("input_layers", [])
    output_layers_raw = config.get("output_layers", [])
    input_layers = input_layers_raw if isinstance(input_layers_raw, list) else []
    output_layers = output_layers_raw if isinstance(output_layers_raw, list) else []

    inputs_list = [tensor_map[str(il[0])] for il in input_layers if isinstance(il, (list, tuple))]
    outputs_list = [tensor_map[str(ol[0])] for ol in output_layers if isinstance(ol, (list, tuple))]

    inputs_arg: TensorLike | list[TensorLike] = (
        inputs_list[0] if len(inputs_list) == 1 else inputs_list
    )
    outputs_arg: TensorLike | list[TensorLike] = (
        outputs_list[0] if len(outputs_list) == 1 else outputs_list
    )
    name_val = config.get("name")
    return Functional(
        inputs=inputs_arg,
        outputs=outputs_arg,
        name=str(name_val) if isinstance(name_val, str) else None,
    )


def save_model(
    model: Model | TensorLike,
    filepath: str | Path,
    overwrite: bool = True,
    **kwargs: ConfigValue,
) -> None:
    """Save a model as a ``.keras`` file.

    Args:
        model: Model instance to save.
        filepath: Destination file path.
        overwrite: Whether to overwrite existing files.
        **kwargs: Additional save options.
    """
    if isinstance(model, Model):
        model.save(filepath, overwrite=overwrite, **kwargs)
    else:
        Model.save(model, filepath, overwrite=overwrite, **kwargs)


def model_from_json(
    json_string: str,
    custom_objects: dict[str, ConfigValue] | None = None,
    safe_mode: bool = True,
) -> Model:
    """Reconstruct a model from a JSON configuration string.

    Args:
        json_string: JSON-encoded model configuration.
        custom_objects: Optional dictionary of custom classes.
        safe_mode: Whether to disallow unsafe ``Lambda`` deserialization.

    Returns:
        Model: Reconstructed model instance.

    Raises:
        ValueError: If ``safe_mode=True`` and a ``Lambda`` layer is present.
    """
    config = json.loads(json_string)
    class_name = config.get("class_name")

    if class_name == "Sequential":
        sub_cfg = config.get("config", {})
        return Sequential.from_config(sub_cfg if isinstance(sub_cfg, dict) else {})

    if class_name in ("Functional", "Model"):
        conf_raw = config.get("config", {})
        conf = conf_raw if isinstance(conf_raw, dict) else {}
        layer_configs = conf.get("layers", [])

        created_layers: dict[str, Layer | TensorLike] = {}
        tensor_map: dict[str, TensorLike] = {}

        if isinstance(layer_configs, list):
            for lc in layer_configs:
                lc_dict = lc if isinstance(lc, dict) else {}
                lname = str(lc_dict.get("name", ""))
                cname = str(lc_dict.get("class_name", ""))
                cconfig_raw = lc_dict.get("config", {})
                cconfig = cconfig_raw if isinstance(cconfig_raw, dict) else {}

                if safe_mode and cname == "Lambda":
                    raise ValueError("safe_mode is True, cannot deserialize Lambda")

                if cname == "InputLayer":
                    shape_raw = cconfig.get("batch_input_shape", (None,))
                    raw_seq = list(shape_raw) if isinstance(shape_raw, (list, tuple)) else [None]
                    shape_items = (
                        raw_seq[1:] if len(raw_seq) > 0 and raw_seq[0] is None else raw_seq
                    )
                    shape: ShapeTuple = tuple(
                        int(d) if isinstance(d, (int, float)) else None for d in shape_items
                    )
                    dt_val = cconfig.get("dtype")
                    inp_t = Input(
                        shape=shape,
                        name=lname,
                        dtype=str(dt_val) if isinstance(dt_val, str) else None,
                    )
                    created_layers[lname] = inp_t
                    tensor_map[lname] = inp_t
                else:
                    created_layers[lname] = deserialize(
                        lc_dict, custom_objects=custom_objects, safe_mode=safe_mode
                    )

            for lc in layer_configs:
                lc_dict = lc if isinstance(lc, dict) else {}
                lname = str(lc_dict.get("name", ""))
                inbound = lc_dict.get("inbound_nodes", [])
                if isinstance(inbound, list) and inbound:
                    layer_obj = created_layers[lname]
                    for connection_group in inbound:
                        cg = (
                            list(connection_group)
                            if isinstance(connection_group, (list, tuple))
                            else []
                        )
                        if len(cg) == 1:
                            conn0 = list(cg[0]) if isinstance(cg[0], (list, tuple)) else []
                            src_layer, call_kwargs = conn0[0], conn0[3]
                            inp_t = tensor_map[str(src_layer)]
                            kw = dict(call_kwargs) if isinstance(call_kwargs, dict) else {}
                            tensor_map[lname] = (
                                layer_obj(inp_t, **kw) if callable(layer_obj) else inp_t
                            )
                        else:
                            inp_ts = [
                                tensor_map[str(c[0])] for c in cg if isinstance(c, (list, tuple))
                            ]
                            tensor_map[lname] = layer_obj(inp_ts) if callable(layer_obj) else inp_ts

        input_layers_raw = conf.get("input_layers", [])
        output_layers_raw = conf.get("output_layers", [])
        input_layers = input_layers_raw if isinstance(input_layers_raw, list) else []
        output_layers = output_layers_raw if isinstance(output_layers_raw, list) else []

        inputs_list = [
            tensor_map[str(il[0])] for il in input_layers if isinstance(il, (list, tuple))
        ]
        outputs_list = [
            tensor_map[str(ol[0])] for ol in output_layers if isinstance(ol, (list, tuple))
        ]

        inputs_arg: TensorLike | list[TensorLike] = (
            inputs_list[0] if len(inputs_list) == 1 else inputs_list
        )
        outputs_arg: TensorLike | list[TensorLike] = (
            outputs_list[0] if len(outputs_list) == 1 else outputs_list
        )
        name_val = conf.get("name")
        return Functional(
            inputs=inputs_arg,
            outputs=outputs_arg,
            name=str(name_val) if isinstance(name_val, str) else None,
        )

    return Model()

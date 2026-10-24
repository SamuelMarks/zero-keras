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
from zero_keras.types import ConfigDict, ConfigValue, ShapeLike, TensorLike


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

    def __init__(
        self,
        layers: Sequence[Layer] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a Sequential model.

        Args:
            layers: Optional sequence of layers to add to the model.
            name: Optional name for the model.
        """
        super().__init__(name=name)
        self.layers: list[Layer] = list(layers) if layers is not None else []
        self._name = name
        self.built = False

    def add(self, layer: Layer) -> None:
        """Add a layer instance on top of the layer stack.

        Args:
            layer: Layer instance to append.
        """
        self.layers.append(layer)

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration of the Sequential model.

        Returns:
            ConfigDict: Configuration dictionary containing ``name`` and ``layers``.
        """
        layer_configs: list[ConfigValue] = []
        for layer in self.layers:
            l_cfg = layer.get_config() if hasattr(layer, "get_config") else {}
            layer_configs.append(
                {
                    "class_name": layer.__class__.__name__,
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
        layers_cfg = config.get("layers", [])
        from zero_keras import layers as layers_module

        name_val = config.get("name")
        inst = cls(name=str(name_val) if isinstance(name_val, str) else None)
        if isinstance(layers_cfg, list):
            for l_c in layers_cfg:
                if isinstance(l_c, dict):
                    cls_name = str(l_c.get("class_name", ""))
                    if hasattr(layers_module, cls_name):
                        l_cls = layers_module.__dict__[cls_name]
                        sub_cfg = l_c.get("config", {})
                        inst.add(l_cls(**sub_cfg))  # type: ignore[arg-type]
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
            if hasattr(layer, "get_weights"):
                w.extend(layer.get_weights())
        return w

    def set_weights(self, weights: Sequence[TensorLike]) -> None:
        """Set the values of all layer weights from a flat sequence of arrays.

        Args:
            weights: Sequence of weight values matching ``get_weights()`` order.
        """
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "get_weights") and hasattr(layer, "set_weights"):
                cw = layer.get_weights()
                layer.set_weights(list(weights[idx : idx + len(cw)]))
                idx += len(cw)

    def build(self, input_shape: ShapeLike | None = None) -> None:
        """Build child layers sequentially for ``input_shape``.

        Args:
            input_shape: Input shape tuple.
        """
        current_shape = input_shape
        for layer in self.layers:
            is_built = bool(layer.built) if hasattr(layer, "built") else False
            if not is_built and hasattr(layer, "build"):
                layer.build(current_shape)
            if hasattr(layer, "compute_output_shape"):
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


def _default_clone_function(layer: Layer) -> Layer:
    """Clone a single layer via ``from_config(layer.get_config())``.

    Args:
        layer: Layer instance to clone.

    Returns:
        Layer: Cloned layer instance.
    """
    if hasattr(layer, "get_config"):
        return layer.__class__.from_config(layer.get_config())
    return layer


def clone_model(
    model: Model,
    input_tensors: Sequence[TensorLike] | TensorLike | None = None,
    clone_function: Callable[[Layer], Layer] | None = None,
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
        lname = str(lc.get("name", ""))  # type: ignore[union-attr]
        cname = str(lc.get("class_name", ""))  # type: ignore[union-attr]
        cconfig_raw = lc.get("config", {})  # type: ignore[union-attr]
        cconfig = cconfig_raw if isinstance(cconfig_raw, dict) else {}

        if cname == "InputLayer":
            shape_raw = cconfig.get("batch_input_shape", (None,))
            shape = (
                tuple(shape_raw[1:])
                if isinstance(shape_raw, (list, tuple))
                and len(shape_raw) > 0
                and shape_raw[0] is None
                else shape_raw
            )
            dt_val = cconfig.get("dtype")
            inp_tensor = Input(
                shape=shape,  # type: ignore[arg-type]
                name=lname,
                dtype=str(dt_val) if isinstance(dt_val, str) else None,
            )
            created_layers[lname] = inp_tensor
            tensor_map[lname] = inp_tensor
        else:
            orig_by_name = {lyr.name: lyr for lyr in model.layers}
            created_layers[lname] = resolved_clone_fn(orig_by_name[lname])

    for lc in layer_configs:
        lname = str(lc.get("name", ""))  # type: ignore[union-attr]
        inbound = lc.get("inbound_nodes", [])  # type: ignore[union-attr]
        if not isinstance(inbound, list) or not inbound:
            continue

        layer_obj = created_layers[lname]
        for connection_group in inbound:
            if len(connection_group) == 1:  # type: ignore[arg-type]
                src_layer, _node_idx, _tensor_idx, call_kwargs = connection_group[0]  # type: ignore[index]
                inp_t = tensor_map[str(src_layer)]
                kw = call_kwargs if isinstance(call_kwargs, dict) else {}
                out_t = layer_obj(inp_t, **kw)  # type: ignore[operator]
            else:
                inp_ts = [tensor_map[str(c[0])] for c in connection_group]  # type: ignore[union-attr]
                out_t = layer_obj(inp_ts)  # type: ignore[operator]
            tensor_map[lname] = out_t

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
    model: Model,
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
    if hasattr(model, "save"):
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
                lname = str(lc.get("name", ""))  # type: ignore[union-attr]
                cname = str(lc.get("class_name", ""))  # type: ignore[union-attr]
                cconfig_raw = lc.get("config", {})  # type: ignore[union-attr]
                cconfig = cconfig_raw if isinstance(cconfig_raw, dict) else {}

                if safe_mode and cname == "Lambda":
                    raise ValueError("safe_mode is True, cannot deserialize Lambda")

                if cname == "InputLayer":
                    shape_raw = cconfig.get("batch_input_shape", (None,))
                    shape = (
                        tuple(shape_raw[1:])
                        if isinstance(shape_raw, (list, tuple))
                        and len(shape_raw) > 0
                        and shape_raw[0] is None
                        else shape_raw
                    )
                    dt_val = cconfig.get("dtype")
                    inp_t = Input(
                        shape=shape,  # type: ignore[arg-type]
                        name=lname,
                        dtype=str(dt_val) if isinstance(dt_val, str) else None,
                    )
                    created_layers[lname] = inp_t
                    tensor_map[lname] = inp_t
                else:
                    created_layers[lname] = deserialize(
                        lc, custom_objects=custom_objects, safe_mode=safe_mode
                    )

            for lc in layer_configs:
                lname = str(lc.get("name", ""))  # type: ignore[union-attr]
                inbound = lc.get("inbound_nodes", [])  # type: ignore[union-attr]
                if not isinstance(inbound, list) or not inbound:
                    continue

                layer_obj = created_layers[lname]
                for connection_group in inbound:
                    if len(connection_group) == 1:  # type: ignore[arg-type]
                        src_layer, _node_idx, _tensor_idx, call_kwargs = connection_group[0]  # type: ignore[index]
                        inp_tensor = tensor_map[str(src_layer)]
                        kw = call_kwargs if isinstance(call_kwargs, dict) else {}
                        out_tensor = layer_obj(inp_tensor, **kw)  # type: ignore[operator]
                    else:
                        inp_tensors = [tensor_map[str(c[0])] for c in connection_group]  # type: ignore[union-attr]
                        out_tensor = layer_obj(inp_tensors)  # type: ignore[operator]
                    tensor_map[lname] = out_tensor

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

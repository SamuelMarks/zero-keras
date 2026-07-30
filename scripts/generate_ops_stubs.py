"""Module docstring."""

import inspect

import keras
import ml_switcheroo_compiler.ops as msc_ops

ops_m = keras.ops
valid_ops = {name for name in dir(ops_m) if not name.startswith("_")}
for name in dir(msc_ops):
    if not name.startswith("_"):
        valid_ops.add(name)

valid_ops.update(
    [
        "conv1d_transpose",
        "conv2d_transpose",
        "conv3d_transpose",
        "dropout",
        "where",
        "reshape",
        "conv1d",
        "conv2d",
        "conv3d",
        "tile",
        "var",
        "pad",
        "take",
        "searchsorted",
        "repeat",
        "squeeze",
        "take_along_axis",
        "newaxis",
        "qr",
        "rnn",
        "dynamic_update_slice",
        "split",
        "top_k",
        "sort",
        "reverse",
    ]
)

valid_ops = sorted(valid_ops)

# Mappings from Keras API names to ml-switcheroo-compiler ops names
# if they differ.
RENAMED_OPS = {}

# Code blocks to manually inject into ops.py
MANUAL_CODE_BLOCKS = {
    "reverse": """
from ml_switcheroo_compiler.ops.shape.frontend import flip as msc_flip
def reverse(*args, **kwargs):
    if "dims" in kwargs:
        kwargs["axis"] = kwargs.pop("dims")
    elif "axes" in kwargs:
        kwargs["axis"] = kwargs.pop("axes")
    return _wrap_op("reverse", msc_flip)(*args, **kwargs)
""",
    "sort": """
from ml_switcheroo_compiler.ops.shape.misc import sort as msc_sort
def sort(*args, **kwargs):
    if "axis" in kwargs:
        kwargs["dim"] = kwargs.pop("axis")
    return _wrap_op("sort", msc_sort)(*args, **kwargs)
""",
    "image": """
class image:
    @staticmethod
    def resize(*args, **kwargs):
        return _wrap_op("resize", getattr(msc_ops, "resize"))(*args, **kwargs)
    
    @staticmethod
    def crop_images(*args, **kwargs):
        return _wrap_op("crop_images", getattr(msc_ops, "crop_images"))(*args, **kwargs)
        
    @staticmethod
    def extract_patches(*args, **kwargs):
        return _wrap_op("extract_patches", getattr(msc_ops, "extract_patches"))(*args, **kwargs)
        
    @staticmethod
    def affine_transform(*args, **kwargs):
        return _wrap_op("affine_transform", getattr(msc_ops, "affine_transform"))(*args, **kwargs)

    @staticmethod
    def elastic_transform(*args, **kwargs):
        return _wrap_op("elastic_transform", getattr(msc_ops, "elastic_transform"))(*args, **kwargs)
        
    @staticmethod
    def gaussian_blur(*args, **kwargs):
        return _wrap_op("gaussian_blur", getattr(msc_ops, "gaussian_blur"))(*args, **kwargs)
        
    @staticmethod
    def hsv_to_rgb(*args, **kwargs):
        return _wrap_op("hsv_to_rgb", getattr(msc_ops, "hsv_to_rgb"))(*args, **kwargs)
        
    @staticmethod
    def map_coordinates(*args, **kwargs):
        return _wrap_op("map_coordinates", getattr(msc_ops, "map_coordinates"))(*args, **kwargs)
        
    @staticmethod
    def pad_images(*args, **kwargs):
        return _wrap_op("pad_images", getattr(msc_ops, "pad_images"))(*args, **kwargs)
        
    @staticmethod
    def perspective_transform(*args, **kwargs):
        return _wrap_op("perspective_transform", getattr(msc_ops, "perspective_transform"))(*args, **kwargs)
        
    @staticmethod
    def rgb_to_grayscale(*args, **kwargs):
        return _wrap_op("rgb_to_grayscale", getattr(msc_ops, "rgb_to_grayscale"))(*args, **kwargs)
        
    @staticmethod
    def rgb_to_hsv(*args, **kwargs):
        return _wrap_op("rgb_to_hsv", getattr(msc_ops, "rgb_to_hsv"))(*args, **kwargs)

    @staticmethod
    def random_flip(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.affine import random_flip as rf
        return _wrap_op("random_flip", rf)(*args, **kwargs)

    @staticmethod
    def random_rotation(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.affine import random_rotation as rr
        return _wrap_op("random_rotation", rr)(*args, **kwargs)

    @staticmethod
    def adjust_contrast(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.color import adjust_contrast as ac
        return _wrap_op("adjust_contrast", ac)(*args, **kwargs)

    @staticmethod
    def auto_contrast(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.color import auto_contrast as ac
        return _wrap_op("auto_contrast", ac)(*args, **kwargs)

    @staticmethod
    def equalization(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.color import equalization as eq
        return _wrap_op("equalization", eq)(*args, **kwargs)

    @staticmethod
    def resize_bilinear(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import resize_bilinear as rb
        return _wrap_op("resize_bilinear", rb)(*args, **kwargs)

    @staticmethod
    def resize_nearest(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import resize_nearest as rn
        return _wrap_op("resize_nearest", rn)(*args, **kwargs)
    @staticmethod
    def resize_bicubic(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import resize_bicubic as rbc
        return _wrap_op("resize_bicubic", rbc)(*args, **kwargs)

    @staticmethod
    def resize_lanczos3(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import resize_lanczos3 as rl3
        return _wrap_op("resize_lanczos3", rl3)(*args, **kwargs)

""",
    "tril": """
def tril(x, k=0):
    return _wrap_op("band_part", getattr(msc_ops, "band_part"))(x, -1, k)
""",
    "triu": """
def triu(x, k=0):
    return _wrap_op("band_part", getattr(msc_ops, "band_part"))(x, k, -1)
""",
    "where": """
from ml_switcheroo_compiler.ops.shape.indexing import where as msc_where
def where(*args, **kwargs):
    return _wrap_op("where", msc_where)(*args, **kwargs)
""",
    "dropout": """
from ml_switcheroo_compiler.ops.nn.dropout import dropout as msc_dropout
def dropout(*args, **kwargs):
    return _wrap_op("dropout", msc_dropout)(*args, **kwargs)
""",
    "conv1d_transpose": """
from ml_switcheroo_compiler.ops.nn.conv_nd import conv_transpose as msc_conv_transpose
def conv1d_transpose(*args, **kwargs):
    return _wrap_op("conv1d_transpose", msc_conv_transpose)(*args, **kwargs)
""",
    "conv2d_transpose": """
from ml_switcheroo_compiler.ops.nn.conv_nd import conv_transpose as msc_conv_transpose
def conv2d_transpose(*args, **kwargs):
    return _wrap_op("conv2d_transpose", msc_conv_transpose)(*args, **kwargs)
""",
    "conv3d_transpose": """
from ml_switcheroo_compiler.ops.nn.conv_nd import conv_transpose as msc_conv_transpose
def conv3d_transpose(*args, **kwargs):
    return _wrap_op("conv3d_transpose", msc_conv_transpose)(*args, **kwargs)
""",
    "reshape": """
from ml_switcheroo_compiler.ops.shape.frontend import reshape as msc_reshape
def reshape(*args, **kwargs):
    return _wrap_op("reshape", msc_reshape)(*args, **kwargs)
""",
    "conv1d": """
from ml_switcheroo_compiler.ops.nn.conv1d import conv1d as msc_conv1d
def conv1d(*args, **kwargs):
    return _wrap_op("conv1d", msc_conv1d)(*args, **kwargs)
""",
    "conv2d": """
from ml_switcheroo_compiler.ops.nn.conv2d import conv2d as msc_conv2d
def conv2d(*args, **kwargs):
    return _wrap_op("conv2d", msc_conv2d)(*args, **kwargs)
""",
    "conv3d": """
from ml_switcheroo_compiler.ops.nn.conv3d import conv3d as msc_conv3d
def conv3d(*args, **kwargs):
    return _wrap_op("conv3d", msc_conv3d)(*args, **kwargs)
""",
    "tile": """
from ml_switcheroo_compiler.ops.shape.misc import tile as msc_tile
def tile(*args, **kwargs):
    return _wrap_op("tile", msc_tile)(*args, **kwargs)
""",
    "var": """
def var(*args, **kwargs):
    return _wrap_op("variance", getattr(msc_ops, "variance"))(*args, **kwargs)
""",
    "pad": """
from ml_switcheroo_compiler.ops.shape.misc import pad as msc_pad
def pad(*args, **kwargs):
    return _wrap_op("pad", msc_pad)(*args, **kwargs)
""",
    "take": """
from ml_switcheroo_compiler.ops.shape.indexing import take as msc_take
def take(*args, **kwargs):
    return _wrap_op("take", msc_take)(*args, **kwargs)
""",
    "searchsorted": """
from ml_switcheroo_compiler.ops.shape.indexing import searchsorted as msc_searchsorted
def searchsorted(*args, **kwargs):
    return _wrap_op("searchsorted", msc_searchsorted)(*args, **kwargs)
""",
    "repeat": """
from ml_switcheroo_compiler.ops.shape.misc import repeat as msc_repeat
def repeat(*args, **kwargs):
    if "axis" in kwargs:
        kwargs["dim"] = kwargs.pop("axis")
    return _wrap_op("repeat", msc_repeat)(*args, **kwargs)
""",
    "squeeze": """
from ml_switcheroo_compiler.ops.shape.frontend import squeeze as msc_squeeze
def squeeze(*args, **kwargs):
    return _wrap_op("squeeze", msc_squeeze)(*args, **kwargs)
""",
    "take_along_axis": """
from ml_switcheroo_compiler.ops.shape.indexing import take_along_axis as msc_take_along_axis
def take_along_axis(*args, **kwargs):
    return _wrap_op("take_along_axis", msc_take_along_axis)(*args, **kwargs)
""",
    "newaxis": """
newaxis = None
""",
    "qr": """
from ml_switcheroo_compiler.ops.linalg.decompositions.qr import qr as msc_qr
def qr(*args, **kwargs):
    return _wrap_op("qr", msc_qr)(*args, **kwargs)
""",
    "rnn": """
from ml_switcheroo_compiler.ops.nn.rnn_utils import rnn as msc_rnn, RNNConfig as MSC_RNNConfig
def rnn(inputs, initial_state, cell_fn, go_backwards=False, mask=None, unroll=False, input_length=None, time_major=False, return_all_outputs=True, zero_output_for_mask=False, **kwargs):
    config = MSC_RNNConfig(
        time_major=time_major,
        go_backwards=go_backwards,
        unroll=unroll,
        return_all_outputs=return_all_outputs
    )
    return _wrap_op("rnn", msc_rnn)(inputs, initial_state, cell_fn, config)
""",
    "dynamic_update_slice": """
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import dynamic_update_slice as msc_dynamic_update_slice
def dynamic_update_slice(*args, **kwargs):
    return _wrap_op("dynamic_update_slice", msc_dynamic_update_slice)(*args, **kwargs)
""",
    "split": """
from ml_switcheroo_compiler.ops.shape.splitting import split as msc_split
def split(*args, **kwargs):
    if "axis" in kwargs:
        kwargs["dim"] = kwargs.pop("axis")
    return _wrap_op("split", msc_split)(*args, **kwargs)
""",
    "top_k": """
from ml_switcheroo_compiler.ops.shape.misc import top_k as msc_top_k
def top_k(*args, **kwargs):
    return _wrap_op("top_k", msc_top_k)(*args, **kwargs)
""",
    "transpose": """
from ml_switcheroo_compiler.ops.shape.frontend import permute as msc_permute
def transpose(*args, **kwargs):
    if "axes" in kwargs:
        kwargs["dims"] = kwargs.pop("axes")
    elif len(args) > 1:
        args = list(args)
        dims = args.pop(1)
        kwargs["dims"] = dims
    return _wrap_op("transpose", msc_permute)(*args, **kwargs)
""",
}

with open("src/zero_keras/ops.py", "w") as f:
    f.write('"""Ops wrapper module."""\n\n')
    f.write("import builtins\n")
    f.write("import ml_switcheroo_compiler.ops as msc_ops\n")
    f.write(
        "from ml_switcheroo_compiler.utils.operation_utils import compute_shape_propagation\n\n"
    )

    f.write("def _wrap_op(name, msc_op):\n")
    f.write("    def wrapper(*args, **kwargs):\n")
    f.write("        from zero_keras.core_layers import KerasTensor\n")
    f.write("        flat_args = []\n")
    f.write("        for a in args:\n")
    f.write("            if isinstance(a, (list, tuple)):\n")
    f.write("                flat_args.extend(a)\n")
    f.write("            else:\n")
    f.write("                flat_args.append(a)\n")
    f.write("        for a in kwargs.values():\n")
    f.write("            if isinstance(a, (list, tuple)):\n")
    f.write("                flat_args.extend(a)\n")
    f.write("            else:\n")
    f.write("                flat_args.append(a)\n\n")
    f.write("        has_kt = False\n")
    f.write("        for x in flat_args:\n")
    f.write("            if isinstance(x, KerasTensor):\n")
    f.write("                has_kt = True\n")
    f.write("                break\n\n")
    f.write("        if has_kt:\n")
    f.write(
        "            kt = next(x for x in flat_args if isinstance(x, KerasTensor))\n"
    )
    f.write(
        "            shape_or_shapes = compute_shape_propagation(name, kt.shape, args, kwargs)\n"
    )
    f.write("            if isinstance(shape_or_shapes, list):\n")
    f.write(
        "                return [KerasTensor(s, dtype=kt.dtype) for s in shape_or_shapes]\n"
    )
    f.write("            return KerasTensor(shape_or_shapes, dtype=kt.dtype)\n")
    f.write("        return msc_op(*args, **kwargs)\n")
    f.write("    return wrapper\n\n")

    for op in valid_ops:
        if op in MANUAL_CODE_BLOCKS:
            f.write(MANUAL_CODE_BLOCKS[op].strip() + "\n\n")
            continue

        msc_op_name = RENAMED_OPS.get(op, op)
        msc_op = getattr(msc_ops, msc_op_name, None)

        # Fallback to checking msc_ops.activations if it's an activation function
        if msc_op is None and hasattr(msc_ops, "activations"):
            msc_op = getattr(msc_ops.activations, msc_op_name, None)
            if msc_op is not None:
                # Provide a wrapper that calls the activation
                f.write(f"def {op}(*args, **kwargs):\n")
                f.write(
                    f'    return _wrap_op("{msc_op_name}", getattr(msc_ops.activations, "{msc_op_name}"))(*args, **kwargs)\n\n'
                )
                continue

        # If it's a module, check if it has a callable with the same name inside it.
        # This handles cases like `msc_ops.reshape` being a module, but Keras wanting the `msc_ops.reshape.reshape` function.
        if (
            msc_op is not None
            and not callable(msc_op)
            and type(msc_op).__name__ == "module"
        ):
            inner_op = getattr(msc_op, msc_op_name, None)
            if inner_op is not None and callable(inner_op):
                msc_op = inner_op
                # We need to construct the getattr path properly
                f.write(f"def {op}(*args, **kwargs):\n")
                f.write(
                    f'    return _wrap_op("{msc_op_name}", getattr(msc_ops.{msc_op_name}, "{msc_op_name}"))(*args, **kwargs)\n\n'
                )
                continue

        if msc_op is None:
            f.write(f"def {op}(*args, **kwargs):\n")
            f.write(
                f'    raise NotImplementedError("ops.{op} is not yet implemented in ml-switcheroo-compiler")\n\n'
            )
        elif not callable(msc_op):
            if type(msc_op).__name__ == "module" or inspect.isclass(msc_op):
                f.write(f"{op} = msc_ops.{msc_op_name}\n\n")
            else:
                f.write(f"{op} = msc_ops.{msc_op_name}\n\n")
        else:
            f.write(f"def {op}(*args, **kwargs):\n")
            f.write(
                f'    return _wrap_op("{msc_op_name}", getattr(msc_ops, "{msc_op_name}"))(*args, **kwargs)\n\n'
            )

    # Export ops module reference
    f.write("class _OpsNamespace:\n")
    f.write("    def __getattr__(self, name):\n")
    f.write("        if name in globals():\n")
    f.write("            return globals()[name]\n")
    f.write(
        "        raise AttributeError(f\"module 'zero_keras.ops' has no attribute '{name}'\")\n\n"
    )
    f.write("ops = _OpsNamespace()\n")

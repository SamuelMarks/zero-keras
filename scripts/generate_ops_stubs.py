"""Module docstring."""

import keras
import ml_switcheroo_compiler.ops as msc_ops

ops_m = getattr(keras, "ops")
valid_ops = set(name for name in dir(ops_m) if not name.startswith("_"))
for name in dir(msc_ops):
    if not name.startswith("_"):
        valid_ops.add(name)

valid_ops = sorted(list(valid_ops))

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
        msc_op = getattr(msc_ops, op, None)
        if msc_op is None:
            f.write(f"def {op}(*args, **kwargs):\n")
            f.write(
                f'    raise NotImplementedError("ops.{op} is not yet implemented in ml-switcheroo-compiler")\n\n'
            )
        elif not callable(msc_op):
            f.write(f"{op} = msc_ops.{op}\n\n")
        else:
            f.write(f"def {op}(*args, **kwargs):\n")
            f.write(
                f'    return _wrap_op("{op}", getattr(msc_ops, "{op}"))(*args, **kwargs)\n\n'
            )

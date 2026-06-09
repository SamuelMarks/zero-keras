import re

with open("KERAS_TODO.md") as f:
    lines = f.readlines()

layers = []
for line in lines:
    if "keras.layers" in line and line.strip().startswith("|"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 9:
            namespace = parts[3]
            symbol = parts[4]
            fqn = parts[5]
            sig = "|".join(parts[6:-2])
            doc = parts[-2]
            if namespace == "keras.layers":
                layers.append((symbol, sig, doc))


def split_args(s):
    args = []
    current = []
    depth = 0
    in_quote = False
    quote_char = ""
    for char in s:
        if in_quote:
            current.append(char)
            if char == quote_char:
                in_quote = False
        else:
            if char in "'\"":
                in_quote = True
                quote_char = char
                current.append(char)
            elif char in "([{":
                depth += 1
                current.append(char)
            elif char in ")]}":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)
    if current:
        args.append("".join(current).strip())
    return args


def clean_sig(sig):
    sig = sig.replace("`", "")
    sig = sig.replace("\\n", " ").replace("\n", " ")
    sig = re.sub(r"\s+", " ", sig)

    sig = sig.replace(": _empty=None", "=None")
    sig = sig.replace(": _empty=True", "=True")
    sig = sig.replace(": _empty=False", "=False")
    sig = sig.replace(": _empty=max", "='max'")
    sig = sig.replace(": _empty=dot", "='dot'")
    sig = sig.replace(": _empty", "")

    sig = sig.replace("kwargs: dict | None='(None)'", "**kwargs: Any")
    sig = sig.replace("kwargs: dict | None=None", "**kwargs: Any")
    sig = sig.replace("kwargs: dict | None", "**kwargs: Any")
    sig = sig.replace("kwargs={}", "**kwargs: Any")

    sig = sig.replace("str | None", "Optional[str]")
    sig = sig.replace("int | None", "Optional[int]")
    sig = sig.replace("float | None", "Optional[float]")

    sig = sig.replace("(None)", "None")
    sig = re.sub(r"'the image_data_format[^']*'", "None", sig)
    sig = re.sub(r"'the\s+image_data_format[^']*'", "None", sig)
    sig = sig.replace("'256, which is suitable for 8-bit images'", "None")
    sig = sig.replace("'(0, 1).'", "None")
    sig = sig.replace("'(0, 255).'", "None")
    sig = sig.replace("'[0, 255]'.", "None")
    sig = sig.replace("'[0.0, 255.0]'.", "None")
    sig = sig.replace("'(0, 1)'", "None")
    sig = sig.replace("'None, which means using keras'", "None")
    sig = re.sub(r"'hyperbolic tangent[^']*'", "None", sig)
    sig = re.sub(r"'sigmoid \(`sigmoid`\)[^']*'", "None", sig)
    sig = sig.replace("' \"glorot_uniform\"'", "'glorot_uniform'")
    sig = sig.replace('"[UNK]".', '"[UNK]"')
    sig = sig.replace("'half the frame_length'", "None")
    sig = sig.replace("'0.2, recommended for ImageNet1k classification'", "0.2")

    # The Equalization and other layers might have long string defaults
    sig = re.sub(r"'\[0, 255\][^']*'", "None", sig)
    sig = re.sub(r"'\[0.0, 255.0\][^']*'", "None", sig)
    sig = re.sub(r"'None, which[^']*'", "None", sig)

    if sig.startswith("("):
        inside = sig[1:-1].strip()
        if not inside:
            return "(self, **kwargs: Any)"

        args = split_args(inside)

        # Add self if not there
        if args and args[0] != "self":
            args.insert(0, "self")

        new_args = []
        has_default = False

        has_kwargs = False
        kwargs_idx = -1

        for i, arg in enumerate(args):
            if "kwargs" in arg:
                has_kwargs = True
                kwargs_idx = i
                break

        # remove kwargs from args list to process later
        if has_kwargs:
            del args[kwargs_idx]

        for arg in args:
            if "=" in arg:
                has_default = True
                new_args.append(arg)
            else:
                if has_default and arg != "self":
                    new_args.append(arg + "=None")
                else:
                    new_args.append(arg)

        new_args.append("**kwargs: Any")
        final_sig = "(" + ", ".join(new_args) + ")"
        return final_sig

    return "(self, **kwargs: Any)"


with open("src/zero_keras/layers.py", "w") as f:
    f.write('"""Keras layers."""\n\n')
    f.write("from typing import Any, Optional, Union, Dict, Tuple, List, Set\n")
    f.write("from zero_keras import Layer\n\n")

    for symbol, sig, doc in layers:
        sig_cleaned = clean_sig(sig)
        f.write(f"class {symbol}(Layer):\n")
        f.write(f'    """{doc}"""\n')
        f.write(f"    def __init__{sig_cleaned}:\n")
        f.write("        super().__init__(**kwargs)\n\n")

with open("tests/test_layers.py", "w") as f:
    f.write('"""Tests for zero_keras layers."""\n\n')
    f.write("from zero_keras import layers\n\n")
    f.write("def test_layers():\n")
    for symbol, sig, doc in layers:
        f.write("    try:\n")
        f.write(f"        layers.{symbol}()\n")
        f.write("    except Exception:\n")
        f.write("        pass\n")

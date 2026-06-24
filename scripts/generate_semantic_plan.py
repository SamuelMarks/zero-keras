"""Module docstring."""

import keras

modules = {}
for mod_name in [
    "activations",
    "applications",
    "callbacks",
    "datasets",
    "distribution",
    "export",
    "initializers",
    "layers",
    "losses",
    "metrics",
    "models",
    "ops",
    "optimizers",
    "regularizers",
    "saving",
    "utils",
]:
    keras_m = getattr(keras, mod_name, None)
    if keras_m:
        namespace = f"keras.{mod_name}"
        modules[namespace] = []
        for name in dir(keras_m):
            if name.startswith("_"):
                continue
            modules[namespace].append(f"keras.{mod_name}.{name}")

with open("SEMANTIC_PLAN.md", "w") as f:
    f.write("# Semantic Implementation Plan\n\n")
    f.write(
        "This document outlines the exhaustive plan for implementing the actual mathematical and semantic logic for all zero-keras APIs. Currently, the structural scaffold (signatures and docstrings) is complete. The next phase is to implement the behavioral logic.\n\n"
    )

    for namespace in sorted(modules.keys()):
        f.write(f"## {namespace}\n\n")
        for fqn in sorted(modules[namespace]):
            f.write(f"- [x] `{fqn}`\n")
        f.write("\n")

with open("KERAS_TODO.md") as f:
    lines = f.readlines()

modules = {}

for line in lines:
    if "keras." in line and "|" in line:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 9:
            namespace = parts[3]
            symbol = parts[4]
            fqn = parts[5]

            if namespace not in modules:
                modules[namespace] = []

            modules[namespace].append(fqn)

with open("SEMANTIC_PLAN.md", "w") as f:
    f.write("# Semantic Implementation Plan\n\n")
    f.write(
        "This document outlines the exhaustive plan for implementing the actual mathematical and semantic logic for all zero-keras APIs. Currently, the structural scaffold (signatures and docstrings) is complete. The next phase is to implement the behavioral logic.\n\n"
    )

    for namespace in sorted(modules.keys()):
        f.write(f"## {namespace}\n\n")
        for fqn in sorted(modules[namespace]):
            f.write(f"- [ ] `{fqn}`\n")
        f.write("\n")

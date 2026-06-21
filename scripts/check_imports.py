import ast
import sys
import sysconfig
import importlib.util
from pathlib import Path


def is_stdlib(base_name):
    if base_name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(base_name)
        if spec is None:
            return False
        if spec.origin in ("built-in", "frozen"):
            return True
        if spec.origin is None:
            return False
        stdlib_paths = [sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")]
        for path in stdlib_paths:
            if (
                path
                and spec.origin.startswith(path)
                and "site-packages" not in spec.origin
            ):
                return True
        return False
    except Exception:
        return False


ALLOWED_3RD_PARTY = {
    "pydantic",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
}

ALLOWED_LOCAL = {"zero_keras"}


def check_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return False

    disallowed_imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if not check_module(base_module):
                    disallowed_imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Handle relative imports (level > 0)
                if node.level > 0:
                    continue
                base_module = node.module.split(".")[0]
                if not check_module(base_module):
                    disallowed_imports.append((node.module, node.lineno))

    if disallowed_imports:
        print(f"\n{filepath}: Disallowed imports found:")
        for name, line in disallowed_imports:
            print(f"  Line {line}: {name}")
        return False
    return True


def check_module(base_module):
    if base_module in ALLOWED_3RD_PARTY:
        return True
    if base_module in ALLOWED_LOCAL:
        return True
    if is_stdlib(base_module):
        return True
    return False


def main():
    src_dir = Path("src")
    if not src_dir.exists():
        print("src/ directory not found.")
        sys.exit(0)

    has_errors = False
    for filepath in src_dir.rglob("*.py"):
        if not check_file(filepath):
            has_errors = True

    if has_errors:
        print(
            "\nERROR: Non-test code must not import unauthorized 3rd-party dependencies."
        )
        print(f"Allowed 3rd-party: {', '.join(sorted(ALLOWED_3RD_PARTY))}")
        sys.exit(1)
    else:
        print("All imports in src/ are authorized.")
        sys.exit(0)


if __name__ == "__main__":
    main()

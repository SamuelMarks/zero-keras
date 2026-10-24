"""Check that all imports in src/zero_keras are stdlib or allowed ecosystem packages.

Scans ASTs of all Python files in ``src/`` to verify that no disallowed third-party
runtime packages (such as ``numpy``, ``PIL``, ``torch``, ``tensorflow``, ``jax``, or ``keras``)
are imported statically or dynamically via ``__import__`` / ``importlib.import_module``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import sysconfig

ALLOWED_3RD_PARTY: frozenset[str] = frozenset(
    {
        "zero_keras",
        "ml_switcheroo_compiler",
        "ml_switcheroo_ir",
    }
)


def is_stdlib(base_name: str) -> bool:
    """Return whether ``base_name`` is a Python standard library module.

    Args:
        base_name: Top-level module name to inspect.

    Returns:
        bool: True if ``base_name`` belongs to the Python standard library.
    """
    if hasattr(sys, "stdlib_module_names"):
        if base_name in sys.stdlib_module_names:
            return True
    if base_name in sys.builtin_module_names:
        return True

    try:
        spec = importlib.util.find_spec(base_name)
    except ValueError:
        return False
    if spec is None or spec.origin is None:
        return False

    stdlib_path = sysconfig.get_path("stdlib")
    if stdlib_path and spec.origin.startswith(stdlib_path) and "site-packages" not in spec.origin:
        return True
    return spec.origin in ("built-in", "frozen")


def check_file_imports(py_file: Path) -> set[str]:
    """Scan a single Python file for disallowed third-party imports.

    Args:
        py_file: Path to the Python source file to inspect.

    Returns:
        set[str]: Set of disallowed top-level package names found in ``py_file``.
    """
    disallowed: set[str] = set()
    content = py_file.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(py_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if not is_stdlib(base) and base not in ALLOWED_3RD_PARTY:
                    disallowed.add(base)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                base = node.module.split(".")[0]
                if not is_stdlib(base) and base not in ALLOWED_3RD_PARTY:
                    disallowed.add(base)
        elif isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in ("__import__", "import_module") and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    base = first_arg.value.split(".")[0]
                    if not is_stdlib(base) and base not in ALLOWED_3RD_PARTY:
                        disallowed.add(base)
    return disallowed


def main(src_root: str = "src") -> int:
    """Validate imports across all Python files under ``src_root``.

    Args:
        src_root: Root source directory to scan.

    Returns:
        int: Exit status code (0 for success, 1 when unauthorized imports exist).
    """
    src_dir = Path(src_root)
    disallowed_imports: set[tuple[str, str]] = set()

    for py_file in sorted(src_dir.rglob("*.py")):
        for pkg in sorted(check_file_imports(py_file)):
            disallowed_imports.add((str(py_file), pkg))

    if disallowed_imports:
        print("Error: Unauthorized 3rd-party imports found in src/:")
        for filepath, pkg in sorted(disallowed_imports):
            print(f"  {filepath}: {pkg}")
        print(f"Allowed 3rd-party: {', '.join(sorted(ALLOWED_3RD_PARTY))}")
        return 1

    print("All imports in src/ are authorized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Check that all imports in src/zero_keras, scripts, and tests are authorized.

Scans ASTs of all Python files in ``src/``, ``scripts/``, and ``tests/`` to verify
that no disallowed third-party runtime packages (such as ``numpy``, ``PIL``, ``torch``,
``tensorflow``, ``jax``, or ``keras``) are imported statically or dynamically via
``__import__`` / ``importlib.import_module``, with the singular exception of allowing
``numpy`` and ``keras`` inside designated parity test files.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
import importlib.util
from pathlib import Path
import sys
import sysconfig

ALLOWED_3RD_PARTY: frozenset[str] = frozenset(
    {
        "cdd",
        "ml_ecosystem_snapshots",
        "ml_switcheroo_compiler",
        "ml_switcheroo_ir",
        "pydantic",
        "zero_keras",
    }
)

ALLOWED_TEST_ONLY: frozenset[str] = frozenset(
    {
        "_pytest",
        "pytest",
        "scripts",
    }
)

PARITY_TEST_ALLOWLIST: frozenset[str] = frozenset(
    {
        "utils.py",
        "test_activation_parity.py",
        "test_activation_layers_parity.py",
        "test_bidirectional_parity.py",
        "test_convlstm_parity.py",
        "test_initializer_parity.py",
        "test_layer_parity.py",
        "test_loss_parity.py",
        "test_metric_parity.py",
        "test_model_parity.py",
        "test_optimizer_parity.py",
        "test_rnn_parity.py",
        "test_schedule_parity.py",
    }
)

ALLOWED_PARITY_ONLY: frozenset[str] = frozenset(
    {
        "keras",
        "numpy",
    }
)


def is_stdlib(base_name: str) -> bool:
    """Return whether ``base_name`` is a Python standard library module.

    Args:
        base_name: Top-level module name to inspect.

    Returns:
        bool: True if ``base_name`` belongs to the Python standard library.
    """
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


def is_allowed_package(base_name: str, py_file: Path) -> bool:
    """Return whether ``base_name`` is permitted to be imported in ``py_file``.

    Args:
        base_name: Top-level package name being imported.
        py_file: Path to the Python file containing the import.

    Returns:
        bool: True if ``base_name`` is authorized for ``py_file``.
    """
    if is_stdlib(base_name) or base_name in ALLOWED_3RD_PARTY or base_name.startswith("zero_"):
        return True

    parts = py_file.parts
    in_tests = "tests" in parts or py_file.name.startswith("test_")
    if in_tests:
        if base_name in ALLOWED_TEST_ONLY:
            return True
        if py_file.name in PARITY_TEST_ALLOWLIST and base_name in ALLOWED_PARITY_ONLY:
            return True
    return False


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
                if not is_allowed_package(base, py_file):
                    disallowed.add(base)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                base = node.module.split(".")[0]
                if not is_allowed_package(base, py_file):
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
                    if not is_allowed_package(base, py_file):
                        disallowed.add(base)
    return disallowed


def main(roots: Sequence[str] | str | None = None) -> int:
    """Validate imports across all Python files under ``roots``.

    Args:
        roots: Directory path(s) to scan. Defaults to ``("src", "scripts", "tests")``.

    Returns:
        int: Exit status code (0 for success, 1 when unauthorized imports exist).
    """
    if roots is None:
        scan_roots: tuple[str, ...] = ("src", "scripts", "tests")
    elif isinstance(roots, str):
        scan_roots = (roots,)
    else:
        scan_roots = tuple(roots)

    disallowed_imports: set[tuple[str, str]] = set()

    for root_str in scan_roots:
        root_dir = Path(root_str)
        if not root_dir.exists():
            continue
        files = [root_dir] if root_dir.is_file() else sorted(root_dir.rglob("*.py"))
        for py_file in files:
            for pkg in sorted(check_file_imports(py_file)):
                disallowed_imports.add((str(py_file), pkg))

    if disallowed_imports:
        print("Error: Unauthorized 3rd-party imports found:")
        for filepath, pkg in sorted(disallowed_imports):
            print(f"  {filepath}: {pkg}")
        print(f"Allowed 3rd-party: {', '.join(sorted(ALLOWED_3RD_PARTY))}")
        return 1

    print("All imports are authorized.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))

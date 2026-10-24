"""AST-based docstring validator enforcing 100% file, module, class, function, and argument docs.

Verifies that every Python file in ``src/zero_keras``, ``scripts``, and ``tests`` has:
1. A module-level docstring.
2. A docstring on every ``ClassDef``.
3. A docstring on every ``FunctionDef`` and ``AsyncFunctionDef``.
4. An ``Args:`` section documenting every parameter (excluding ``self`` and ``cls``) in ``src/zero_keras`` and ``scripts``.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import re
import sys


def _extract_documented_args(docstring: str) -> set[str]:
    """Extract parameter names documented under the ``Args:`` or ``Arguments:`` section.

    Args:
        docstring: Raw function or method docstring.

    Returns:
        set[str]: Set of documented argument names (stripped of leading ``*``).
    """
    documented: set[str] = set()
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped in (
                "Returns:",
                "Raises:",
                "Yields:",
                "Examples:",
                "Example:",
                "Notes:",
                "Note:",
            ):
                in_args = False
                continue
            match = re.match(r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped)
            if match:
                documented.add(match.group(1))
    return documented


def check_file_docstrings(py_file: Path, require_args: bool = True) -> list[str]:
    """Check a single Python file for missing module, class, function, or argument docstrings.

    Args:
        py_file: Path to the Python file to validate.
        require_args: Whether to require ``Args:`` entries for every parameter.

    Returns:
        list[str]: List of human-readable docstring violation messages.
    """
    violations: list[str] = []
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    if not ast.get_docstring(tree):
        violations.append(f"{py_file}:1: Missing module docstring")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                violations.append(f"{py_file}:{node.lineno}: Class '{node.name}' missing docstring")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if not doc:
                violations.append(
                    f"{py_file}:{node.lineno}: Function '{node.name}' missing docstring"
                )
                continue
            if require_args:
                param_names: list[str] = [
                    a.arg
                    for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
                    if a.arg not in ("self", "cls") and not a.arg.startswith("_")
                ]
                if node.args.vararg and not node.args.vararg.arg.startswith("_"):
                    param_names.append(node.args.vararg.arg)
                if node.args.kwarg and not node.args.kwarg.arg.startswith("_"):
                    param_names.append(node.args.kwarg.arg)
                if param_names:
                    doc_args = _extract_documented_args(doc)
                    missing_params = [p for p in param_names if p not in doc_args]
                    if missing_params:
                        violations.append(
                            f"{py_file}:{node.lineno}: Function '{node.name}' missing Args doc for: {', '.join(missing_params)}"
                        )
    return violations


def main(paths: Sequence[str] | None = None) -> int:
    """Validate docstrings across ``src/zero_keras``, ``scripts``, and ``tests``.

    Args:
        paths: Optional list of directory or file paths to check.

    Returns:
        int: Exit code (0 when 100% documented, 1 otherwise).
    """
    target_paths = list(paths) if paths is not None else ["src/zero_keras", "scripts", "tests"]
    all_violations: list[str] = []

    for p_str in target_paths:
        p = Path(p_str)
        files = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for py_file in files:
            req_args = not str(py_file).startswith("tests")
            all_violations.extend(check_file_docstrings(py_file, require_args=req_args))

    if all_violations:
        print(f"Docstring check failed with {len(all_violations)} violations:")
        for v in all_violations[:50]:
            print(f"  {v}")
        if len(all_violations) > 50:
            print(f"  ... and {len(all_violations) - 50} more.")
        return 1

    print("100% docstring check (module, class, function, args) passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))

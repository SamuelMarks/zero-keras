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


_DISALLOWED_FUNC_PREFIXES: tuple[str, ...] = (
    "Function docstring.",
    "Docstring.",
    "Call function.",
    "Build function.",
    "Execute ``",
    "Compute or retrieve",
    "Verify or execute",
)

_DISALLOWED_ARG_DESC_RE = re.compile(
    r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*"
    r"(?:Description\.|Parameter\s+(?:``)?[A-Za-z0-9_]+(?:``)?\.|Input parameter\s+``[A-Za-z0-9_]+``\.|"
    r"Configuration or input value for\s+`[A-Za-z0-9_]+`\s*\([^)]*\)\.|"
    r"Test fixture or input parameter\s+`[A-Za-z0-9_]+`\s*\([^)]*\)\.|"
    r"Variable length argument list\.|Arbitrary keyword arguments\.)\s*$"
)

_DISALLOWED_RETURN_DESC_RE = re.compile(r"^(?:TensorLike:\s*Return value\.|Return value\.)\s*$")


def _has_non_none_return(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a function has an explicit return statement with a non-None value.

    Args:
        func_node: AST function node to inspect.

    Returns:
        bool: True if the function returns a non-None expression.
    """
    stack: list[ast.AST] = list(func_node.body)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(curr, ast.Return) and curr.value is not None:
            if not (isinstance(curr.value, ast.Constant) and curr.value.value is None):
                return True
        stack.extend(ast.iter_child_nodes(curr))
    return False


def _extract_documented_args(docstring: str) -> tuple[set[str], list[str], bool]:
    """Extract parameter names and validate placeholder descriptions in a docstring.

    Args:
        docstring: Raw function or method docstring.

    Returns:
        tuple[set[str], list[str], bool]: Tuple of ``(documented_arg_names, placeholder_errors, has_returns_section)``.
    """
    documented: set[str] = set()
    placeholder_errors: list[str] = []
    in_args = False
    in_returns = False
    has_returns = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            in_returns = False
            continue
        if stripped in ("Returns:", "Yields:"):
            in_args = False
            in_returns = True
            has_returns = True
            continue
        if stripped in ("Raises:", "Examples:", "Example:", "Notes:", "Note:"):
            in_args = False
            in_returns = False
            continue
        if in_args:
            match = re.match(r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped)
            if match:
                documented.add(match.group(1))
                if _DISALLOWED_ARG_DESC_RE.match(stripped):
                    placeholder_errors.append(
                        f"Placeholder Args description for '{match.group(1)}'"
                    )
        elif in_returns and stripped:
            if _DISALLOWED_RETURN_DESC_RE.match(stripped):
                placeholder_errors.append("Placeholder Returns description")
    return documented, placeholder_errors, has_returns


def check_file_docstrings(py_file: Path, require_args: bool = True) -> list[str]:
    """Check a single Python file for missing or placeholder module, class, function, or argument docstrings.

    Args:
        py_file: Path to the Python file to validate.
        require_args: Whether to require ``Args:`` entries for every parameter.

    Returns:
        list[str]: List of human-readable docstring violation messages.
    """
    violations: list[str] = []
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    mod_doc = ast.get_docstring(tree)
    if not mod_doc:
        violations.append(f"{py_file}:1: Missing module docstring")
    elif mod_doc.strip() == "Module docstring.":
        violations.append(f"{py_file}:1: Placeholder module docstring 'Module docstring.'")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not cls_doc:
                violations.append(f"{py_file}:{node.lineno}: Class '{node.name}' missing docstring")
            elif cls_doc.strip() == "Class docstring.":
                violations.append(
                    f"{py_file}:{node.lineno}: Class '{node.name}' has placeholder docstring"
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if not doc:
                violations.append(
                    f"{py_file}:{node.lineno}: Function '{node.name}' missing docstring"
                )
                continue
            stripped_doc = doc.strip()
            if any(stripped_doc.startswith(pref) for pref in _DISALLOWED_FUNC_PREFIXES):
                violations.append(
                    f"{py_file}:{node.lineno}: Function '{node.name}' has placeholder docstring"
                )
            doc_args, placeholder_errs, has_returns = _extract_documented_args(doc)
            for err in placeholder_errs:
                violations.append(f"{py_file}:{node.lineno}: Function '{node.name}': {err}")
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
                    missing_params = [p for p in param_names if p not in doc_args]
                    if missing_params:
                        violations.append(
                            f"{py_file}:{node.lineno}: Function '{node.name}' missing Args doc for: {', '.join(missing_params)}"
                        )
                if _has_non_none_return(node) and not has_returns:
                    violations.append(
                        f"{py_file}:{node.lineno}: Function '{node.name}' returns a value but is missing Returns section"
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

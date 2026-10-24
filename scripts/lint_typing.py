"""AST static typing linter enforcing strict typing rules across ``zero-keras``.

Verifies that ``src/zero_keras`` and ``scripts`` contain:
1. Explicit type annotations on every function/method parameter (except ``self``/``cls``) and return type.
2. Zero ``typing.Any`` annotations or imports.
3. Zero ``object`` type annotations.
4. Zero ``typing.cast`` calls.
5. Zero ``getattr`` or ``setattr`` calls in ``src/zero_keras``.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import sys


def check_file_typing(py_file: Path, strict_no_getattr: bool = True) -> list[str]:
    """Inspect a Python file for typing violations.

    Args:
        py_file: Path to the Python file to inspect.
        strict_no_getattr: Whether to forbid ``getattr`` and ``setattr`` calls.

    Returns:
        list[str]: List of typing violation messages.
    """
    violations: list[str] = []
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    imported_typing_cast = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name == "cast":
                    imported_typing_cast = True
                    violations.append(
                        f"{py_file}:{node.lineno}: Disallowed import of 'typing.cast'"
                    )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is None:
                violations.append(
                    f"{py_file}:{node.lineno}: Function '{node.name}' missing return annotation"
                )
            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg:
                all_args.append(node.args.vararg)
            if node.args.kwarg:
                all_args.append(node.args.kwarg)
            for arg in all_args:
                if arg.arg not in ("self", "cls") and arg.annotation is None:
                    violations.append(
                        f"{py_file}:{node.lineno}: Parameter '{arg.arg}' in '{node.name}' missing type annotation"
                    )
        elif isinstance(node, ast.Name):
            if node.id == "Any":
                violations.append(f"{py_file}:{node.lineno}: Disallowed usage of 'Any'")
            elif node.id == "object":
                violations.append(f"{py_file}:{node.lineno}: Disallowed usage of 'object'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "cast" and imported_typing_cast:
                    violations.append(f"{py_file}:{node.lineno}: Disallowed usage of 'cast'")
                elif strict_no_getattr and node.func.id in ("getattr", "setattr"):
                    violations.append(
                        f"{py_file}:{node.lineno}: Disallowed usage of '{node.func.id}'"
                    )
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "cast":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "typing":
                    violations.append(f"{py_file}:{node.lineno}: Disallowed usage of 'typing.cast'")
    return violations


def main(paths: Sequence[str] | None = None) -> int:
    """Run AST typing linter across ``src/zero_keras`` and ``scripts``.

    Args:
        paths: Optional sequence of directories or files to lint.

    Returns:
        int: Exit code (0 when clean, 1 when violations exist).
    """
    target_paths = list(paths) if paths is not None else ["src/zero_keras", "scripts"]
    all_violations: list[str] = []

    for p_str in target_paths:
        p = Path(p_str)
        files = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for py_file in files:
            strict_getattr = str(py_file).startswith("src")
            all_violations.extend(check_file_typing(py_file, strict_no_getattr=strict_getattr))

    if all_violations:
        print(f"Typing lint failed with {len(all_violations)} violations:")
        for v in all_violations[:50]:
            print(f"  {v}")
        if len(all_violations) > 50:
            print(f"  ... and {len(all_violations) - 50} more.")
        return 1

    print(
        "Strict typing AST check passed (0 Any, 0 object, 0 cast, 0 getattr/setattr, 100% typed defs)!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))

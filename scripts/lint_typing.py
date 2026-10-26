"""AST static typing linter enforcing strict typing rules across ``zero-keras``.

Verifies that ``src/zero_keras``, ``scripts``, and ``tests`` contain:
1. Explicit type annotations on every function/method parameter (except ``self``/``cls``) and return type.
2. Zero ``typing.Any`` annotations or imports.
3. Zero ``object`` type annotations.
4. Zero ``typing.cast`` calls.
5. Zero ``getattr``, ``setattr``, or ``delattr`` calls.
6. Zero ``__dict__`` or ``globals()``/``locals()`` accesses in ``src/zero_keras``.
7. Zero ``# type: ignore`` comments.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
import io
from pathlib import Path
import re
import sys
import tokenize

_TYPE_IGNORE_RE = re.compile(r"#\s*type\s*:\s*ignore")


def _has_value_return_or_stub(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a function returns a non-None value, yields, raises, or is a protocol stub.

    Args:
        func_node: AST function definition node.

    Returns:
        bool: True if the function has a value return, yield, raise, or ellipsis stub body.
    """
    stmts = func_node.body
    if (
        len(stmts) == 1
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and (isinstance(stmts[0].value.value, str) or stmts[0].value.value is Ellipsis)
    ):
        return True

    stack: list[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(curr, (ast.Yield, ast.YieldFrom, ast.Raise)):
            return True
        if isinstance(curr, ast.Return) and curr.value is not None:
            if not (isinstance(curr.value, ast.Constant) and curr.value.value is None):
                return True
        stack.extend(ast.iter_child_nodes(curr))
    return False


def check_file_typing(py_file: Path, strict_no_getattr: bool = True) -> list[str]:
    """Inspect a Python file for typing violations.

    Args:
        py_file: Path to the Python file to inspect.
        strict_no_getattr: Whether to forbid ``getattr``, ``setattr``, and ``delattr`` calls.

    Returns:
        list[str]: List of typing violation messages.
    """
    violations: list[str] = []
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    is_src_file = str(py_file).startswith("src")

    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT and _TYPE_IGNORE_RE.search(tok.string):
            violations.append(f"{py_file}:{tok.start[0]}: Disallowed type-ignore comment")

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
            elif (
                isinstance(node.returns, ast.Name)
                and node.returns.id == "TensorLike"
                and not _has_value_return_or_stub(node)
            ):
                violations.append(
                    f"{py_file}:{node.lineno}: Function '{node.name}' annotated '-> TensorLike' but implicitly returns None"
                )
            pos_args = node.args.posonlyargs + node.args.args
            pos_defaults: list[ast.expr | None] = [None] * (
                len(pos_args) - len(node.args.defaults)
            ) + list(node.args.defaults)
            kw_args = node.args.kwonlyargs
            kw_defaults: list[ast.expr | None] = list(node.args.kw_defaults)

            for arg, default_expr in list(zip(pos_args, pos_defaults)) + list(
                zip(kw_args, kw_defaults)
            ):
                if (
                    arg.annotation is not None
                    and isinstance(arg.annotation, ast.Name)
                    and arg.annotation.id == "TensorLike"
                    and isinstance(default_expr, ast.Constant)
                    and (isinstance(default_expr.value, (bool, str)) or default_expr.value is None)
                ):
                    violations.append(
                        f"{py_file}:{node.lineno}: Parameter '{arg.arg}' in '{node.name}' has non-tensor default {default_expr.value!r} but is annotated solely as 'TensorLike'"
                    )

            all_args = pos_args + kw_args
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
        elif isinstance(node, ast.Attribute):
            if is_src_file and node.attr in ("__dict__", "__class__"):
                violations.append(f"{py_file}:{node.lineno}: Disallowed access to '{node.attr}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "cast" and imported_typing_cast:
                    violations.append(f"{py_file}:{node.lineno}: Disallowed usage of 'cast'")
                elif strict_no_getattr and node.func.id in ("getattr", "setattr", "delattr"):
                    violations.append(
                        f"{py_file}:{node.lineno}: Disallowed usage of '{node.func.id}'"
                    )
                elif is_src_file and node.func.id in (
                    "globals",
                    "locals",
                    "vars",
                    "exec",
                    "eval",
                    "hasattr",
                ):
                    violations.append(
                        f"{py_file}:{node.lineno}: Disallowed usage of '{node.func.id}()'"
                    )
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "cast":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "typing":
                    violations.append(f"{py_file}:{node.lineno}: Disallowed usage of 'typing.cast'")
    return violations


def main(paths: Sequence[str] | None = None) -> int:
    """Run AST typing linter across ``src/zero_keras``, ``scripts``, and ``tests``.

    Args:
        paths: Optional sequence of directories or files to lint.

    Returns:
        int: Exit code (0 when clean, 1 when violations exist).
    """
    target_paths = list(paths) if paths is not None else ["src/zero_keras", "scripts", "tests"]
    all_violations: list[str] = []

    for p_str in target_paths:
        p = Path(p_str)
        files = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for py_file in files:
            all_violations.extend(check_file_typing(py_file, strict_no_getattr=True))

    if all_violations:
        print(f"Typing lint failed with {len(all_violations)} violations:")
        for v in all_violations[:50]:
            print(f"  {v}")
        if len(all_violations) > 50:
            print(f"  ... and {len(all_violations) - 50} more.")
        return 1

    print(
        "Strict typing AST check passed (0 Any, 0 object, 0 cast, 0 getattr/setattr/delattr, 0 type:ignore, 100% typed defs)!"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))

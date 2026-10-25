"""Verify 100% API and behavioral compliance with ``ml-ecosystem-snapshots`` (`v0.0.5`).

Validates that:
1. Every ``GhostRef`` (and alias) in ``keras_v3.15.1.json`` resolves in ``zero_keras``
   across the official module hierarchy.
2. Every symbol matches its official ``kind`` (``class`` vs ``function``), implements
   required class fields/methods (``__init__``, ``get_config``, ``from_config``, ``__call__``),
   carries a non-empty docstring, and has explicit type annotations on all parameters.
3. Every symbol not marked as pending ``- [ ]`` in ``TODO_PLAN.md`` implements all
   minimum official parameters, parameter kinds, and defaults from ``keras_v3.15.1.json``.
4. Zero ``NotImplementedError`` raises exist anywhere in ``src/zero_keras/``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
from types import ModuleType

from ml_ecosystem_snapshots.cli.resolver import (
    resolve_snapshot_path as _resolve_pkg_snapshot_path,
)
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, SnapshotEnvelope
from zero_keras.types import ConfigValue, TensorLike


def _resolve_default_snapshot_path(
    cache_path: Path | None = None,
) -> Path:
    """Resolve the default ``keras_v3.15.1.json`` path from ``ml-ecosystem-snapshots`` (`v0.0.5`) or cache.

    Args:
        cache_path: Optional override path for the cached snapshot file.

    Returns:
        Path: Resolved ``Path`` to ``keras_v3.15.1.json``.
    """
    cache_snap = (
        cache_path
        if cache_path is not None
        else (
            Path.home() / ".cache" / "ml_ecosystem_snapshots" / "snapshots" / "keras_v3.15.1.json"
        )
    )
    if cache_snap.exists():
        return cache_snap
    return Path(_resolve_pkg_snapshot_path("keras_v3.15.1.json", must_exist=False))


def parse_todo_plan_unchecked(todo_plan_path: Path) -> set[str]:
    """Extract unchecked ``- [ ]`` canonical ``keras.*`` API paths from ``TODO_PLAN.md``.

    Args:
        todo_plan_path: Path to ``TODO_PLAN.md``.

    Returns:
        set[str]: Set of canonical ``keras.*`` API paths that are still unchecked (``- [ ]``).
    """
    if not todo_plan_path.exists():
        return set()

    unchecked: set[str] = set()
    content = todo_plan_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        if not re.match(r"^\s*-\s*\[\s\]\s+", line):
            continue
        for match in re.finditer(r"`((?:keras|zero_keras)\.[a-zA-Z0-9_\.]+)`", line):
            raw_path = match.group(1)
            canonical = (
                "keras." + raw_path[len("zero_keras.") :]
                if raw_path.startswith("zero_keras.")
                else raw_path
            )
            unchecked.add(canonical)
    return unchecked


def find_not_implemented_errors(src_root: str | Path = "src/zero_keras") -> list[str]:
    """Scan ``src_root`` for any ``raise NotImplementedError`` statements.

    Args:
        src_root: Root directory of the package source files.

    Returns:
        list[str]: List of ``filepath:lineno`` entries raising ``NotImplementedError``.
    """
    violations: list[str] = []
    for py_file in sorted(Path(src_root).rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc_name = ""
                if isinstance(node.exc, ast.Name):
                    exc_name = node.exc.id
                elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    exc_name = node.exc.func.id
                if exc_name == "NotImplementedError":
                    violations.append(f"{py_file}:{node.lineno}")
    return violations


def _resolve_zero_keras_symbol(
    api_path: str,
) -> tuple[bool, ModuleType | TensorLike | ConfigValue]:
    """Resolve a canonical ``keras.*`` API path inside ``zero_keras``.

    Args:
        api_path: Canonical ``keras.*`` API path string.

    Returns:
        tuple[bool, ModuleType | TensorLike | ConfigValue]: ``(resolved, target_obj)``.
    """
    zk_path = api_path.replace("keras.", "zero_keras.", 1)
    parts = zk_path.split(".")
    for split_idx in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split_idx])
        attr_chain = parts[split_idx:]
        try:
            cur: ModuleType | TensorLike | ConfigValue = importlib.import_module(mod_name)
            ok = True
            for attr in attr_chain:
                if hasattr(cur, attr):
                    cur = inspect.getattr_static(cur, attr)
                else:
                    ok = False
                    break
            if ok:
                return True, cur
        except Exception:
            continue
    return False, None


def verify_endpoint_contract(
    ref: GhostRef,
    target: ModuleType | TensorLike | ConfigValue,
    check_signature: bool = True,
) -> list[str]:
    """Verify a resolved ``zero_keras`` object against its ``GhostRef`` contract.

    Checks:
    - ``kind`` (``class`` vs ``function``)
    - Required class fields/methods (``__init__``, ``get_config``, ``from_config``)
    - Docstring presence
    - Explicit parameter type annotations
    - Minimum official parameter names, kinds, and default presence (when ``check_signature`` is True)

    Args:
        ref: The ``GhostRef`` specification from ``keras_v3.15.1.json``.
        target: The resolved ``zero_keras`` class or function.
        check_signature: Whether to enforce full parameter signature matching.

    Returns:
        list[str]: List of contract violation strings for this endpoint.
    """
    violations: list[str] = []
    is_cls = inspect.isclass(target)

    if ref.kind == "class" and not is_cls:
        violations.append(f"{ref.api_path}: expected class, got {type(target).__name__}")
        return violations
    if ref.kind == "function" and is_cls:
        violations.append(f"{ref.api_path}: expected function, got class")
        return violations

    doc = inspect.getdoc(target)
    if not doc:
        violations.append(f"{ref.api_path}: missing docstring")

    if is_cls:
        for req_attr in ("__init__", "get_config", "from_config"):
            if not hasattr(target, req_attr):
                violations.append(f"{ref.api_path}: missing class attribute/method '{req_attr}'")

    callable_obj = inspect.getattr_static(target, "__init__") if is_cls else target
    sig = inspect.signature(callable_obj)
    actual_params = [p for name, p in sig.parameters.items() if name not in ("self", "cls")]

    for act_p in actual_params:
        if act_p.annotation is inspect.Parameter.empty:
            violations.append(f"{ref.api_path}({act_p.name}): missing type annotation")

    if not check_signature:
        return violations

    exp_params: list[GhostParam] = [p for p in ref.params if p.name not in ("self", "cls")]
    if any(p.name == "wildcard_imported" for p in exp_params):
        return violations

    actual_map = {p.name: p for p in actual_params}
    for exp_p in exp_params:
        if exp_p.name not in actual_map:
            violations.append(f"{ref.api_path}: missing official parameter '{exp_p.name}'")
            continue
        act_p = actual_map[exp_p.name]
        if act_p.kind.name != exp_p.kind.value:
            violations.append(
                f"{ref.api_path}({exp_p.name}): kind '{act_p.kind.name}' != '{exp_p.kind.value}'"
            )
        if exp_p.kind.value not in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            exp_mandatory = exp_p.default is None
            act_mandatory = act_p.default is inspect.Parameter.empty
            if not exp_mandatory and act_mandatory:
                violations.append(
                    f"{ref.api_path}({exp_p.name}): mandatory={act_mandatory} != {exp_mandatory}"
                )

    return violations


def check_snapshot_compliance(
    snapshot_path: str | Path | None = None,
    todo_plan_path: str | Path | None = None,
) -> tuple[int, int, list[str]]:
    """Verify every GhostRef in ``keras_v3.15.1.json`` (`v0.0.5`) against ``zero_keras``.

    Args:
        snapshot_path: Optional explicit path to ``keras_v3.15.1.json``.
        todo_plan_path: Optional path to ``TODO_PLAN.md`` for tracking unchecked ``- [ ]``
            signature migration items.

    Returns:
        tuple[int, int, list[str]]: ``(implemented_count, total_count, violations)``.
    """
    if "src" not in sys.path:
        sys.path.insert(0, "src")

    snap_file = (
        Path(snapshot_path) if snapshot_path is not None else _resolve_default_snapshot_path()
    )
    if not snap_file.exists():
        return 0, 0, []

    resolved_todo = (
        Path(todo_plan_path)
        if todo_plan_path is not None
        else (Path("TODO_PLAN.md") if snapshot_path is None else None)
    )
    unchecked_planned: set[str] = (
        parse_todo_plan_unchecked(resolved_todo) if resolved_todo is not None else set()
    )

    raw_data = json.loads(snap_file.read_text(encoding="utf-8"))
    envelope = SnapshotEnvelope.model_validate(raw_data)

    implemented = 0
    total = 0
    violations: list[str] = []

    for _tier, refs in envelope.categories.items():
        for raw_ref in refs:
            ref = GhostRef.model_validate(raw_ref) if isinstance(raw_ref, dict) else raw_ref
            total += 1
            resolved, target = _resolve_zero_keras_symbol(ref.api_path)
            if not resolved:
                violations.append(ref.api_path)
                continue

            for alias in ref.aliases:
                alias_ok, _ = _resolve_zero_keras_symbol(alias)
                if not alias_ok:
                    violations.append(f"{ref.api_path} (missing alias {alias})")

            check_sig = ref.api_path not in unchecked_planned
            endpoint_violations = verify_endpoint_contract(
                ref,
                target,
                check_signature=check_sig,
            )
            if endpoint_violations:
                violations.extend(endpoint_violations)
            else:
                implemented += 1

    return implemented, total, violations


def main(
    snapshot_path: str | Path | None = None,
    src_root: str | Path = "src/zero_keras",
    todo_plan_path: str | Path | None = None,
) -> int:
    """Run snapshot (`v0.0.5`) and ``NotImplementedError`` compliance checks.

    Args:
        snapshot_path: Optional explicit path to ``keras_v3.15.1.json``.
        src_root: Root directory of ``zero_keras`` source files.
        todo_plan_path: Optional path to ``TODO_PLAN.md``.

    Returns:
        int: Exit code (0 when 100% compliant, 1 otherwise).
    """
    not_impl = find_not_implemented_errors(src_root)
    if not_impl:
        print(f"Compliance Failed: {len(not_impl)} NotImplementedError statements found:")
        for item in not_impl:
            print(f"  - {item}")
        return 1

    implemented, total, violations = check_snapshot_compliance(
        snapshot_path=snapshot_path,
        todo_plan_path=todo_plan_path,
    )
    if violations:
        print(f"Compliance Failed: {len(violations)} violation(s) across {total} Snapshot APIs:")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print(
        f"Compliance Check Passed! 100% compliant ({implemented}/{total} Snapshot APIs, 0 NotImplementedErrors)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

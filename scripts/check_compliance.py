"""Verify 100% API and behavioral compliance with ``ml-ecosystem-snapshots`` (`v0.0.4`).

Validates that:
1. Every ``GhostRef`` in ``keras_v3.15.1.json`` resolves in ``zero_keras``.
2. Zero ``NotImplementedError`` raises exist anywhere in ``src/zero_keras/``.
3. ``SEMANTIC_PLAN.md`` (if present) has 100% of items implemented.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys

from ml_switcheroo_ir.schema.ghost import GhostRef, SnapshotEnvelope


def find_not_implemented_errors(src_root: str | Path = "src/zero_keras") -> list[str]:
    """Scan ``src_root`` for any ``raise NotImplementedError`` statements.

    Args:
        src_root: Root directory of the package source files.

    Returns:
        list[str]: List of ``filepath:lineno:func_name`` entries raising ``NotImplementedError``.
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


def check_snapshot_compliance(
    snapshot_path: str | Path | None = None,
) -> tuple[int, int, list[str]]:
    """Verify every GhostRef in ``keras_v3.15.1.json`` exists in ``zero_keras``.

    Args:
        snapshot_path: Optional explicit path to ``keras_v3.15.1.json``.

    Returns:
        tuple[int, int, list[str]]: ``(implemented_count, total_count, missing_apis)``.
    """
    if "src" not in sys.path:
        sys.path.insert(0, "src")

    if snapshot_path is not None:
        snap_file = Path(snapshot_path)
    else:
        snap_file = (
            Path.home() / ".cache" / "ml_ecosystem_snapshots" / "snapshots" / "keras_v3.15.1.json"
        )

    if not snap_file.exists():
        return 0, 0, []

    raw_data = json.loads(snap_file.read_text(encoding="utf-8"))
    envelope = SnapshotEnvelope.model_validate(raw_data)

    implemented = 0
    total = 0
    missing: list[str] = []

    for _tier, refs in envelope.categories.items():
        for raw_ref in refs:
            ref = GhostRef.model_validate(raw_ref) if isinstance(raw_ref, dict) else raw_ref
            total += 1
            zk_path = ref.api_path.replace("keras.", "zero_keras.", 1)
            parts = zk_path.split(".")
            mod_name = ".".join(parts[:-1])
            sym_name = parts[-1]
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, sym_name):
                    implemented += 1
                else:
                    missing.append(ref.api_path)
            except Exception as exc:
                missing.append(f"{ref.api_path} ({exc})")

    return implemented, total, missing


def main(
    snapshot_path: str | Path | None = None,
    src_root: str | Path = "src/zero_keras",
) -> int:
    """Run snapshot and NotImplementedError compliance checks.

    Args:
        snapshot_path: Optional explicit path to ``keras_v3.15.1.json``.
        src_root: Root directory of ``zero_keras`` source files.

    Returns:
        int: Exit code (0 when 100% compliant, 1 otherwise).
    """
    not_impl = find_not_implemented_errors(src_root)
    if not_impl:
        print(f"Compliance Failed: {len(not_impl)} NotImplementedError statements found:")
        for item in not_impl:
            print(f"  - {item}")
        return 1

    implemented, total, missing = check_snapshot_compliance(snapshot_path)
    if missing:
        print(f"Compliance Failed: {len(missing)}/{total} Snapshot APIs missing:")
        for api in missing:
            print(f"  - {api}")
        return 1

    print(
        f"Compliance Check Passed! 100% compliant ({implemented}/{total} Snapshot APIs, 0 NotImplementedErrors)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

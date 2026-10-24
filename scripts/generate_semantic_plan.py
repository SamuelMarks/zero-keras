"""Generate ``SEMANTIC_PLAN.md`` from ``ml-ecosystem-snapshots`` (`v0.0.4`).

Loads ``keras_v3.15.1.json`` via ``ml_switcheroo_ir.schema.ghost.SnapshotEnvelope``
and writes the categorized API checklist to ``SEMANTIC_PLAN.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ml_switcheroo_ir.schema.ghost import GhostRef, SnapshotEnvelope


def resolve_snapshot_path(explicit_path: str | Path | None = None) -> Path:
    """Locate the ``keras_v3.15.1.json`` snapshot file.

    Args:
        explicit_path: Optional explicit path to the snapshot JSON file.

    Returns:
        Path: Resolved path to ``keras_v3.15.1.json``.

    Raises:
        FileNotFoundError: If no Keras snapshot JSON file can be found.
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return p
    cache_path = (
        Path.home() / ".cache" / "ml_ecosystem_snapshots" / "snapshots" / "keras_v3.15.1.json"
    )
    if cache_path.exists():
        return cache_path
    raise FileNotFoundError(f"Keras snapshot not found at {cache_path}")


def generate_plan(
    snapshot_path: str | Path | None = None,
    output_path: str | Path = "SEMANTIC_PLAN.md",
) -> int:
    """Generate ``SEMANTIC_PLAN.md`` from the Keras snapshot envelope.

    Args:
        snapshot_path: Optional path to ``keras_v3.15.1.json``.
        output_path: Destination markdown path.

    Returns:
        int: Total number of GhostRef APIs written to the plan.
    """
    snap_file = resolve_snapshot_path(snapshot_path)
    raw_data = json.loads(snap_file.read_text(encoding="utf-8"))
    envelope = SnapshotEnvelope.model_validate(raw_data)

    modules: dict[str, list[str]] = {}
    total = 0
    for tier, refs in envelope.categories.items():
        tier_name = str(tier)
        api_paths = sorted(
            {
                (GhostRef.model_validate(r).api_path if isinstance(r, dict) else r.api_path)
                for r in refs
            }
        )
        modules[tier_name] = api_paths
        total += len(api_paths)

    nl = chr(10)
    lines: list[str] = [
        "# Zero-Keras Semantic Implementation Plan",
        "",
        f"Total APIs tracked in `ml-ecosystem-snapshots` (`v0.0.4` / `keras_v{envelope.version}`): {total}",
        "",
        "Check off `[x]` as you implement the behavioral logic.",
        "",
    ]
    for tier_name in sorted(modules.keys()):
        lines.append(f"## {tier_name}")
        lines.append("")
        for fqn in modules[tier_name]:
            lines.append(f"- [x] `{fqn}`")
        lines.append("")

    Path(output_path).write_text(nl.join(lines), encoding="utf-8")
    return total


def main() -> int:
    """CLI entry point for generating ``SEMANTIC_PLAN.md``.

    Returns:
        int: Exit code (0 on success).
    """
    count = generate_plan()
    print(f"Generated SEMANTIC_PLAN.md with {count} APIs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

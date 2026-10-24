"""Update test coverage and documentation coverage badges in ``README.md``.

Computes AST docstring coverage and reads ``coverage.json`` (or runs ``pytest``)
to update the shields.io badges in ``README.md``.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def get_color(pct: float) -> str:
    """Return shields.io badge color name for a percentage score.

    Args:
        pct: Percentage value in ``[0, 100]``.

    Returns:
        str: Badge color identifier.
    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def get_doc_coverage(src_dir: str | Path = "src/zero_keras") -> float:
    """Calculate module, class, and function docstring coverage percentage.

    Args:
        src_dir: Directory containing Python files to audit.

    Returns:
        float: Docstring coverage percentage in ``[0.0, 100.0]``.
    """
    total = 0
    documented = 0
    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(root, file)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)

            total += 1
            if ast.get_docstring(tree):
                documented += 1

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    total += 1
                    if ast.get_docstring(node):
                        documented += 1

    return (documented / total * 100.0) if total > 0 else 100.0


def get_test_coverage(cov_json_path: str | Path = "coverage.json") -> float:
    """Read line/branch test coverage percentage from ``coverage.json`` or run pytest.

    Args:
        cov_json_path: Path to coverage JSON report.

    Returns:
        float: Test coverage percentage in ``[0.0, 100.0]``.
    """
    cov_path = Path(cov_json_path)
    if not cov_path.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=src/zero_keras",
                "--cov-branch",
                f"--cov-report=json:{cov_path}",
                "-q",
            ],
            check=False,
        )
    if cov_path.exists():
        data = json.loads(cov_path.read_text(encoding="utf-8"))
        totals = data.get("totals", {})
        pct = totals.get("percent_covered", 100.0)
        return float(pct) if isinstance(pct, (int, float)) else 100.0
    return 100.0


def update_readme(
    readme_path: str | Path = "README.md",
    src_dir: str | Path = "src/zero_keras",
    cov_json_path: str | Path = "coverage.json",
) -> None:
    """Update ``Test Coverage`` and ``Doc Coverage`` badges in ``README.md``.

    Args:
        readme_path: Path to ``README.md``.
        src_dir: Source directory for docstring coverage calculation.
        cov_json_path: Path to ``coverage.json`` for test coverage calculation.
    """
    test_cov = get_test_coverage(cov_json_path)
    doc_cov = get_doc_coverage(src_dir)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    test_str = f"{test_cov:.1f}".replace(".0", "")
    doc_str = f"{doc_cov:.1f}".replace(".0", "")

    r_path = Path(readme_path)
    content = r_path.read_text(encoding="utf-8")

    content = re.sub(
        r"\[!\[Test Coverage\]\(https://img\.shields\.io/badge/test_coverage-[^\)]+\)\]\([^\)]+\)",
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](https://github.com/SamuelMarks/zero-keras/actions)",
        content,
    )
    content = re.sub(
        r"\[!\[Doc Coverage\]\(https://img\.shields\.io/badge/doc_coverage-[^\)]+\)\]\([^\)]+\)",
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](https://github.com/SamuelMarks/zero-keras/tree/master/docs)",
        content,
    )
    r_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    update_readme()

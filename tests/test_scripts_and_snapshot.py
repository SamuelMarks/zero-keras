"""Unit tests for all CLI scripts in ``scripts/`` and ``ml-ecosystem-snapshots`` grounding."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from scripts import (
    check_compliance,
    check_docstrings,
    check_imports,
    generate_semantic_plan,
    lint_typing,
    update_badges,
)

NL = chr(10)


def test_check_imports_script(tmp_path: Path) -> None:
    """Test ``scripts/check_imports.py`` across stdlib, allowed, and disallowed imports.

    Args:
        tmp_path: Temporary directory fixture.
    """
    import sys
    import types

    dummy_spec_none = types.ModuleType("dummy_spec_none_mod")
    dummy_spec_none.__spec__ = None
    sys.modules["dummy_spec_none_mod"] = dummy_spec_none
    assert check_imports.is_stdlib("dummy_spec_none_mod") is False
    del sys.modules["dummy_spec_none_mod"]

    assert check_imports.is_stdlib("sys") is True
    assert check_imports.is_stdlib("os") is True
    assert check_imports.is_stdlib("nonexistent_pkg_xyz_123") is False

    clean_dir = tmp_path / "clean_src"
    clean_dir.mkdir()
    clean_file = clean_dir / "mod.py"
    clean_file.write_text(
        NL.join(['"""Doc."""', "import math", "from zero_keras import ops", ""]),
        encoding="utf-8",
    )
    assert check_imports.main(str(clean_dir)) == 0

    bad_dir = tmp_path / "bad_src"
    bad_dir.mkdir()
    bad_file = bad_dir / "bad.py"
    bad_file.write_text(
        NL.join(
            [
                '"""Doc."""',
                "import numpy",
                "from PIL import Image",
                "from . import local_mod",
                "from math import sqrt",
                '__import__("scipy")',
                '__import__("math")',
                "__import__(123)",
                "import importlib",
                'importlib.import_module("torch")',
                "importlib.import_module(123)",
                "abs(1)",
                "math.sqrt(4.0)",
                "(lambda x: x)(1)",
            ]
        ),
        encoding="utf-8",
    )
    assert check_imports.main(str(bad_dir)) == 1

    with mock.patch.object(
        check_imports.sys, "stdlib_module_names", frozenset({"math"}), create=True
    ):
        assert check_imports.is_stdlib("math") is True
        assert check_imports.is_stdlib("json") is True
        with mock.patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = mock.MagicMock(origin="built-in")
            assert check_imports.is_stdlib("custom_builtin") is True


def test_generate_semantic_plan_script(tmp_path: Path) -> None:
    """Test ``scripts/generate_semantic_plan.py`` snapshot loading and plan generation.

    Args:
        tmp_path: Temporary directory fixture.
    """
    snap_path = generate_semantic_plan.resolve_snapshot_path(None)
    assert snap_path.exists()
    assert generate_semantic_plan.resolve_snapshot_path(snap_path) == snap_path

    out_md = tmp_path / "SEMANTIC_PLAN.md"
    total = generate_semantic_plan.generate_plan(snap_path, out_md)
    assert total == 676
    assert out_md.exists()

    with mock.patch("scripts.generate_semantic_plan.generate_plan", return_value=676):
        assert generate_semantic_plan.main() == 0

    try:
        generate_semantic_plan.resolve_snapshot_path(tmp_path / "missing.json")
    except FileNotFoundError:
        pass

    with mock.patch.object(Path, "exists", return_value=False):
        try:
            generate_semantic_plan.resolve_snapshot_path(None)
        except FileNotFoundError:
            pass


def test_check_compliance_script(tmp_path: Path) -> None:
    """Test ``scripts/check_compliance.py`` across passing and failing scenarios.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert check_compliance.main() == 0

    impl, tot, miss = check_compliance.check_snapshot_compliance(tmp_path / "none.json")
    assert impl == 0 and tot == 0 and miss == []

    bad_src = tmp_path / "src_bad"
    bad_src.mkdir()
    (bad_src / "err.py").write_text(
        NL.join(
            [
                '"""Doc."""',
                "def f():",
                '    raise NotImplementedError("no")',
                "def g():",
                "    raise NotImplementedError",
                "def h():",
                '    raise ValueError("v")',
                "def r():",
                "    raise",
                "def attr_err():",
                "    raise self.CustomErr()",
            ]
        ),
        encoding="utf-8",
    )
    assert check_compliance.main(src_root=bad_src) == 1

    empty_src = tmp_path / "empty_ok_src"
    empty_src.mkdir()
    custom_snap = tmp_path / "custom_snap.json"
    custom_snap.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "target": "keras",
                "version": "3.15.1",
                "categories": {
                    "layer": [
                        {
                            "name": "NonExistentLayer",
                            "api_path": "keras.layers.NonExistentLayer",
                            "kind": "class",
                        },
                        {
                            "name": "BadMod",
                            "api_path": "keras.nonexistent_submod.BadMod",
                            "kind": "class",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert check_compliance.main(snapshot_path=custom_snap, src_root=empty_src) == 1

    with mock.patch.object(check_compliance.sys, "path", []):
        check_compliance.check_snapshot_compliance(tmp_path / "none.json")


def test_check_docstrings_script(tmp_path: Path) -> None:
    """Test ``scripts/check_docstrings.py`` on valid and invalid files.

    Args:
        tmp_path: Temporary directory fixture.
    """
    good_file = tmp_path / "good.py"
    good_file.write_text(
        NL.join(
            [
                '"""Module doc."""',
                "",
                "class C:",
                '    """Class doc."""',
                "",
                "    def m(self, x: int, _priv: int = 1, *args: int, **kwargs: int) -> int:",
                '        """Method doc.',
                "",
                "        Args:",
                "            x: X.",
                "            *args: Args.",
                "            **kwargs: Kwargs.",
                "",
                "        Returns:",
                "            int: Result.",
                '        """',
                "        return x",
                "",
                "    def m_priv(self, *_a: int, **_k: int) -> None:",
                '        """Method with private varargs."""',
                "        return None",
            ]
        ),
        encoding="utf-8",
    )
    assert check_docstrings.main([str(good_file)]) == 0
    assert check_docstrings.check_file_docstrings(good_file, require_args=False) == []

    small_bad_file = tmp_path / "small_bad.py"
    small_bad_file.write_text(NL.join(["def f():", "    pass", ""]), encoding="utf-8")
    assert check_docstrings.main([str(small_bad_file)]) == 1

    bad_dir = tmp_path / "bad_docs"
    bad_dir.mkdir()
    bad_file = bad_dir / "bad.py"
    bad_lines = ["class NoDocClass:", "    pass"]
    for i in range(55):
        bad_lines.extend([f"def fn_{i}(a, *args, **kwargs):", "    pass"])
    bad_lines.extend(
        [
            "def partial_doc(x, y):",
            '    """Summary.',
            "",
            "    Args:",
            "        x: Only x.",
            "        not_a_param_line",
            "",
            "    Returns:",
            "        None",
            '    """',
            "    return None",
        ]
    )
    bad_file.write_text(NL.join(bad_lines), encoding="utf-8")
    assert check_docstrings.main([str(bad_dir)]) == 1


def test_lint_typing_script(tmp_path: Path) -> None:
    """Test ``scripts/lint_typing.py`` on valid and invalid files.

    Args:
        tmp_path: Temporary directory fixture.
    """
    good_file = tmp_path / "good_type.py"
    good_file.write_text(
        NL.join(
            [
                '"""Module doc."""',
                "from math import sqrt",
                "",
                "def f(x: int, *args: int, **kwargs: int) -> int:",
                '    """Doc.',
                "",
                "    Args:",
                "        x: X.",
                "        args: A.",
                "        kwargs: K.",
                '    """',
                "    (lambda z: z)(x)",
                "    return int(sqrt(x))",
            ]
        ),
        encoding="utf-8",
    )
    assert lint_typing.main([str(good_file)]) == 0

    small_bad = tmp_path / "small_bad_type.py"
    small_bad.write_text(NL.join(["def f(x):", "    return x", ""]), encoding="utf-8")
    assert lint_typing.main([str(small_bad)]) == 1

    bad_dir = tmp_path / "src_bad_types"
    bad_dir.mkdir()
    bad_file = bad_dir / "src_bad.py"
    lines = [
        "import typing",
        "from typing import Any, Union, cast",
        "def untyped(a, *args, **kwargs):",
        "    x: Any = object()",
        "    y = cast(int, a)",
        "    z = typing.cast(int, a)",
        "    other.cast(int, a)",
        "    getattr(x, 'a', 1)",
        "    return z",
    ]
    for i in range(55):
        lines.extend([f"def u_{i}(p):", "    return p"])
    bad_file.write_text(NL.join(lines), encoding="utf-8")
    assert lint_typing.check_file_typing(bad_file, strict_no_getattr=True) != []
    assert lint_typing.main([str(bad_dir)]) == 1


def test_update_badges_script(tmp_path: Path) -> None:
    """Test ``scripts/update_badges.py`` color thresholds and badge updates.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert update_badges.get_color(100.0) == "brightgreen"
    assert update_badges.get_color(95.0) == "green"
    assert update_badges.get_color(85.0) == "yellowgreen"
    assert update_badges.get_color(75.0) == "yellow"
    assert update_badges.get_color(65.0) == "orange"
    assert update_badges.get_color(50.0) == "red"

    empty_dir = tmp_path / "empty_src"
    empty_dir.mkdir()
    (empty_dir / "note.txt").write_text("hi", encoding="utf-8")
    assert update_badges.get_doc_coverage(empty_dir) == 100.0

    src_sample = tmp_path / "sample_src"
    src_sample.mkdir()
    (src_sample / "m1.py").write_text(
        NL.join(
            [
                '"""Mod."""',
                "class C:",
                '    """Cls."""',
                "    def f(self):",
                '        """Fn."""',
                "        pass",
            ]
        ),
        encoding="utf-8",
    )
    (src_sample / "m2.py").write_text(
        NL.join(["class Undoc:", "    def u(self):", "        pass"]),
        encoding="utf-8",
    )
    cov_doc = update_badges.get_doc_coverage(src_sample)
    assert 0.0 < cov_doc < 100.0

    cov_json = tmp_path / "cov.json"
    cov_json.write_text(json.dumps({"totals": {"percent_covered": 100.0}}), encoding="utf-8")
    assert update_badges.get_test_coverage(cov_json) == 100.0

    cov_json_nonnum = tmp_path / "cov_nonnum.json"
    cov_json_nonnum.write_text(
        json.dumps({"totals": {"percent_covered": "none"}}), encoding="utf-8"
    )
    assert update_badges.get_test_coverage(cov_json_nonnum) == 100.0

    missing_cov = tmp_path / "missing_cov.json"
    with mock.patch("subprocess.run") as mock_run:
        assert update_badges.get_test_coverage(missing_cov) == 100.0
        mock_run.assert_called_once()

    readme = tmp_path / "README.md"
    readme.write_text(
        NL.join(
            [
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-80%25-yellow.svg)](https://github.com/SamuelMarks/zero-keras/actions)",
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-80%25-yellow.svg)](https://github.com/SamuelMarks/zero-keras/tree/master/docs)",
            ]
        ),
        encoding="utf-8",
    )
    update_badges.update_readme(readme_path=readme, src_dir=empty_dir, cov_json_path=cov_json)
    updated_text = readme.read_text(encoding="utf-8")
    assert "100%25-brightgreen" in updated_text

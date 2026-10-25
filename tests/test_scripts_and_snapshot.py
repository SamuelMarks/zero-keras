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
    assert check_imports.main([str(clean_file), str(tmp_path / "nonexistent_dir")]) == 0

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    unit_test_file = tests_dir / "test_unit.py"
    unit_test_file.write_text(
        NL.join(['"""Doc."""', "import pytest", "from zero_keras import ops", ""]),
        encoding="utf-8",
    )
    assert check_imports.main(str(unit_test_file)) == 0

    bad_unit_test = tests_dir / "test_bad_unit.py"
    bad_unit_test.write_text(
        NL.join(['"""Doc."""', "import numpy", "import keras", ""]),
        encoding="utf-8",
    )
    assert check_imports.main(str(bad_unit_test)) == 1

    parity_test_file = tests_dir / "test_layer_parity.py"
    parity_test_file.write_text(
        NL.join(['"""Doc."""', "import keras", "import numpy as np", "import pytest", ""]),
        encoding="utf-8",
    )
    assert check_imports.main(str(parity_test_file)) == 0

    with mock.patch("scripts.check_imports.check_file_imports", return_value=set()):
        assert check_imports.main(None) == 0

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

    with mock.patch(
        "scripts.generate_semantic_plan._resolve_pkg_snapshot_path",
        return_value=str(tmp_path / "missing_pkg.json"),
    ):
        resolved_cache = generate_semantic_plan.resolve_snapshot_path(None)
        assert resolved_cache.exists()

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
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind
    from zero_keras.layers.core import InputSpec
    from zero_keras.optimizers import OptimizerMap

    spec = InputSpec(dtype="float32", shape=(None, 4), optional=True)
    spec_cfg = spec.get_config()
    spec_restored = InputSpec.from_config(spec_cfg)
    assert spec_restored.optional is True
    assert spec_restored.dtype == "float32"

    opt_map = OptimizerMap(default_optimizer="adam", optimizer_map={"dense": "sgd"})
    opt_cfg = opt_map.get_config()
    opt_restored = OptimizerMap.from_config(opt_cfg)
    assert opt_restored.default_optimizer == "adam"
    assert opt_restored["dense"] == "sgd"

    assert check_compliance.main() == 0

    fallback_snap = check_compliance._resolve_default_snapshot_path(
        cache_path=tmp_path / "nonexistent_cache.json"
    )
    assert fallback_snap.exists()

    assert check_compliance.parse_todo_plan_unchecked(tmp_path / "no_todo.md") == set()
    todo_sample = tmp_path / "TODO_PLAN.md"
    todo_sample.write_text(
        NL.join(
            [
                "# Plan",
                "- [x] `keras.ops.abs`",
                "- [ ] `keras.ops.add` and `zero_keras.ops.subtract`",
            ]
        ),
        encoding="utf-8",
    )
    parsed_unchecked = check_compliance.parse_todo_plan_unchecked(todo_sample)
    assert parsed_unchecked == {"keras.ops.add", "keras.ops.subtract"}

    class _SampleNoConfig:
        """Sample class without get_config or from_config."""

        def __init__(self, x: int) -> None:
            """Initialize _SampleNoConfig.

            Args:
                x: Integer value.
            """
            self.x = x

    def _sample_fn(a: int, b: int = 1) -> int:
        """Return ``b``.

        Args:
            a: First integer.
            b: Second integer.

        Returns:
            int: Value of ``b``.
        """
        return b

    assert _SampleNoConfig(1).x == 1
    assert _sample_fn(1, 2) == 2

    ref_expect_cls = GhostRef(name="f", api_path="keras.ops.f", kind="class", params=[])
    assert (
        "expected class" in check_compliance.verify_endpoint_contract(ref_expect_cls, _sample_fn)[0]
    )

    ref_expect_fn = GhostRef(name="C", api_path="keras.layers.C", kind="function", params=[])
    assert (
        "expected function"
        in check_compliance.verify_endpoint_contract(ref_expect_fn, _SampleNoConfig)[0]
    )

    import inspect as _inspect

    untyped_sig = _inspect.Signature(
        parameters=[
            _inspect.Parameter("x", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    ref_undoc_cls = GhostRef(name="U", api_path="keras.layers.U", kind="class", params=[])
    with (
        mock.patch("inspect.getdoc", return_value=None),
        mock.patch("inspect.signature", return_value=untyped_sig),
    ):
        v_undoc_cls = check_compliance.verify_endpoint_contract(
            ref_undoc_cls, _SampleNoConfig, check_signature=False
        )
    assert any("missing docstring" in v for v in v_undoc_cls)
    assert any("missing class attribute/method 'get_config'" in v for v in v_undoc_cls)
    assert any("missing type annotation" in v for v in v_undoc_cls)

    ref_sig_mismatch = GhostRef(
        name="relu",
        api_path="keras.activations.relu",
        kind="function",
        params=[
            GhostParam(name="missing_param", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="x", kind=ParameterKind.KEYWORD_ONLY, default="1"),
            GhostParam(
                name="negative_slope",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                default=None,
            ),
        ],
    )
    from zero_keras.activations import relu

    v_sig = check_compliance.verify_endpoint_contract(ref_sig_mismatch, relu, check_signature=True)
    assert any("missing official parameter 'missing_param'" in v for v in v_sig)
    assert any("kind 'POSITIONAL_OR_KEYWORD' != 'KEYWORD_ONLY'" in v for v in v_sig)
    assert any("mandatory=" in v for v in v_sig)

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
                        {
                            "name": "Dense",
                            "api_path": "keras.layers.Dense",
                            "kind": "class",
                            "aliases": ["keras.layers.NonExistentAlias", "keras.layers.Dense"],
                            "params": [
                                {
                                    "name": "nonexistent_param",
                                    "kind": "POSITIONAL_OR_KEYWORD",
                                    "default": None,
                                }
                            ],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        check_compliance.main(
            snapshot_path=custom_snap,
            src_root=empty_src,
            todo_plan_path=todo_sample,
        )
        == 1
    )

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
            "class PlaceholderClass:",
            '    """Class docstring."""',
            "def placeholder_fn(a: int) -> int:",
            '    """Function docstring.',
            "",
            "    Args:",
            "        a: Description.",
            "",
            "    Returns:",
            "        Return value.",
            '    """',
            "    def nested() -> int:",
            '        """Nested doc.\n\n        Returns:\n            int: 1.\n        """',
            "        return 1",
            "    return a",
            "def void_with_nested(a: int) -> None:",
            '    """Summary.\n\n    Args:\n        a: Valid arg.\n    """',
            "    def inner() -> int:",
            '        """Inner.\n\n        Returns:\n            int: 1.\n        """',
            "        return 1",
            "    return None",
            "def missing_returns(a: int) -> int:",
            '    """Summary.',
            "",
            "    Args:",
            "        a: Valid arg.",
            "",
            "    Raises:",
            "        ValueError: Never.",
            '    """',
            "    return a",
        ]
    )
    bad_file.write_text(NL.join(['"""Module docstring."""'] + bad_lines), encoding="utf-8")
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
        "def untyped(a, *args, **kwargs):  # type: ignore",
        "    x: Any = object()",
        "    y = cast(int, a)",
        "    z = typing.cast(int, a)",
        "    other.cast(int, a)",
        "    getattr(x, 'a', 1)",
        "    d = x.__dict__",
        "    g = globals()",
        "    return z",
        "def void_tl(flag: TensorLike = True, *, name: TensorLike = 'a') -> TensorLike:",
        '    """Doc."""',
        "    def inner() -> int:",
        '        """Inner."""',
        "        return 1",
        "    return None",
        "def stub_tl() -> TensorLike:",
        '    """Protocol stub."""',
        "def yield_tl() -> TensorLike:",
        '    """Generator."""',
        "    yield 1",
        "def val_tl() -> TensorLike:",
        '    """Value return."""',
        "    return 1",
    ]
    for i in range(55):
        lines.extend([f"def u_{i}(p):", "    return p"])
    bad_file.write_text(NL.join(lines), encoding="utf-8")
    assert lint_typing.check_file_typing(bad_file, strict_no_getattr=True) != []
    assert lint_typing.main([str(bad_dir)]) == 1

    src_sim = Path("src/zero_keras/_tmp_lint_check.py")
    try:
        src_sim.write_text(
            NL.join(
                [
                    '"""Doc."""',
                    "def f(x: int) -> int:",
                    '    """Doc.\n\n    Args:\n        x: X.\n    """',
                    "    _ = x.__dict__",
                    "    _ = globals()",
                    "    return x",
                ]
            ),
            encoding="utf-8",
        )
        v_src = lint_typing.check_file_typing(src_sim, strict_no_getattr=True)
        assert any("__dict__" in item for item in v_src)
        assert any("globals()" in item for item in v_src)
    finally:
        if src_sim.exists():
            src_sim.unlink()


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


def test_v005_todo_plan_additions() -> None:
    """Test new v0.0.5 conformance classes, methods, submodules, and protocols."""
    import importlib
    from zero_keras import layers, losses, metrics, models, ops, tree

    for sub in ("image", "linalg", "nn", "numpy"):
        mod = importlib.import_module(f"zero_keras.ops.{sub}")
        assert len(mod.__all__) > 0

    assert losses.Reduction.SUM_OVER_BATCH_SIZE == "sum_over_batch_size"
    ctc_inst = losses.CTC()
    assert ctc_inst.name == "ctc"

    auc_inst = metrics.AUC()
    assert auc_inst.interpolate_pr_auc() is not None

    x_img = ops.ones((1, 4, 4, 3), dtype="float32")
    rh = layers.RandomHeight(factor=0.2)
    rw = layers.RandomWidth(factor=0.2)
    trelu = layers.ThresholdedReLU(theta=0.5)
    assert rh(x_img).shape == (1, 4, 4, 3)
    assert rw(x_img).shape == (1, 4, 4, 3)
    assert trelu(ops.convert_to_tensor([-1.0, 0.2, 1.5])).shape == (3,)

    seq = models.Sequential()
    try:
        seq.pop()
    except TypeError:
        pass
    assert seq.input_shape is None
    assert seq.output_shape is None
    seq.add(layers.Dense(4))
    seq.add("non_layer_placeholder")
    seq.build((2, 3))
    assert seq.input_shape == (2, 3)
    assert seq.output_shape == (2, 4)
    popped = seq.pop()
    assert popped == "non_layer_placeholder"
    seq.build([2, 3])

    out_s = ops.image.scale_and_translate(x_img, (1, 2, 2, 3))
    assert out_s.shape == (1, 2, 2, 3)
    out_s2 = ops.image.scale_and_translate(x_img, 2)
    assert out_s2.shape == (1, 4, 4, 3)
    assert ops.multi_hot(ops.convert_to_tensor([[0, 1]], dtype="int32"), num_classes=3) is not None

    nested = {"w": ops.ones((2, 2)), "b": ops.zeros((2,))}
    mapped = tree.map_structure(lambda t: ops.add(t, 1.0), nested)
    assert isinstance(mapped, dict) and "w" in mapped
    clstm = layers.ConvLSTM2D(filters=2, kernel_size=(2, 2), strides=(1, 1), dilation_rate=(1, 1))
    assert clstm.cell.strides == (1, 1)

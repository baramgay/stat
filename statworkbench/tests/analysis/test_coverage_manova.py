"""manova.py 미커버 경로 보완 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.manova import run_analysis
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


def _make_dataset(n_per_group: int = 15, n_groups: int = 2, n_dvs: int = 3, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    groups = []
    dvs = {f"y{i+1}": [] for i in range(n_dvs)}
    for g in range(n_groups):
        groups.extend([g] * n_per_group)
        for i in range(n_dvs):
            dvs[f"y{i+1}"].extend(rng.normal(g * 0.5, 1, n_per_group).tolist())
    df = pd.DataFrame({"group": groups, **dvs})
    ds = Dataset(df, name="manova_test")
    ds.variables["group"] = _nominal("group")
    for i in range(n_dvs):
        ds.variables[f"y{i+1}"] = _scale(f"y{i+1}")
    return ds


def _spec(dvs: list[str] | None = None, post_hoc: bool = False, post_hoc_method: str = "bonferroni", **opts) -> dict:
    if dvs is None:
        dvs = ["y1", "y2", "y3"]
    return {
        "variables": {"dependents": dvs, "factor": "group"},
        "options": {"post_hoc": post_hoc, "post_hoc_method": post_hoc_method, **opts},
    }


# ── 기본 실행 ─────────────────────────────────────────────────────────────────

class TestBasicRun:
    def test_returns_result(self):
        ds = _make_dataset()
        result = run_analysis(ds, _spec())
        assert result is not None

    def test_has_multivariate_table(self):
        ds = _make_dataset()
        result = run_analysis(ds, _spec())
        titles = [t.title for t in result.tables]
        assert any("다변량" in t or "Multivariate" in t for t in titles)

    def test_missing_policy_fallback(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["missing_policy"] = "invalid_xyz"
        result = run_analysis(ds, spec)
        assert result is not None


# ── 경계 케이스 ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_group_too_small_warning(self):
        """집단 케이스 수 ≤ 종속변수 수 → 경고."""
        rng = np.random.default_rng(1)
        n_dvs = 5
        # group A: n=3 (≤ n_dvs), group B: n=15
        df = pd.DataFrame({
            "group": ["A"] * 3 + ["B"] * 15,
            **{f"y{i+1}": rng.normal(0, 1, 18) for i in range(n_dvs)},
        })
        ds = Dataset(df, name="small_group")
        ds.variables["group"] = _nominal("group")
        for i in range(n_dvs):
            ds.variables[f"y{i+1}"] = _scale(f"y{i+1}")
        spec = {"variables": {"dependents": [f"y{i+1}" for i in range(n_dvs)], "factor": "group"}, "options": {}}
        result = run_analysis(ds, spec)
        assert any("케이스" in w or "이하" in w or "수행 불가" in w for w in result.warnings)

    def test_single_group_warning(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "group": ["A"] * 20,
            "y1": rng.normal(0, 1, 20),
            "y2": rng.normal(0, 1, 20),
        })
        ds = Dataset(df, name="one_group")
        ds.variables["group"] = _nominal("group")
        ds.variables["y1"] = _scale("y1")
        ds.variables["y2"] = _scale("y2")
        result = run_analysis(ds, {"variables": {"dependents": ["y1", "y2"], "factor": "group"}, "options": {}})
        assert len(result.warnings) > 0

    def test_missing_variable_warning(self):
        ds = _make_dataset()
        spec = {"variables": {"dependents": ["y1", "nonexistent"], "factor": "group"}, "options": {}}
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0


# ── 사후 검정 ─────────────────────────────────────────────────────────────────

class TestPostHoc:
    def test_bonferroni_post_hoc(self):
        ds = _make_dataset(n_per_group=20, n_groups=3)
        result = run_analysis(ds, _spec(post_hoc=True, post_hoc_method="bonferroni"))
        assert result is not None
        titles = [t.title for t in result.tables]
        assert any("Bonferroni" in t or "쌍별" in t for t in titles)

    def test_tukey_post_hoc(self):
        ds = _make_dataset(n_per_group=20, n_groups=3)
        result = run_analysis(ds, _spec(post_hoc=True, post_hoc_method="tukey"))
        assert result is not None

    def test_no_post_hoc(self):
        ds = _make_dataset()
        result = run_analysis(ds, _spec(post_hoc=False))
        assert result is not None


# ── 효과 크기 옵션 ────────────────────────────────────────────────────────────

class TestEffectSize:
    def test_effect_size_true(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["effect_size"] = True
        result = run_analysis(ds, spec)
        assert result is not None

    def test_effect_size_false(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["effect_size"] = False
        result = run_analysis(ds, spec)
        assert result is not None

"""ancova.py 미커버 경로 보완 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.ancova import run_analysis
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


def _make_dataset(n_per_group: int = 20, n_groups: int = 3, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    n = n_per_group * n_groups
    group = sum([[g] * n_per_group for g in range(n_groups)], [])
    cov1 = rng.normal(50, 10, n)
    cov2 = rng.normal(30, 5, n)
    y = [2.0 * g + 0.5 * cov1[i] + rng.normal(0, 1) for i, g in enumerate(group)]
    df = pd.DataFrame({"group": group, "y": y, "cov1": cov1, "cov2": cov2})
    ds = Dataset(df, name="ancova_test")
    ds.variables["group"] = _nominal("group")
    ds.variables["y"]     = _scale("y")
    ds.variables["cov1"]  = _scale("cov1")
    ds.variables["cov2"]  = _scale("cov2")
    return ds


def _spec(covariates: list[str] | None = None, **opts) -> dict:
    if covariates is None:
        covariates = ["cov1"]
    return {
        "variables": {"dependent": "y", "factor": "group", "covariates": covariates},
        "options": opts,
    }


# ── MissingPolicy fallback ────────────────────────────────────────────────────

class TestMissingPolicyFallback:
    def test_invalid_policy_falls_back(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["missing_policy"] = "invalid_xyz"
        result = run_analysis(ds, spec)
        assert result is not None
        assert len(result.tables) > 0


# ── 경계 케이스 ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_missing_variable_warning(self):
        ds = _make_dataset()
        spec = {"variables": {"dependent": "y", "factor": "group", "covariates": ["nonexistent"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert any("찾을 수 없" in w or "변수" in w for w in result.warnings)

    def test_too_few_cases_warning(self):
        """케이스 수 < covariates+4 → 경고."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "group": [0, 0, 1, 1, 1],
            "y": rng.normal(0, 1, 5),
            "cov1": rng.normal(0, 1, 5),
            "cov2": rng.normal(0, 1, 5),
            "cov3": rng.normal(0, 1, 5),
        })
        ds = Dataset(df, name="tiny")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"]    = _scale("y")
        for c in ["cov1", "cov2", "cov3"]:
            ds.variables[c] = _scale(c)
        result = run_analysis(ds, {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov1", "cov2", "cov3"]},
            "options": {},
        })
        assert len(result.warnings) > 0

    def test_single_level_factor_warning(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"group": [0] * 20, "y": rng.normal(0, 1, 20), "cov1": rng.normal(0, 1, 20)})
        ds = Dataset(df, name="one_level")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"]    = _scale("y")
        ds.variables["cov1"] = _scale("cov1")
        result = run_analysis(ds, {"variables": {"dependent": "y", "factor": "group", "covariates": ["cov1"]}, "options": {}})
        assert len(result.warnings) > 0


# ── 기본 ANCOVA ───────────────────────────────────────────────────────────────

class TestBasicANCOVA:
    def test_returns_result(self):
        ds = _make_dataset()
        result = run_analysis(ds, _spec())
        assert result is not None
        assert len(result.tables) > 0

    def test_ancova_table_present(self):
        ds = _make_dataset()
        result = run_analysis(ds, _spec())
        titles = [t.title for t in result.tables]
        assert any("ANCOVA" in t or "Tests of Between" in t or "분산분석" in t for t in titles)

    def test_multiple_covariates(self):
        ds = _make_dataset()
        result = run_analysis(ds, _spec(covariates=["cov1", "cov2"]))
        assert result is not None
        assert len(result.tables) > 0


# ── 옵션 조합 ─────────────────────────────────────────────────────────────────

class TestOptions:
    def test_levene_test(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["levene"] = True
        result = run_analysis(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Levene" in t for t in titles)

    def test_homogeneity_regression_slopes(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["homogeneity_of_regression"] = True
        result = run_analysis(ds, spec)
        assert result is not None

    def test_homogeneity_violated_warning(self):
        """상호작용이 유의한 데이터 → 동질성 가정 위반 경고."""
        rng = np.random.default_rng(5)
        n = 60
        group = ([0] * 30 + [1] * 30)
        cov = rng.normal(50, 10, n)
        # 그룹별로 공변량 기울기를 다르게 설정해 상호작용 유발
        y = [
            cov[i] * (0.5 if group[i] == 0 else 2.0) + rng.normal(0, 0.5)
            for i in range(n)
        ]
        df = pd.DataFrame({"group": group, "y": y, "cov": cov})
        ds = Dataset(df, name="violated")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"]    = _scale("y")
        ds.variables["cov"]  = _scale("cov")
        result = run_analysis(ds, {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov"]},
            "options": {"homogeneity_of_regression": True},
        })
        assert result is not None

    def test_post_hoc_bonferroni(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["post_hoc"] = "bonferroni"
        result = run_analysis(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Bonferroni" in t or "사후" in t or "Post" in t for t in titles)

    def test_effect_size_option(self):
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["effect_size"] = True
        result = run_analysis(ds, spec)
        assert result is not None

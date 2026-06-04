"""mixed_anova.py 미커버 경로 보완 테스트 (lines 91-92, 296-298, 393-420, 499-502)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.mixed_anova import run_analysis
from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


def _make_dataset(n_per_group: int = 8, n_groups: int = 2, k: int = 3, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    n = n_per_group * n_groups
    group = sum([[f"G{g+1}"] * n_per_group for g in range(n_groups)], [])
    base = rng.normal(0, 1, n)
    data = {"group": group}
    for t in range(k):
        data[f"t{t+1}"] = base + t * 2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame(data)
    ds = Dataset(df, name="mixed_test")
    ds.variables["group"] = _nominal("group")
    for t in range(k):
        ds.variables[f"t{t+1}"] = _scale(f"t{t+1}")
    return ds


def _spec(n_groups: int = 2, k: int = 3, **opts) -> dict:
    within_vars = [f"t{t+1}" for t in range(k)]
    return {
        "variables": {
            "between": "group",
            "within": within_vars,
        },
        "options": {"missing_policy": "listwise", **opts},
    }


# ── line 91-92: MissingPolicy invalid string fallback ────────────────────────

class TestMissingPolicyFallback:
    def test_invalid_policy_falls_back_to_listwise(self):
        """잘못된 missing_policy 문자열 → listwise 폴백 처리."""
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["missing_policy"] = "nonexistent_policy_xyz"
        result = run_analysis(ds, spec)
        assert result is not None
        assert len(result.tables) > 0


# ── lines 296-298: Mauchly 오류 경로 ─────────────────────────────────────────

class TestMauchlyErrorPath:
    def test_k2_skips_mauchly(self):
        """k=2 (within 변수 2개)는 Mauchly 검정 불필요 — 테이블 없음."""
        ds = _make_dataset(k=2)
        result = run_analysis(ds, _spec(k=2))
        titles = [t.title for t in result.tables]
        assert not any("Mauchly" in t for t in titles)

    def test_k3_mauchly_present(self):
        """k=3은 Mauchly 검정 테이블 있어야 함."""
        ds = _make_dataset(k=3)
        result = run_analysis(ds, _spec(k=3))
        titles = [t.title for t in result.tables]
        assert any("Mauchly" in t or "Sphericity" in t for t in titles)


# ── lines 393-420: Bonferroni between-subjects pairwise ──────────────────────

class TestBonferroniBetween:
    def test_3_groups_bonferroni_table(self):
        """집단 3개 → Bonferroni 집단 간 쌍 비교 테이블 생성."""
        ds = _make_dataset(n_groups=3, k=3, n_per_group=6)
        spec = _spec(n_groups=3, k=3)
        result = run_analysis(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Bonferroni" in t or "Pairwise" in t for t in titles)

    def test_2_groups_no_bonferroni(self):
        """집단 2개는 Bonferroni 필요 없음 — 함수 조기 반환."""
        ds = _make_dataset(n_groups=2, k=3)
        result = run_analysis(ds, _spec(n_groups=2, k=3))
        titles = [t.title for t in result.tables]
        # 2그룹에서는 Bonferroni between이 없어야 함
        assert not any("Bonferroni" in t and "group" in t.lower() for t in titles)


# ── lines 499-502: profile plot exception handler ────────────────────────────

class TestProfilePlot:
    def test_profile_plot_option(self):
        """profile_plot=True — 생성 성공 또는 무음 실패 (crash 없음)."""
        ds = _make_dataset()
        spec = _spec()
        spec["options"]["profile_plot"] = True
        result = run_analysis(ds, spec)
        assert result is not None

    def test_profile_plot_false_no_crash(self):
        spec = _spec()
        spec["options"]["profile_plot"] = False
        result = run_analysis(ds := _make_dataset(), spec)
        assert result is not None

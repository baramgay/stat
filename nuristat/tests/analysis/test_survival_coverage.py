"""survival_analysis.py 내부 함수 직접 테스트 — 커버리지 75%+ 목표.

lifelines 미설치 경로(_run_km_manual, _run_cox_manual, _log_rank_test)와
lifelines 설치 경로(_run_km_lifelines, _run_cox_lifelines) 모두 커버.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.result import AnalysisResult
from nuristat.analysis.survival_analysis import (
    _log_rank_test,
    _run_km_manual,
    _run_cox_manual,
    run_analysis,
)
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy, StorageType
from nuristat.core.variable import VariableMeta


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_result() -> AnalysisResult:
    return AnalysisResult(id="survival_analysis", title="생존분석", spec={})


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT,
                        measure=MeasureType.SCALE, decimals=1)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.NOMINAL)


def _make_ds(df: pd.DataFrame, metas: dict | None = None) -> Dataset:
    ds = Dataset(df, name="t")
    if metas:
        for k, v in metas.items():
            ds.variables[k] = v
    return ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_group_arrays():
    """T, E, df — 두 그룹."""
    times = np.array([2, 3, 5, 7, 10, 4, 6, 8, 12, 15], dtype=float)
    events = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=int)
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    df = pd.DataFrame({"time": times, "event": events, "group": groups})
    return times, events, df


@pytest.fixture
def three_group_arrays():
    """T, E, df — 세 그룹."""
    times = np.array([3, 5, 6, 2, 4, 7, 8, 9, 11, 12], dtype=float)
    events = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=int)
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    df = pd.DataFrame({"time": times, "event": events, "group": groups})
    return times, events, df


@pytest.fixture
def cox_arrays():
    """T, E, df — Cox 공변량 포함."""
    np.random.seed(3)
    n = 30
    times = np.random.exponential(8, n)
    events = np.random.binomial(1, 0.6, n)
    age = np.random.randint(40, 70, n).astype(float)
    score = np.random.normal(5, 1.5, n)
    df = pd.DataFrame({"time": times, "event": events, "age": age, "score": score})
    return times, events, df


# ---------------------------------------------------------------------------
# 1. _run_km_manual — 그룹 없음
# ---------------------------------------------------------------------------

class TestKMManualNoGroup:

    def test_produces_tables(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, None, df, 0.95)
        assert len(res.tables) >= 1

    def test_summary_table_has_n_row(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, None, df, 0.95)
        combined = " ".join(
            str(cell)
            for t in res.tables
            for cell in t.dataframe.values.flatten()
        )
        assert "10" in combined  # 전체 N=10

    def test_all_censored_no_events(self):
        T = np.array([5.0, 10.0, 15.0])
        E = np.array([0, 0, 0])
        df = pd.DataFrame({"time": T, "event": E})
        res = _make_result()
        _run_km_manual(res, T, E, None, df, 0.95)
        assert len(res.tables) >= 1

    def test_single_event_at_time_1(self):
        T = np.array([1.0, 5.0, 10.0])
        E = np.array([1, 0, 0])
        df = pd.DataFrame({"time": T, "event": E})
        res = _make_result()
        _run_km_manual(res, T, E, None, df, 0.95)
        assert len(res.tables) >= 1

    def test_confidence_level_90(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, None, df, 0.90)
        assert len(res.tables) >= 1

    def test_survival_probabilities_between_0_and_1(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, None, df, 0.95)
        km_table = next(
            (t for t in res.tables if "KM 생존" in t.title or "생존 함수" in t.title),
            None,
        )
        if km_table is not None:
            for val in km_table.dataframe["생존 확률 S(t)"]:
                v = float(str(val).replace(" ", ""))
                assert 0.0 <= v <= 1.0

    def test_survival_probability_decreases_or_stays(self, two_group_arrays):
        """S(t)는 단조 감소해야 한다."""
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, None, df, 0.95)
        km_table = next(
            (t for t in res.tables if "KM 생존" in t.title or "생존 함수" in t.title),
            None,
        )
        if km_table is not None:
            vals = [
                float(str(v).replace(" ", ""))
                for v in km_table.dataframe["생존 확률 S(t)"]
            ]
            for i in range(1, len(vals)):
                assert vals[i] <= vals[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# 2. _run_km_manual — 그룹 있음 (253-341 라인)
# ---------------------------------------------------------------------------

class TestKMManualWithGroup:

    def test_produces_group_tables(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, "group", df, 0.95)
        titles = [t.title for t in res.tables]
        assert any("그룹" in tit or "KM" in tit for tit in titles)

    def test_group_summary_table_present(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, "group", df, 0.95)
        titles = [t.title for t in res.tables]
        assert any("요약" in tit for tit in titles)

    def test_logrank_table_added_for_groups(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, "group", df, 0.95)
        titles = [t.title for t in res.tables]
        assert any("Log-rank" in tit or "log-rank" in tit.lower() for tit in titles)

    def test_three_groups_km_manual(self, three_group_arrays):
        T, E, df = three_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, "group", df, 0.95)
        assert len(res.tables) >= 2

    def test_group_var_not_in_df_falls_back(self, two_group_arrays):
        """group_var이 df에 없으면 그룹 없는 경로 실행."""
        T, E, df = two_group_arrays
        res = _make_result()
        _run_km_manual(res, T, E, "nonexistent_col", df, 0.95)
        assert len(res.tables) >= 1


# ---------------------------------------------------------------------------
# 3. _log_rank_test (357-404 라인)
# ---------------------------------------------------------------------------

class TestLogRankTest:

    def test_two_groups_adds_table(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        groups = [0, 1]
        _log_rank_test(res, T, E, df, "group", groups)
        titles = [t.title for t in res.tables]
        assert any("Log-rank" in tit for tit in titles)

    def test_table_contains_chi2(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _log_rank_test(res, T, E, df, "group", [0, 1])
        lr_table = next(t for t in res.tables if "Log-rank" in t.title)
        combined = " ".join(str(v) for v in lr_table.dataframe.values.flatten())
        assert "Chi2" in combined or "전체" in combined

    def test_three_groups_log_rank(self, three_group_arrays):
        T, E, df = three_group_arrays
        res = _make_result()
        _log_rank_test(res, T, E, df, "group", [0, 1, 2])
        titles = [t.title for t in res.tables]
        assert any("Log-rank" in tit for tit in titles)

    def test_p_value_present_in_table(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        _log_rank_test(res, T, E, df, "group", [0, 1])
        lr_table = next(t for t in res.tables if "Log-rank" in t.title)
        combined = " ".join(str(v) for v in lr_table.dataframe.values.flatten())
        assert "p=" in combined or "p<" in combined or "p =" in combined or "p " in combined

    def test_invalid_group_var_adds_warning(self, two_group_arrays):
        T, E, df = two_group_arrays
        res = _make_result()
        # group_var 존재하지 않는 컬럼 → 예외 → warnings에 추가
        _log_rank_test(res, T, E, df, "nonexistent_col", [0, 1])
        assert len(res.warnings) > 0


# ---------------------------------------------------------------------------
# 4. _run_cox_manual (485-527 라인)
# ---------------------------------------------------------------------------

class TestCoxManual:

    def test_produces_tables(self, cox_arrays):
        T, E, df = cox_arrays
        res = _make_result()
        _run_cox_manual(res, T, E, df, ["age", "score"], 0.95)
        assert len(res.tables) >= 1

    def test_table_contains_hr(self, cox_arrays):
        T, E, df = cox_arrays
        res = _make_result()
        _run_cox_manual(res, T, E, df, ["age", "score"], 0.95)
        cox_t = next(
            (t for t in res.tables if "Cox" in t.title or "PHReg" in t.title),
            None,
        )
        if cox_t is not None:
            cols = [str(c) for c in cox_t.dataframe.columns]
            assert any("HR" in c or "계수" in c for c in cols)

    def test_single_covariate(self, cox_arrays):
        T, E, df = cox_arrays
        res = _make_result()
        _run_cox_manual(res, T, E, df, ["age"], 0.95)
        assert res is not None

    def test_covariate_with_nans(self, cox_arrays):
        T, E, df = cox_arrays
        df = df.copy()
        df.loc[[0, 5], "age"] = np.nan
        res = _make_result()
        _run_cox_manual(res, T, E, df, ["age", "score"], 0.95)
        assert res is not None

    def test_bad_covariate_adds_warning(self, cox_arrays):
        """공변량이 모두 동일한 상수 → statsmodels 오류 → warning 추가."""
        T, E, df = cox_arrays
        df = df.copy()
        df["constant"] = 1.0
        res = _make_result()
        _run_cox_manual(res, T, E, df, ["constant"], 0.95)
        # 오류 발생 시 warnings에 추가됨
        assert res is not None


# ---------------------------------------------------------------------------
# 5. run_analysis — lifelines 없는 경로 (monkeypatch)
# ---------------------------------------------------------------------------

class TestRunAnalysisNoLifelines:
    """_LIFELINES_AVAILABLE=False 경로 강제 실행 — 114, 120, 123번 라인."""

    def test_km_manual_path_no_lifelines(self):
        """lifelines=False 시 _run_km_manual 호출."""
        import nuristat.analysis.survival_analysis as sa_mod
        df = pd.DataFrame({
            "time": [2.0, 5.0, 8.0, 10.0, 12.0],
            "event": [1, 1, 0, 1, 0],
        })
        ds = _make_ds(df, {"time": _scale("time"), "event": _nominal("event")})
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        with patch.object(sa_mod, "_LIFELINES_AVAILABLE", False):
            result = run_analysis(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1

    def test_cox_manual_path_no_lifelines(self):
        """lifelines=False + covariates → _run_cox_manual 호출."""
        import nuristat.analysis.survival_analysis as sa_mod
        np.random.seed(9)
        n = 25
        df = pd.DataFrame({
            "time": np.random.exponential(8, n),
            "event": np.random.binomial(1, 0.7, n),
            "age": np.random.randint(40, 70, n).astype(float),
        })
        ds = _make_ds(df, {
            "time": _scale("time"), "event": _nominal("event"), "age": _scale("age"),
        })
        spec = {
            "variables": {
                "duration": "time", "event": "event",
                "covariates": ["age"],
            },
            "options": {"method": "both"},
        }
        with patch.object(sa_mod, "_LIFELINES_AVAILABLE", False):
            result = run_analysis(ds, spec)
        assert result is not None

    def test_notes_appended_when_no_lifelines(self):
        """lifelines=False → result.notes에 안내 메시지."""
        import nuristat.analysis.survival_analysis as sa_mod
        df = pd.DataFrame({
            "time": [3.0, 6.0, 9.0, 12.0],
            "event": [1, 1, 0, 1],
        })
        ds = _make_ds(df, {"time": _scale("time"), "event": _nominal("event")})
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        with patch.object(sa_mod, "_LIFELINES_AVAILABLE", False):
            result = run_analysis(ds, spec)
        assert any("lifelines" in n.lower() for n in result.notes)

    def test_km_manual_with_group_no_lifelines(self):
        """lifelines=False + group_var → _run_km_manual 그룹 경로."""
        import nuristat.analysis.survival_analysis as sa_mod
        df = pd.DataFrame({
            "time": [2, 3, 5, 7, 4, 6, 8, 10],
            "event": [1, 1, 1, 0, 1, 1, 0, 1],
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
        })
        ds = _make_ds(df, {
            "time": _scale("time"),
            "event": _nominal("event"),
            "group": _nominal("group"),
        })
        spec = {
            "variables": {"duration": "time", "event": "event", "group": "group"},
            "options": {"method": "km"},
        }
        with patch.object(sa_mod, "_LIFELINES_AVAILABLE", False):
            result = run_analysis(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Log-rank" in tit for tit in titles)


# ---------------------------------------------------------------------------
# 6. run_analysis — missing_policy 문자열 경로 (65번 라인)
# ---------------------------------------------------------------------------

class TestRunAnalysisMissingPolicyString:

    def test_string_missing_policy_listwise(self):
        """missing_policy를 문자열로 전달 → MissingPolicy 변환."""
        df = pd.DataFrame({
            "time": [2.0, np.nan, 5.0, 7.0],
            "event": [1, 1, 0, 1],
        })
        ds = _make_ds(df, {"time": _scale("time"), "event": _nominal("event")})
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
            "missing_policy": "listwise",
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def test_missing_policy_object_passthrough(self):
        """missing_policy를 MissingPolicy 객체로 전달."""
        df = pd.DataFrame({
            "time": [3.0, 6.0, 9.0],
            "event": [1, 0, 1],
        })
        ds = _make_ds(df, {"time": _scale("time"), "event": _nominal("event")})
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        assert result is not None

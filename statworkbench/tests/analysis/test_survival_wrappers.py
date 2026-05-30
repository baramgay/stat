"""run_kaplan_meier / run_cox_regression wrapper 함수 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("lifelines", reason="lifelines 패키지 필요")

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType
from statworkbench.analysis.survival_analysis import run_kaplan_meier, run_cox_regression


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT,
                        measure=MeasureType.SCALE, decimals=1)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.NOMINAL)


CTRL_TIME  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
CTRL_EVENT = [1, 1, 1, 1, 1, 1, 0, 1,  0,  1]
TRT_TIME   = [6, 8, 9, 11, 14, 15, 16, 18, 20, 22]
TRT_EVENT  = [1, 1, 0,  1,  1,  0,  1,  1,  0,  1]


def _make_km_dataset():
    times  = CTRL_TIME + TRT_TIME
    events = CTRL_EVENT + TRT_EVENT
    groups = [0] * 10 + [1] * 10
    df = pd.DataFrame({"time": times, "event": events, "group": groups})
    ds = Dataset(df, name="km_test")
    ds.variables["time"]  = _scale("time")
    ds.variables["event"] = _nominal("event")
    ds.variables["group"] = _nominal("group")
    return ds


def _make_cox_dataset():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 50
    age = rng.normal(55, 10, n)
    event = rng.binomial(1, 0.6, n).astype(float)
    time = rng.exponential(10, n) + 1
    df = pd.DataFrame({"time": time, "event": event, "age": age})
    ds = Dataset(df, name="cox_test")
    ds.variables["time"]  = _scale("time")
    ds.variables["event"] = _nominal("event")
    ds.variables["age"]   = _scale("age")
    return ds


# ──────────────────────────────────────────────────────────────
# Kaplan-Meier wrapper
# ──────────────────────────────────────────────────────────────

class TestRunKaplanMeier:
    def _spec(self, group=True):
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {},
            "confidence_level": 0.95,
        }
        if group:
            spec["variables"]["group"] = "group"
        return spec

    def test_returns_analysis_result(self):
        from statworkbench.analysis.result import AnalysisResult
        ds = _make_km_dataset()
        result = run_kaplan_meier(ds, self._spec())
        assert isinstance(result, AnalysisResult)

    def test_result_id(self):
        ds = _make_km_dataset()
        result = run_kaplan_meier(ds, self._spec())
        assert result.id == "kaplan_meier"

    def test_result_title(self):
        ds = _make_km_dataset()
        result = run_kaplan_meier(ds, self._spec())
        assert result.title == "Kaplan-Meier 생존분석"

    def test_has_tables(self):
        ds = _make_km_dataset()
        result = run_kaplan_meier(ds, self._spec())
        assert len(result.tables) > 0

    def test_logrank_chi2_in_results(self):
        """로그순위 검정 chi² ≈ 7.728 (SPSS 29 기준)."""
        ds = _make_km_dataset()
        result = run_kaplan_meier(ds, self._spec(group=True))
        # 테이블에서 검정 통계량 추출
        chi2_found = False
        for table in result.tables:
            df = table.dataframe
            for col in df.columns:
                for val in df[col].astype(str).values:
                    try:
                        if abs(float(val) - 7.728) < 0.1:
                            chi2_found = True
                    except ValueError:
                        pass
        assert chi2_found, f"로그순위 chi² ≈ 7.728 not found in tables"

    def test_no_group_still_runs(self):
        ds = _make_km_dataset()
        result = run_kaplan_meier(ds, self._spec(group=False))
        assert result.id == "kaplan_meier"
        assert len(result.tables) > 0

    def test_options_method_overridden(self):
        """options.method 가 외부에서 설정되어도 km으로 강제됨."""
        ds = _make_km_dataset()
        spec = self._spec(group=False)
        spec["options"]["method"] = "cox"
        result = run_kaplan_meier(ds, spec)
        assert result.id == "kaplan_meier"


# ──────────────────────────────────────────────────────────────
# Cox wrapper
# ──────────────────────────────────────────────────────────────

class TestRunCoxRegression:
    def _spec(self):
        return {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age"],
            },
            "options": {},
            "confidence_level": 0.95,
        }

    def test_returns_analysis_result(self):
        from statworkbench.analysis.result import AnalysisResult
        ds = _make_cox_dataset()
        result = run_cox_regression(ds, self._spec())
        assert isinstance(result, AnalysisResult)

    def test_result_id(self):
        ds = _make_cox_dataset()
        result = run_cox_regression(ds, self._spec())
        assert result.id == "cox_regression"

    def test_result_title(self):
        ds = _make_cox_dataset()
        result = run_cox_regression(ds, self._spec())
        assert result.title == "Cox 비례위험 회귀"

    def test_has_tables(self):
        ds = _make_cox_dataset()
        result = run_cox_regression(ds, self._spec())
        assert len(result.tables) > 0

    def test_hr_column_present(self):
        """HR(Hazard Ratio) 컬럼이 결과 테이블에 존재해야 함."""
        ds = _make_cox_dataset()
        result = run_cox_regression(ds, self._spec())
        hr_found = any(
            any("HR" in str(c) or "Exp(B)" in str(c) or "exp" in str(c).lower()
                for c in t.dataframe.columns)
            for t in result.tables
        )
        assert hr_found, "HR 컬럼이 Cox 결과 테이블에 없음"

    def test_age_covariate_in_output(self):
        """age 공변량이 결과 테이블에 나타나야 함."""
        ds = _make_cox_dataset()
        result = run_cox_regression(ds, self._spec())
        age_found = any(
            "age" in t.dataframe.to_string().lower()
            for t in result.tables
        )
        assert age_found, "age 공변량이 Cox 결과에 없음"

    def test_options_method_overridden(self):
        """options.method 가 외부에서 설정되어도 cox로 강제됨."""
        ds = _make_cox_dataset()
        spec = self._spec()
        spec["options"]["method"] = "km"
        result = run_cox_regression(ds, spec)
        assert result.id == "cox_regression"

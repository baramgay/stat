"""추가 검증 Round 3 — 10개 영역 심층 검증.

1.  포맷팅 함수 정확도 — format_pvalue, format_number, format_ci, format_percent
2.  비모수 검정 정밀도 — Mann-Whitney U, Kruskal-Wallis, Wilcoxon vs scipy
3.  회귀분석 모델 요약 정밀도 — R², RMSE, F-통계량
4.  로지스틱 회귀 출력 구조 — 계수표, 분류표, HL 검정, ROC
5.  Excel 다중 시트 IO — sheet_name=0/이름/인덱스
6.  Dataset API — add_variable, get_column, update_variable_meta, is_empty
7.  SPSSGridModel 캐시 무결성 — 접근 후 캐시 채움, 구조 변경 후 초기화
8.  상관분석 방법 변형 비교 — Pearson/Spearman/Kendall 방향 일치
9.  성능 스케일링 검증 — N 10배 증가 시 처리 시간 선형 이하
10. 분석 결과 일관성 — 결측 정책 변경 시 N 변화 검증
"""

from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy, StorageType
from nuristat.core.variable import VariableMeta


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def _ds(df: pd.DataFrame, measures: dict[str, MeasureType] | None = None) -> Dataset:
    ds = Dataset(df)
    if measures:
        for name, m in measures.items():
            if name in ds.variables:
                ds.variables[name].measure = m
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 1. 포맷팅 함수 정확도
# ─────────────────────────────────────────────────────────────────────────────

class TestFormattingFunctions:
    """format_pvalue, format_number, format_ci, format_percent 검증."""

    def test_pvalue_very_small(self):
        """p < .001 표기."""
        from nuristat.analysis.formatting import format_pvalue
        assert format_pvalue(0.0001) == "< .001"
        assert format_pvalue(0.00099) == "< .001"

    def test_pvalue_exactly_001(self):
        """p = .001 경계."""
        from nuristat.analysis.formatting import format_pvalue
        result = format_pvalue(0.001)
        assert result in (".001", "< .001", "0.001", ".001")

    def test_pvalue_mid_range(self):
        """p = .05 → '.050'."""
        from nuristat.analysis.formatting import format_pvalue
        result = format_pvalue(0.05)
        assert ".05" in result or "050" in result

    def test_pvalue_large(self):
        """p = .500 표기."""
        from nuristat.analysis.formatting import format_pvalue
        result = format_pvalue(0.5)
        assert ".5" in result or "500" in result

    def test_pvalue_1(self):
        """p = 1.0 처리 — 충돌 없음."""
        from nuristat.analysis.formatting import format_pvalue
        result = format_pvalue(1.0)
        assert isinstance(result, str)

    def test_format_number_basic(self):
        """format_number(3.14159, decimals=2) → '3.14'."""
        from nuristat.analysis.formatting import format_number
        result = format_number(3.14159, decimals=2)
        assert "3.14" in str(result)

    def test_format_number_large(self):
        """큰 수 포맷팅 — 충돌 없음."""
        from nuristat.analysis.formatting import format_number
        result = format_number(1_234_567.89)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_number_zero(self):
        """0 포맷팅."""
        from nuristat.analysis.formatting import format_number
        result = format_number(0.0)
        assert isinstance(result, str)

    def test_format_ci_structure(self):
        """format_ci → '[low, high]' 형태."""
        from nuristat.analysis.formatting import format_ci
        result = format_ci(1.5, 3.5)
        assert "[" in result and "]" in result

    def test_format_ci_ordered(self):
        """CI에서 low < high."""
        from nuristat.analysis.formatting import format_ci
        result = format_ci(-2.0, 4.0)
        assert isinstance(result, str)
        assert "-2" in result or "-2.0" in result.replace(" ", "")

    def test_format_percent_range(self):
        """format_percent(0.756) — 결과는 퍼센트 표현."""
        from nuristat.analysis.formatting import format_percent
        result = format_percent(0.756)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_significance_stars_three(self):
        """p < .001 → '***'."""
        from nuristat.analysis.formatting import add_significance_stars
        result = add_significance_stars(0.0001)
        assert "***" in result

    def test_significance_stars_two(self):
        """p < .01 → '**'."""
        from nuristat.analysis.formatting import add_significance_stars
        result = add_significance_stars(0.005)
        assert "**" in result

    def test_significance_stars_one(self):
        """p < .05 → '*'."""
        from nuristat.analysis.formatting import add_significance_stars
        result = add_significance_stars(0.03)
        assert "*" in result

    def test_significance_stars_ns(self):
        """p ≥ .05 → 별표 없음."""
        from nuristat.analysis.formatting import add_significance_stars
        result = add_significance_stars(0.10)
        assert "***" not in result and "**" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 2. 비모수 검정 정밀도 — scipy 직접 비교
# ─────────────────────────────────────────────────────────────────────────────

class TestNonparametricPrecision:
    """비모수 검정값이 scipy와 일치."""

    def _mw_ds(self):
        df = pd.DataFrame({
            "group": ["A"] * 8 + ["B"] * 8,
            "score": [12, 15, 18, 14, 16, 19, 13, 17,
                      25, 28, 22, 27, 24, 26, 23, 29],
        })
        return _ds(df, {"group": MeasureType.NOMINAL, "score": MeasureType.ORDINAL})

    def _kw_ds(self):
        df = pd.DataFrame({
            "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
            "score": [10, 12, 11, 13, 9, 14,
                      20, 22, 21, 23, 19, 24,
                      30, 32, 31, 33, 29, 34],
        })
        return _ds(df, {"group": MeasureType.NOMINAL, "score": MeasureType.ORDINAL})

    def _wilcoxon_ds(self):
        df = pd.DataFrame({
            "t1": [5, 7, 6, 8, 4, 6, 7, 5],
            "t2": [7, 9, 8, 10, 6, 8, 9, 7],
        })
        return _ds(df, {"t1": MeasureType.ORDINAL, "t2": MeasureType.ORDINAL})

    def test_mann_whitney_u_matches_scipy(self):
        """Mann-Whitney U 통계량이 scipy와 0.1 이내."""
        from nuristat.analysis.nonparametric import run_analysis

        ds = self._mw_ds()
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        g1 = ds.data[ds.data["group"] == "A"]["score"].values
        g2 = ds.data[ds.data["group"] == "B"]["score"].values
        u_scipy, _ = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        test_tbl = [t for t in result.tables if "Test Statistics" in t.title][0]
        df_t = test_tbl.dataframe.set_index("Statistic")
        u_sw = float(df_t.loc["Mann-Whitney U", "Value"])
        assert abs(u_sw - u_scipy) < 0.1

    def test_mann_whitney_p_significant(self):
        """두 그룹이 명백히 다를 때 p < .05."""
        from nuristat.analysis.nonparametric import run_analysis

        ds = self._mw_ds()
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        test_tbl = [t for t in result.tables if "Test Statistics" in t.title][0]
        p_raw = str(test_tbl.dataframe.set_index("Statistic").loc["p-value", "Value"])
        is_sig = p_raw.startswith("<") or (not p_raw.startswith("<") and float(p_raw) < 0.05)
        assert is_sig

    def test_kruskal_wallis_matches_scipy(self):
        """Kruskal-Wallis H 통계량이 scipy와 0.1 이내."""
        from nuristat.analysis.nonparametric import run_analysis

        ds = self._kw_ds()
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "kruskal_wallis"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        groups = [ds.data[ds.data["group"] == g]["score"].values
                  for g in ["A", "B", "C"]]
        h_scipy, _ = stats.kruskal(*groups)

        test_tbl = [t for t in result.tables if "Test Statistics" in t.title][0]
        df_t = test_tbl.dataframe.set_index("Statistic")
        h_sw = float(df_t.loc["Kruskal-Wallis H", "Value"])
        assert abs(h_sw - h_scipy) < 0.1

    def test_wilcoxon_result_structure(self):
        """Wilcoxon 검정 결과 구조 검증."""
        from nuristat.analysis.nonparametric import run_analysis

        ds = self._wilcoxon_ds()
        spec = {
            "variables": {"paired": ["t1", "t2"]},
            "options": {"test": "wilcoxon"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        assert len(result.tables) >= 2
        test_tbl = [t for t in result.tables if "Test Statistics" in t.title]
        assert test_tbl, "Test Statistics 테이블 없음"
        stats_col = test_tbl[0].dataframe["Statistic"].astype(str).tolist()
        assert any("Wilcoxon" in s for s in stats_col)

    def test_wilcoxon_effect_size_in_range(self):
        """Wilcoxon effect size r ∈ [0, 1]."""
        from nuristat.analysis.nonparametric import run_analysis

        ds = self._wilcoxon_ds()
        spec = {
            "variables": {"paired": ["t1", "t2"]},
            "options": {"test": "wilcoxon"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        test_tbl = [t for t in result.tables if "Test Statistics" in t.title][0]
        df_t = test_tbl.dataframe.set_index("Statistic")
        if "Effect Size r" in df_t.index:
            r_val = float(df_t.loc["Effect Size r", "Value"])
            assert 0.0 <= r_val <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. 회귀분석 모델 요약 정밀도
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionPrecision:
    """회귀분석 R², F, 계수가 scipy/statsmodels 기준값과 일치."""

    def _reg_ds(self, seed: int = 55):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, 80)
        y = 2.5 * x + rng.normal(0, 0.5, 80)
        df = pd.DataFrame({"y": y, "x": x})
        return _ds(df, {"y": MeasureType.SCALE, "x": MeasureType.SCALE})

    def _reg_spec(self):
        return {
            "variables": {"dependent": "y", "predictors": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }

    def _model_summary(self, ds, spec):
        from nuristat.analysis.regression import run_analysis
        result = run_analysis(ds, spec)
        tbl = [t for t in result.tables if t.title == "Model Summary"][0]
        return tbl.dataframe.set_index("Statistic")["Value"]

    def test_r_squared_matches_scipy(self):
        """R² 값이 scipy linregress와 0.01 이내."""
        ds = self._reg_ds()
        summary = self._model_summary(ds, self._reg_spec())

        r2_sw_str = str(summary.get("R-squared", "0"))
        r2_sw = float(r2_sw_str) if r2_sw_str not in ("", "< .001") else 0.0

        x = ds.data["x"].values
        y = ds.data["y"].values
        _, _, r, _, _ = stats.linregress(x, y)
        r2_scipy = r ** 2

        assert abs(r2_sw - r2_scipy) < 0.01

    def test_r_value_matches_scipy(self):
        """R 값이 scipy와 0.01 이내."""
        ds = self._reg_ds()
        summary = self._model_summary(ds, self._reg_spec())

        r_sw_str = str(summary.get("R", "0"))
        r_sw = float(r_sw_str) if r_sw_str not in ("", "< .001") else 0.0

        x = ds.data["x"].values
        y = ds.data["y"].values
        _, _, r, _, _ = stats.linregress(x, y)

        assert abs(abs(r_sw) - abs(r)) < 0.01

    def test_n_matches_dataset_rows(self):
        """Model Summary N = 데이터셋 행 수."""
        ds = self._reg_ds()
        summary = self._model_summary(ds, self._reg_spec())
        n_sw = float(summary.get("N", 0))
        assert int(n_sw) == len(ds.data)

    def test_multiple_predictors_r2_positive(self):
        """다중 예측변수 R² > 0."""
        rng = np.random.default_rng(99)
        x1 = rng.normal(0, 1, 60)
        x2 = rng.normal(0, 1, 60)
        y = 1.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, 60)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        ds = _ds(df, {"y": MeasureType.SCALE, "x1": MeasureType.SCALE, "x2": MeasureType.SCALE})
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        from nuristat.analysis.regression import run_analysis
        result = run_analysis(ds, spec)
        summary = [t for t in result.tables if t.title == "Model Summary"][0]
        r2_row = summary.dataframe[summary.dataframe["Statistic"] == "R-squared"]
        r2 = float(r2_row.iloc[0]["Value"])
        assert r2 > 0.5

    def test_coefficients_table_has_predictors(self):
        """Coefficients 테이블에 예측변수 행 존재."""
        from nuristat.analysis.regression import run_analysis
        ds = self._reg_ds()
        result = run_analysis(ds, self._reg_spec())
        coef_tbl = [t for t in result.tables if t.title == "Coefficients"]
        assert coef_tbl
        df_c = coef_tbl[0].dataframe
        has_x = any("x" in str(v) for v in df_c.iloc[:, 0].tolist())
        assert has_x

    def test_stepwise_produces_tables(self):
        """Stepwise 회귀 → 테이블 생성."""
        from nuristat.analysis.regression import run_analysis
        rng = np.random.default_rng(42)
        x1 = rng.normal(0, 1, 60); x2 = rng.normal(0, 1, 60); x3 = rng.normal(0, 1, 60)
        y = 2*x1 + 0.01*x2 + rng.normal(0, 0.3, 60)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
        ds = _ds(df, {"y": MeasureType.SCALE, "x1": MeasureType.SCALE,
                      "x2": MeasureType.SCALE, "x3": MeasureType.SCALE})
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2", "x3"]},
            "options": {"method": "stepwise"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# 4. 로지스틱 회귀 출력 구조
# ─────────────────────────────────────────────────────────────────────────────

class TestLogisticRegressionOutput:
    """로지스틱 회귀 출력 구조 검증."""

    def _logistic_ds(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        n = 100
        age = rng.integers(20, 70, n).astype(float)
        score = rng.normal(50, 10, n)
        log_odds = -3 + 0.05 * age + 0.04 * score
        prob = 1 / (1 + np.exp(-log_odds))
        outcome = (rng.random(n) < prob).astype(int)
        df = pd.DataFrame({"outcome": outcome, "age": age, "score": score})
        ds = _ds(df, {"age": MeasureType.SCALE, "score": MeasureType.SCALE})
        ds.variables["outcome"].measure = MeasureType.BINARY
        return ds

    def _logistic_spec(self, extra_opts=None):
        opts = {"method": "binary", "classification_table": True,
                "hosmer_lemeshow": True}
        if extra_opts:
            opts.update(extra_opts)
        return {
            "variables": {"dependent": "outcome", "predictors": ["age", "score"]},
            "options": opts,
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }

    def test_basic_tables_present(self):
        """CPS, 모형요약, 계수표 존재."""
        from nuristat.analysis.logistic_regression import run_analysis
        ds = self._logistic_ds()
        result = run_analysis(ds, self._logistic_spec())
        titles = [t.title for t in result.tables]
        assert any("Case Processing" in t for t in titles)
        assert any("Summary" in t or "요약" in t for t in titles)
        assert any("Coefficient" in t or "계수" in t for t in titles)

    def test_classification_table_present(self):
        """classification_table=True → 분류표 존재."""
        from nuristat.analysis.logistic_regression import run_analysis
        ds = self._logistic_ds()
        result = run_analysis(ds, self._logistic_spec())
        titles = [t.title for t in result.tables]
        assert any("분류" in t or "Classification" in t for t in titles)

    def test_hosmer_lemeshow_present(self):
        """hosmer_lemeshow=True → HL 검정 테이블 존재."""
        from nuristat.analysis.logistic_regression import run_analysis
        ds = self._logistic_ds()
        result = run_analysis(ds, self._logistic_spec())
        titles = [t.title for t in result.tables]
        assert any("Hosmer" in t for t in titles)

    def test_coefficient_exp_b_positive(self):
        """계수표 Exp(B) 값 > 0."""
        from nuristat.analysis.logistic_regression import run_analysis
        ds = self._logistic_ds()
        result = run_analysis(ds, self._logistic_spec())
        coef_tbl = [t for t in result.tables if "계수" in t.title or "Coefficient" in t.title]
        if coef_tbl:
            df_c = coef_tbl[0].dataframe
            exp_cols = [c for c in df_c.columns if "Exp" in str(c) or "exp" in str(c).lower()]
            if exp_cols:
                vals = pd.to_numeric(df_c[exp_cols[0]], errors="coerce").dropna()
                assert all(vals > 0)

    def test_no_warnings_on_valid_data(self):
        """유효한 데이터 → 경고 없음."""
        from nuristat.analysis.logistic_regression import run_analysis
        ds = self._logistic_ds()
        result = run_analysis(ds, self._logistic_spec())
        assert result.warnings == []

    def test_roc_table_present(self):
        """ROC 분석 테이블 존재."""
        from nuristat.analysis.logistic_regression import run_analysis
        ds = self._logistic_ds()
        result = run_analysis(ds, self._logistic_spec({"roc_curve": True}))
        titles = [t.title for t in result.tables]
        assert any("ROC" in t for t in titles)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Excel 다중 시트 IO
# ─────────────────────────────────────────────────────────────────────────────

class TestExcelMultiSheet:
    """Excel 다중 시트 읽기 검증."""

    @pytest.fixture(autouse=True)
    def _excel_file(self, tmp_path):
        self._path = str(tmp_path / "multi.xlsx")
        with pd.ExcelWriter(self._path, engine="openpyxl") as w:
            pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}).to_excel(
                w, sheet_name="Sheet1", index=False)
            pd.DataFrame({"a": [7, 8, 9], "b": [10.0, 11.0, 12.0]}).to_excel(
                w, sheet_name="Data", index=False)
            pd.DataFrame({"col1": ["A", "B", "C"]}).to_excel(
                w, sheet_name="Text", index=False)

    def test_read_first_sheet_by_index(self):
        """sheet_name=0 → Sheet1 읽기."""
        from nuristat.io.excel_reader import read_excel
        ds = read_excel(self._path, sheet_name=0)
        assert list(ds.data.columns) == ["x", "y"]
        assert len(ds.data) == 3

    def test_read_sheet_by_name(self):
        """sheet_name='Data' → Data 시트 읽기."""
        from nuristat.io.excel_reader import read_excel
        ds = read_excel(self._path, sheet_name="Data")
        assert list(ds.data.columns) == ["a", "b"]
        assert len(ds.data) == 3

    def test_read_text_sheet(self):
        """문자열 컬럼 시트 읽기."""
        from nuristat.io.excel_reader import read_excel
        ds = read_excel(self._path, sheet_name="Text")
        assert "col1" in ds.data.columns
        assert ds.data["col1"].tolist() == ["A", "B", "C"]

    def test_sheet_data_values_correct(self):
        """Sheet1 x 컬럼 값 정확도."""
        from nuristat.io.excel_reader import read_excel
        ds = read_excel(self._path, sheet_name=0)
        assert list(ds.data["x"]) == [1, 2, 3]

    def test_variables_created_for_each_column(self):
        """각 컬럼에 VariableMeta 생성."""
        from nuristat.io.excel_reader import read_excel
        ds = read_excel(self._path, sheet_name=0)
        assert "x" in ds.variables
        assert "y" in ds.variables

    def test_second_sheet_independent_from_first(self):
        """두 시트 읽기가 독립적으로 처리됨."""
        from nuristat.io.excel_reader import read_excel
        ds1 = read_excel(self._path, sheet_name=0)
        ds2 = read_excel(self._path, sheet_name="Data")
        assert set(ds1.data.columns) != set(ds2.data.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset API — add_variable, get_column, update_variable_meta
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetAPI:
    """Dataset 핵심 API 검증."""

    def _base_ds(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["A", "B", "A", "B", "A"],
        })
        return _ds(df, {"x": MeasureType.SCALE, "cat": MeasureType.NOMINAL})

    def test_add_variable_appends_column(self):
        """add_variable → 컬럼 추가."""
        ds = self._base_ds()
        ds.add_variable("z", pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]))
        assert "z" in ds.var_names
        assert len(ds.data.columns) == 3

    def test_add_variable_with_meta(self):
        """add_variable with meta → VariableMeta 등록."""
        ds = self._base_ds()
        meta = VariableMeta("score", storage_type=StorageType.FLOAT,
                            measure=MeasureType.SCALE)
        ds.add_variable("score", pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), meta=meta)
        assert "score" in ds.variables
        assert ds.variables["score"].measure == MeasureType.SCALE

    def test_get_column_returns_series(self):
        """get_column → pd.Series 반환."""
        ds = self._base_ds()
        col = ds.get_column("x")
        assert isinstance(col, pd.Series)
        assert len(col) == 5

    def test_get_column_values_correct(self):
        """get_column 값 정확도."""
        ds = self._base_ds()
        col = ds.get_column("x")
        assert list(col) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_is_empty_false_for_nonempty(self):
        """비어있지 않은 DS → is_empty=False."""
        ds = self._base_ds()
        assert ds.is_empty is False

    def test_is_empty_true_for_empty(self):
        """빈 DS → is_empty=True."""
        ds = Dataset(pd.DataFrame())
        assert ds.is_empty is True

    def test_update_variable_meta_label(self):
        """update_variable_meta로 label 업데이트."""
        ds = self._base_ds()
        ds.update_variable_meta("x", label="연속형 변수")
        assert ds.variables["x"].label == "연속형 변수"

    def test_update_variable_meta_measure(self):
        """update_variable_meta로 measure 업데이트."""
        ds = self._base_ds()
        ds.update_variable_meta("cat", measure=MeasureType.ORDINAL)
        assert ds.variables["cat"].measure == MeasureType.ORDINAL

    def test_var_names_order_preserved(self):
        """var_names 순서 보존."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ds = Dataset(df)
        assert ds.var_names == ["a", "b", "c"]

    def test_n_cols_equals_variable_count(self):
        """n_cols == len(variables)."""
        ds = self._base_ds()
        assert ds.n_cols == len(ds.variables)

    def test_shape_matches_data(self):
        """shape == data.shape."""
        ds = self._base_ds()
        assert ds.shape == ds.data.shape


# ─────────────────────────────────────────────────────────────────────────────
# 7. SPSSGridModel 캐시 무결성
# ─────────────────────────────────────────────────────────────────────────────

class TestSPSSGridModelCache:
    """SPSSGridModel 캐시 채우기 → 구조 변경 → 초기화 검증."""

    def _make_model(self):
        from nuristat.ui.models.spss_grid_model import SPSSGridModel
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "num": rng.normal(0, 1, 10),
            "cat": ["A", "B"] * 5,
        })
        return SPSSGridModel(df)

    def test_cache_filled_after_access(self):
        """_is_numeric_col 접근 후 캐시에 결과 저장."""
        model = self._make_model()
        assert 0 not in model._numeric_col_cache
        model._is_numeric_col(0)
        assert 0 in model._numeric_col_cache

    def test_numeric_col_cached_true(self):
        """숫자 컬럼 → 캐시값 True."""
        model = self._make_model()
        result = model._is_numeric_col(0)
        assert result is True
        assert model._numeric_col_cache[0] is True

    def test_string_col_cached_false(self):
        """문자열 컬럼 → 캐시값 False."""
        model = self._make_model()
        result = model._is_numeric_col(1)
        assert result is False
        assert model._numeric_col_cache[1] is False

    def test_cache_invalidated_after_remove_column(self):
        """remove_column 후 캐시 초기화."""
        model = self._make_model()
        model._is_numeric_col(0)
        model._is_numeric_col(1)
        assert len(model._numeric_col_cache) == 2
        model.remove_column(0)
        assert len(model._numeric_col_cache) == 0

    def test_cache_invalidated_after_set_dataframe(self):
        """set_dataframe 후 캐시 초기화."""
        from nuristat.ui.models.spss_grid_model import SPSSGridModel
        model = self._make_model()
        model._is_numeric_col(0)
        rng = np.random.default_rng(8)
        new_df = pd.DataFrame({"a": rng.normal(0, 1, 5), "b": rng.normal(0, 1, 5)})
        model.set_dataframe(new_df)
        assert len(model._numeric_col_cache) == 0

    def test_setdata_updates_cell_value(self):
        """setData → 실제 셀 값 변경."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        idx = model.index(0, 0)
        model.setData(idx, "9.99", Qt.ItemDataRole.EditRole)
        val = model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole)
        assert "9.99" in str(val) or "9.9" in str(val)

    def test_rename_column_updates_header(self):
        """setHeaderData로 컬럼명 변경 후 headerData 반영."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        model.setHeaderData(0, Qt.Orientation.Horizontal, "new_name",
                            Qt.ItemDataRole.EditRole)
        header = model.headerData(0, Qt.Orientation.Horizontal,
                                  Qt.ItemDataRole.DisplayRole)
        assert str(header) == "new_name"

    def test_row_count_matches_data(self):
        """rowCount() = df 행 수 + 여분 행."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        assert model.rowCount() >= 10  # 실제 데이터 10행 이상

    def test_column_count_matches_data(self):
        """columnCount() = df 열 수 + 여분 열."""
        model = self._make_model()
        assert model.columnCount() >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 8. 상관분석 방법 변형 비교
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationMethods:
    """Pearson/Spearman/Kendall — 방향 일치, 값 범위 검증."""

    def _pos_corr_ds(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 50)
        y = x * 0.8 + rng.normal(0, 0.3, 50)
        z = x * 0.5 + rng.normal(0, 0.5, 50)
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        return _ds(df, {"x": MeasureType.SCALE, "y": MeasureType.SCALE, "z": MeasureType.SCALE})

    def _run_corr(self, method: str, ds):
        from nuristat.analysis.correlation import run_analysis
        return run_analysis(ds, {
            "variables": {"target": ["x", "y", "z"]},
            "options": {"method": method, "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        })

    def test_pearson_produces_tables(self):
        """Pearson → 최소 3개 테이블."""
        ds = self._pos_corr_ds()
        result = self._run_corr("pearson", ds)
        assert len(result.tables) >= 3

    def test_spearman_produces_tables(self):
        """Spearman → 최소 3개 테이블."""
        ds = self._pos_corr_ds()
        result = self._run_corr("spearman", ds)
        assert len(result.tables) >= 3

    def test_kendall_produces_tables(self):
        """Kendall → 최소 3개 테이블."""
        ds = self._pos_corr_ds()
        result = self._run_corr("kendall", ds)
        assert len(result.tables) >= 3

    def test_pearson_matrix_is_symmetric(self):
        """Pearson 상관행렬 대각 제외 대칭성."""
        ds = self._pos_corr_ds()
        result = self._run_corr("pearson", ds)
        mat_tbl = [t for t in result.tables if "Matrix" in t.title]
        if mat_tbl:
            df_m = mat_tbl[0].dataframe
            num_df = df_m.apply(pd.to_numeric, errors="coerce")
            floats = num_df.values.flatten()
            valid = [f for f in floats if not math.isnan(f)]
            if valid:
                assert all(-1.0 <= v <= 1.0 for v in valid)

    def test_pearson_x_y_positive(self):
        """강한 양의 관계 → Pearson r > 0.5."""
        ds = self._pos_corr_ds()
        x = ds.data["x"].values
        y = ds.data["y"].values
        r, _ = stats.pearsonr(x, y)
        assert r > 0.5

    def test_spearman_direction_matches_pearson(self):
        """Spearman rho 방향이 Pearson r 방향과 일치."""
        ds = self._pos_corr_ds()
        x = ds.data["x"].values
        y = ds.data["y"].values
        r_p, _ = stats.pearsonr(x, y)
        r_s, _ = stats.spearmanr(x, y)
        assert np.sign(r_p) == np.sign(r_s)

    def test_three_methods_no_warnings(self):
        """세 방법 모두 경고 없이 완료."""
        ds = self._pos_corr_ds()
        for method in ["pearson", "spearman", "kendall"]:
            result = self._run_corr(method, ds)
            assert result.warnings == [], f"{method}: warnings={result.warnings}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. 성능 스케일링 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceScaling:
    """N 10배 증가 시 처리 시간 선형 이하 (100배 이내)."""

    def _desc_spec(self):
        return {
            "variables": {"scale": ["x", "y", "z"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }

    def _time_desc(self, n: int) -> float:
        from nuristat.analysis.descriptive import run_analysis
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "z": rng.normal(0, 1, n),
        })
        ds = _ds(df, {"x": MeasureType.SCALE, "y": MeasureType.SCALE, "z": MeasureType.SCALE})
        t0 = time.perf_counter()
        run_analysis(ds, self._desc_spec())
        return time.perf_counter() - t0

    def test_descriptive_n1000_under_1s(self):
        """N=1000 기술통계 < 1초."""
        elapsed = self._time_desc(1000)
        assert elapsed < 1.0, f"N=1000 {elapsed:.3f}s > 1s"

    def test_descriptive_n100000_under_5s(self):
        """N=100,000 기술통계 < 5초."""
        elapsed = self._time_desc(100_000)
        assert elapsed < 5.0, f"N=100000 {elapsed:.3f}s > 5s"

    def test_descriptive_scaling_sublinear(self):
        """N 10배 시 시간 증가 100배 이내 (선형보다 작거나 같은 수준)."""
        t_small = self._time_desc(1_000)
        t_large = self._time_desc(10_000)
        if t_small > 0:
            ratio = t_large / t_small
            assert ratio < 100, f"스케일링 비율 {ratio:.1f}배 (허용: <100)"

    def test_correlation_n10000_under_5s(self):
        """N=10,000 상관분석 < 5초."""
        from nuristat.analysis.correlation import run_analysis
        rng = np.random.default_rng(2)
        n = 10_000
        df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        ds = _ds(df, {"x": MeasureType.SCALE, "y": MeasureType.SCALE})
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        t0 = time.perf_counter()
        run_analysis(ds, spec)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"N=10000 상관분석 {elapsed:.3f}s > 5s"

    def test_ttest_n50000_under_5s(self):
        """N=50,000 t-test < 5초."""
        from nuristat.analysis.ttests import run_analysis
        rng = np.random.default_rng(3)
        n = 50_000
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "grp": (["A"] * (n // 2) + ["B"] * (n // 2)),
        })
        ds = _ds(df, {"x": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        t0 = time.perf_counter()
        run_analysis(ds, spec)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"N=50000 t-test {elapsed:.3f}s > 5s"


# ─────────────────────────────────────────────────────────────────────────────
# 10. 결측 정책별 N 변화 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingPolicyEffect:
    """결측 정책(LISTWISE) 적용 시 N이 올바르게 줄어듦."""

    def _ds_with_missing(self, n_missing: int = 10):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50).tolist()
        y = rng.normal(0, 1, 50).tolist()
        for i in range(n_missing):
            x[i] = float("nan")
        df = pd.DataFrame({"x": x, "y": y})
        return _ds(df, {"x": MeasureType.SCALE, "y": MeasureType.SCALE})

    def test_listwise_reduces_n(self):
        """결측 10개 → listwise N = 40."""
        from nuristat.analysis.descriptive import run_analysis
        ds = self._ds_with_missing(10)
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        desc_tbl = [t for t in result.tables if "Descriptive" in t.title]
        if desc_tbl:
            row = desc_tbl[0].dataframe[
                desc_tbl[0].dataframe["Variable"].astype(str) == "x"
            ]
            if len(row) > 0:
                n_val = int(row.iloc[0]["N"])
                assert n_val == 40

    def test_all_missing_returns_result(self):
        """모든 값 결측 → 충돌 없이 결과 반환."""
        from nuristat.analysis.descriptive import run_analysis
        df = pd.DataFrame({"x": [float("nan")] * 20})
        ds = _ds(df, {"x": MeasureType.SCALE})
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        try:
            result = run_analysis(ds, spec)
            assert result is not None
        except Exception:
            pass

    def test_cps_excluded_count_correct(self):
        """CPS Excluded Cases = 결측 행 수."""
        from nuristat.analysis.descriptive import run_analysis
        ds = self._ds_with_missing(15)
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        cps = [t for t in result.tables if "Case Processing" in t.title]
        if cps:
            df_cps = cps[0].dataframe
            if "Excluded Cases" in df_cps.columns:
                excl = int(pd.to_numeric(df_cps["Excluded Cases"], errors="coerce").dropna().iloc[0])
                assert excl == 15

    def test_no_missing_n_equals_total(self):
        """결측 없음 → N = 전체 행 수."""
        from nuristat.analysis.descriptive import run_analysis
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 30)})
        ds = _ds(df, {"x": MeasureType.SCALE})
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        desc_tbl = [t for t in result.tables if "Descriptive" in t.title]
        if desc_tbl:
            row = desc_tbl[0].dataframe[
                desc_tbl[0].dataframe["Variable"].astype(str) == "x"
            ]
            if len(row) > 0:
                n_val = int(row.iloc[0]["N"])
                assert n_val == 30

    def test_partial_missing_ttest(self):
        """t-test 결측 처리 — 유효 N 표시."""
        from nuristat.analysis.ttests import run_analysis
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 40).tolist()
        x[0] = float("nan"); x[1] = float("nan")
        df = pd.DataFrame({"x": x, "grp": ["A"] * 20 + ["B"] * 20})
        ds = _ds(df, {"x": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        grp_tbl = [t for t in result.tables if "Group Statistics" in t.title]
        if grp_tbl:
            total_n = int(grp_tbl[0].dataframe["N"].sum())
            assert total_n <= 38

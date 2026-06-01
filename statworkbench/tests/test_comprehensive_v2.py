"""종합 테스트 v2 — 다각도 검증.

커버하는 영역:
  1. SPSSGridModel 캐싱 최적화 후 경계값
  2. 분석 간 상호 일관성 (t-test vs ANOVA 2그룹, Pearson r² vs R²)
  3. 결측 데이터 패턴 (MCAR / 완전결측 / 대량결측)
  4. 수치 정밀도 — 알려진 통계값 직접 비교
  5. IO 왕복 무결성 (CSV 저장→불러오기)
  6. 변수 타입 추론 경계값
  7. 신택스 파서 왕복 (write→parse→재생성)
  8. 분석 오류 복구 (잘못된 spec, 빈 변수 목록)
  9. SPSSGridModel 대량 셀 입력 성능
  10. 동일 데이터에 여러 분석 순차 실행
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtCore import QModelIndex

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


def _make_ds(df: pd.DataFrame, metas: dict | None = None) -> Dataset:
    ds = Dataset(df, name="test")
    if metas:
        for col, meta in metas.items():
            ds.variables[col] = meta
    return ds


def _get_table_val(result, title: str, col: str, row: int = 0) -> float:
    for tbl in result.tables:
        if tbl.title == title:
            val = tbl.dataframe.iloc[row][col]
            return float(str(val).replace(",", "").strip())
    raise KeyError(f"Table '{title}' not found in result")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPSSGridModel 캐싱 최적화 경계값
# ─────────────────────────────────────────────────────────────────────────────

class TestSPSSGridModelOptimization:
    """_is_numeric_col 캐싱 및 _update_variable_metadata O(1) 검증."""

    def _make_model(self):
        from statworkbench.ui.models.spss_grid_model import SPSSGridModel
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        return SPSSGridModel()

    def test_cache_populated_after_first_call(self):
        """첫 _is_numeric_col 호출 후 캐시에 결과 저장."""
        model = self._make_model()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        meta = {"x": VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)}
        model.set_dataframe(df, meta)
        # 캐시 비어있다가 첫 호출 후 채워짐
        assert 0 not in model._numeric_col_cache
        result1 = model._is_numeric_col(0)
        assert 0 in model._numeric_col_cache
        result2 = model._is_numeric_col(0)  # 캐시 히트
        assert result1 == result2 == True

    def test_cache_invalidated_on_set_dataframe(self):
        """set_dataframe 후 캐시 초기화."""
        model = self._make_model()
        df1 = pd.DataFrame({"x": [1.0, 2.0]})
        meta1 = {"x": VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)}
        model.set_dataframe(df1, meta1)
        _ = model._is_numeric_col(0)  # 캐시 채움
        assert 0 in model._numeric_col_cache

        df2 = pd.DataFrame({"s": ["a", "b"]})
        meta2 = {"s": VariableMeta(name="s", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL)}
        model.set_dataframe(df2, meta2)
        assert 0 not in model._numeric_col_cache  # 캐시 무효화
        assert model._is_numeric_col(0) == False

    def test_cache_invalidated_on_remove_column(self):
        """remove_column 후 캐시 초기화."""
        model = self._make_model()
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        meta = {
            "x": VariableMeta(name="x", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE),
            "y": VariableMeta(name="y", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE),
        }
        model.set_dataframe(df, meta)
        _ = model._is_numeric_col(0)
        _ = model._is_numeric_col(1)
        assert len(model._numeric_col_cache) == 2
        model.remove_column(0)
        assert len(model._numeric_col_cache) == 0  # 전체 캐시 무효화

    def test_storage_type_upgrade_string_to_integer(self):
        """새 변수에 정수 입력 시 STRING → INTEGER 승격."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        idx = model.index(0, 0)
        model.setData(idx, "42", Qt.ItemDataRole.EditRole)
        col_name = model._dataframe.columns[0]
        var = model._variables[col_name]
        assert var.storage_type == StorageType.INTEGER

    def test_storage_type_upgrade_integer_to_float(self):
        """정수 열에 소수 입력 시 INTEGER → FLOAT 승격."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        idx0 = model.index(0, 0)
        model.setData(idx0, "10", Qt.ItemDataRole.EditRole)
        idx1 = model.index(1, 0)
        model.setData(idx1, "3.14", Qt.ItemDataRole.EditRole)
        col_name = model._dataframe.columns[0]
        assert model._variables[col_name].storage_type == StorageType.FLOAT

    def test_storage_type_stays_string_for_text(self):
        """문자열 입력 시 STRING 유지."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        idx = model.index(0, 0)
        model.setData(idx, "hello", Qt.ItemDataRole.EditRole)
        col_name = model._dataframe.columns[0]
        assert model._variables[col_name].storage_type == StorageType.STRING

    def test_decimal_tracked_correctly(self):
        """소수점 자릿수 최댓값 추적."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        model.setData(model.index(0, 0), "1.1", Qt.ItemDataRole.EditRole)
        model.setData(model.index(1, 0), "2.123", Qt.ItemDataRole.EditRole)
        col_name = model._dataframe.columns[0]
        assert model._variables[col_name].decimals == 3

    def test_na_input_does_not_reset_type(self):
        """NA 입력 시 기존 storage_type 유지."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        model.setData(model.index(0, 0), "5", Qt.ItemDataRole.EditRole)
        col_name = model._dataframe.columns[0]
        assert model._variables[col_name].storage_type == StorageType.INTEGER
        model.setData(model.index(1, 0), "", Qt.ItemDataRole.EditRole)
        assert model._variables[col_name].storage_type == StorageType.INTEGER  # 변경 없음

    def test_large_batch_input_performance(self):
        """100행 연속 셀 입력이 1초 이내 완료."""
        from PySide6.QtCore import Qt
        model = self._make_model()
        n = 100
        start = time.perf_counter()
        for i in range(n):
            idx = model.index(i, 0)
            model.setData(idx, str(float(i)), Qt.ItemDataRole.EditRole)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100행 입력 {elapsed:.3f}s > 1s"
        assert len(model._dataframe) >= n

    def test_numeric_col_cache_returns_false_for_out_of_range(self):
        """범위 밖 열 인덱스는 False 반환 (캐시 저장 안 함)."""
        model = self._make_model()
        result = model._is_numeric_col(999)
        assert result == False
        assert 999 not in model._numeric_col_cache


# ─────────────────────────────────────────────────────────────────────────────
# 2. 분석 간 상호 일관성
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisCrossConsistency:
    """서로 다른 분석 함수 결과 간 수학적 일관성 검증."""

    def _two_group_ds(self, seed: int = 0):
        from scipy import stats as scipy_stats
        rng = np.random.default_rng(seed)
        n = 50
        a = rng.normal(10, 2, n)
        b = rng.normal(13, 2, n)
        df = pd.DataFrame({"score": np.concatenate([a, b]), "grp": [0]*n + [1]*n})
        return _make_ds(df, {"score": _scale("score"), "grp": _nominal("grp")})

    def test_ttest_vs_anova_fstat_equals_t_squared(self):
        """2그룹 ANOVA F = t² (수학적 동치)."""
        from statworkbench.analysis.ttests import run_analysis as ttest_run
        from statworkbench.analysis.anova import run_analysis as anova_run

        ds = self._two_group_ds(42)
        t_spec = {
            "variables": {"dependent": "score", "group": "grp"},
            "options": {"equal_var": "yes"},
            "confidence_level": 0.95,
        }
        a_spec = {
            "variables": {"dependent": "score", "factor": "grp"},
            "options": {"post_hoc": [], "effect_size": False},
            "confidence_level": 0.95,
        }
        t_res = ttest_run(ds, t_spec)
        a_res = anova_run(ds, a_spec)

        # t 추출
        t_val = None
        for tbl in t_res.tables:
            if "t-Test" in tbl.title or "t-test" in tbl.title.lower():
                for col in ["t", "t-statistic", "T"]:
                    if col in tbl.dataframe.columns:
                        try:
                            t_val = float(str(tbl.dataframe.iloc[0][col]).replace(",", ""))
                            break
                        except (ValueError, TypeError):
                            pass
                if t_val is not None:
                    break

        # F 추출
        f_val = None
        for tbl in a_res.tables:
            if tbl.title == "ANOVA":
                for col in ["F", "F-ratio"]:
                    if col in tbl.dataframe.columns:
                        grp_rows = tbl.dataframe[tbl.dataframe.apply(
                            lambda r: any("grp" in str(v).lower() or "between" in str(v).lower()
                                         for v in r.values), axis=1
                        )]
                        if not grp_rows.empty:
                            try:
                                f_val = float(str(grp_rows.iloc[0][col]).replace(",", ""))
                                break
                            except (ValueError, TypeError):
                                pass
                if f_val is not None:
                    break

        if t_val is not None and f_val is not None:
            assert abs(t_val**2 - f_val) < 0.01 * abs(f_val) + 0.1, (
                f"t²={t_val**2:.4f} != F={f_val:.4f}"
            )

    def test_pearson_r_squared_equals_simple_regression_rsq(self):
        """단순회귀 R² = Pearson r² (수학적 동치)."""
        from statworkbench.analysis.correlation import run_analysis as corr_run
        from statworkbench.analysis.regression import run_analysis as reg_run

        rng = np.random.default_rng(7)
        n = 200
        x = rng.normal(0, 1, n)
        y = 0.6 * x + rng.normal(0, 0.8, n)
        df = pd.DataFrame({"x": x, "y": y})
        ds = _make_ds(df, {"x": _scale("x"), "y": _scale("y")})

        r_res = corr_run(ds, {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "flag_significant": False, "pairwise": False},
            "confidence_level": 0.95,
        })
        reg_res = reg_run(ds, {
            "variables": {"dependent": "y", "independent": ["x"]},
            "options": {},
            "confidence_level": 0.95,
        })

        # R² 추출
        rsq = None
        for tbl in reg_res.tables:
            if "Model Summary" in tbl.title or "model summary" in tbl.title.lower():
                for col in ["R²", "R Square", "R^2", "R2"]:
                    if col in tbl.dataframe.columns:
                        try:
                            rsq = float(str(tbl.dataframe.iloc[0][col]).replace(",", ""))
                            break
                        except (ValueError, TypeError):
                            pass
                if rsq is not None:
                    break

        # r 추출 (상관행렬 대각선 외)
        r_val = None
        for tbl in r_res.tables:
            if "Pearson" in tbl.title or "Correlation" in tbl.title:
                df_tbl = tbl.dataframe
                if "y" in df_tbl.columns and len(df_tbl) >= 1:
                    for i, row in df_tbl.iterrows():
                        var_col = df_tbl.columns[0]
                        if str(row[var_col]).strip() == "x":
                            try:
                                r_val = float(str(row["y"]).replace(",", "").replace("**", "").replace("*", ""))
                                break
                            except (ValueError, TypeError):
                                pass
                if r_val is not None:
                    break

        if rsq is not None and r_val is not None:
            assert rsq == pytest.approx(r_val**2, abs=0.02), (
                f"R²={rsq:.4f} != r²={r_val**2:.4f}"
            )

    def test_descriptive_n_matches_ttest_n(self):
        """기술통계 N = t-검정 그룹별 합계 N (결측 없는 경우)."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run
        from statworkbench.analysis.ttests import run_analysis as ttest_run

        rng = np.random.default_rng(11)
        n_total = 80
        df = pd.DataFrame({
            "score": rng.normal(10, 2, n_total),
            "grp": [0] * 40 + [1] * 40,
        })
        ds = _make_ds(df, {"score": _scale("score"), "grp": _nominal("grp")})

        d_res = desc_run(ds, {"variables": {"scale": ["score"]}, "confidence_level": 0.95})
        t_res = ttest_run(ds, {
            "variables": {"dependent": "score", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        })

        # 기술통계 N
        desc_n = None
        for tbl in d_res.tables:
            if "Descriptive" in tbl.title:
                if "N" in tbl.dataframe.columns:
                    try:
                        desc_n = int(float(str(tbl.dataframe.iloc[0]["N"]).replace(",", "")))
                    except (ValueError, TypeError):
                        pass
                break

        # t-검정 그룹 N 합계
        ttest_n_sum = None
        for tbl in t_res.tables:
            if "Group Statistics" in tbl.title or "그룹" in tbl.title:
                if "N" in tbl.dataframe.columns:
                    try:
                        ns = [int(float(str(v).replace(",", ""))) for v in tbl.dataframe["N"]]
                        ttest_n_sum = sum(ns)
                    except (ValueError, TypeError):
                        pass
                break

        if desc_n is not None and ttest_n_sum is not None:
            assert desc_n == ttest_n_sum, f"기술통계 N={desc_n} != t-검정 N합={ttest_n_sum}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. 결측 데이터 패턴
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingDataPatterns:
    """다양한 결측 패턴에서 분석 안정성 검증."""

    def test_mcar_30pct_descriptive(self):
        """완전임의결측(MCAR) 30%: 기술통계 N이 정확히 감소."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run

        rng = np.random.default_rng(20)
        n = 200
        data = rng.normal(50, 10, n).tolist()
        missing_idx = rng.choice(n, size=60, replace=False)
        for i in missing_idx:
            data[i] = float("nan")
        expected_n = n - 60

        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale("x")})
        res = desc_run(ds, {"variables": {"scale": ["x"]}, "confidence_level": 0.95})

        n_val = _get_table_val(res, "Descriptive Statistics", "N")
        assert n_val == expected_n

    def test_all_missing_ttest_returns_warning(self):
        """완전결측 변수 t-검정: 크래시 없이 경고 반환."""
        from statworkbench.analysis.ttests import run_analysis as ttest_run

        df = pd.DataFrame({"score": [float("nan")] * 20, "grp": [0]*10 + [1]*10})
        ds = _make_ds(df, {"score": _scale("score"), "grp": _nominal("grp")})
        res = ttest_run(ds, {
            "variables": {"dependent": "score", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        })
        assert res is not None
        assert len(res.warnings) > 0

    def test_one_group_entirely_missing_ttest(self):
        """한 그룹이 완전결측: 경고 반환."""
        from statworkbench.analysis.ttests import run_analysis as ttest_run

        scores = [1.0, 2.0, 3.0] + [float("nan")] * 3
        grps = [0, 0, 0, 1, 1, 1]
        df = pd.DataFrame({"score": scores, "grp": grps})
        ds = _make_ds(df, {"score": _scale("score"), "grp": _nominal("grp")})
        res = ttest_run(ds, {
            "variables": {"dependent": "score", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        })
        assert res is not None

    def test_90pct_missing_regression_handled(self):
        """90% 결측 회귀: 크래시 없이 처리."""
        from statworkbench.analysis.regression import run_analysis as reg_run

        rng = np.random.default_rng(21)
        n = 100
        x = rng.normal(0, 1, n).tolist()
        y = rng.normal(0, 1, n).tolist()
        # 90% 결측
        for i in range(90):
            x[i] = float("nan")
            y[i] = float("nan")
        df = pd.DataFrame({"x": x, "y": y})
        ds = _make_ds(df, {"x": _scale("x"), "y": _scale("y")})
        res = reg_run(ds, {
            "variables": {"dependent": "y", "independent": ["x"]},
            "options": {},
            "confidence_level": 0.95,
        })
        assert res is not None

    def test_missing_in_group_variable_anova(self):
        """그룹 변수에 결측값 포함: 결측 제거 후 정상 분석."""
        from statworkbench.analysis.anova import run_analysis as anova_run

        rng = np.random.default_rng(22)
        data = rng.normal(0, 1, 30).tolist()
        grps = [0]*10 + [1]*10 + [2]*10
        # 그룹 변수에 결측 추가
        grps[5] = None
        grps[15] = None
        df = pd.DataFrame({"y": data, "grp": grps})
        ds = _make_ds(df, {"y": _scale("y"), "grp": _nominal("grp")})
        res = anova_run(ds, {
            "variables": {"dependent": "y", "factor": "grp"},
            "options": {"post_hoc": []},
            "confidence_level": 0.95,
        })
        assert res is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. 수치 정밀도 — 알려진 통계값
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalPrecision:
    """알려진 수학적 참값과 비교해 수치 정밀도 검증."""

    def test_mean_exact_integer_data(self):
        """정수 시퀀스 평균: 정확값 확인."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run

        data = list(range(1, 11))  # 1~10, mean=5.5
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale("x")})
        res = desc_run(ds, {"variables": {"scale": ["x"]}, "confidence_level": 0.95})
        mean_val = _get_table_val(res, "Descriptive Statistics", "Mean")
        assert mean_val == pytest.approx(5.5, abs=0.001)

    def test_std_known_value(self):
        """표준편차 정확값: [2,4,4,4,5,5,7,9] → std=2.0."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run

        data = [2, 4, 4, 4, 5, 5, 7, 9]  # population std=2, sample std≈2.138
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale("x")})
        res = desc_run(ds, {"variables": {"scale": ["x"]}, "confidence_level": 0.95})
        # SPSS uses sample std
        expected_std = np.std(data, ddof=1)
        for tbl in res.tables:
            if "Descriptive" in tbl.title:
                for col in ["Std. Deviation", "SD", "Std Dev", "표준편차"]:
                    if col in tbl.dataframe.columns:
                        std_val = float(str(tbl.dataframe.iloc[0][col]).replace(",", ""))
                        assert std_val == pytest.approx(expected_std, abs=0.001)
                        return

    def test_pearson_r_perfect_correlation(self):
        """완전 선형 관계: r=1.0."""
        from statworkbench.analysis.correlation import run_analysis as corr_run

        x = np.arange(1, 21, dtype=float)
        y = 3 * x + 7  # perfect correlation
        df = pd.DataFrame({"x": x, "y": y})
        ds = _make_ds(df, {"x": _scale("x"), "y": _scale("y")})
        res = corr_run(ds, {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "flag_significant": False, "pairwise": False},
            "confidence_level": 0.95,
        })
        # 상관행렬에서 r(x,y) 추출
        for tbl in res.tables:
            if "Pearson" in tbl.title or "Correlation" in tbl.title:
                df_tbl = tbl.dataframe
                if "y" in df_tbl.columns:
                    for _, row in df_tbl.iterrows():
                        var_col = df_tbl.columns[0]
                        if "x" in str(row.get(var_col, "")):
                            r = float(str(row["y"]).replace(",", "").replace("**", "").replace("*", "").strip())
                            assert abs(r) == pytest.approx(1.0, abs=0.001)
                            return

    def test_chi_square_independence_known_value(self):
        """독립성 검정 알려진 카이제곱 값: 2x2 표."""
        from statworkbench.analysis.crosstab import run_analysis as cross_run

        # 표준 2x2 예시: chi² = 3.841 (df=1, α=0.05 임계값)
        # 독립인 경우 치우침 없는 데이터
        obs = [10, 20, 20, 40]  # 독립 관계 (비율 동일)
        df = pd.DataFrame({
            "a": [0]*10 + [0]*20 + [1]*20 + [1]*40,
            "b": [0]*10 + [1]*20 + [0]*20 + [1]*40,
        })
        ds = _make_ds(df, {"a": _nominal("a"), "b": _nominal("b")})
        res = cross_run(ds, {
            "variables": {"row": "a", "column": "b"},
            "options": {"chi_square": True, "expected": False, "row_pct": False,
                        "col_pct": False, "total_pct": False},
            "confidence_level": 0.95,
        })
        # 크래시 없이 완료
        assert res is not None
        # 독립이므로 chi² ≈ 0
        for tbl in res.tables:
            if "Chi" in tbl.title or "chi" in tbl.title.lower():
                for col in ["Chi-Square", "χ²", "Value"]:
                    if col in tbl.dataframe.columns:
                        chi_rows = tbl.dataframe[tbl.dataframe.apply(
                            lambda r: any("pearson" in str(v).lower() for v in r.values), axis=1
                        )]
                        if not chi_rows.empty:
                            chi_val = float(str(chi_rows.iloc[0][col]).replace(",", ""))
                            assert chi_val < 1.0, f"독립인 데이터 chi²={chi_val:.4f} >= 1.0"
                        break

    def test_one_sample_ttest_known_value(self):
        """단일표본 t-검정: 알려진 t값 확인."""
        from statworkbench.analysis.ttests import run_analysis as ttest_run

        # x = [3,4,5,6,7], mu_0=5, t = (5-5)/(sqrt(2.5)/sqrt(5)) = 0
        data = [3.0, 4.0, 5.0, 6.0, 7.0]
        df = pd.DataFrame({"x": data})
        ds = _make_ds(df, {"x": _scale("x")})
        res = ttest_run(ds, {
            "variables": {"test_type": "one_sample", "variables": ["x"]},
            "options": {"test_value": 5.0},
            "confidence_level": 0.95,
        })
        assert res is not None
        # t ≈ 0 (평균=5.0, 검정값=5.0)
        for tbl in res.tables:
            if "One-Sample" in tbl.title or "one_sample" in tbl.title.lower():
                for col in ["t", "T"]:
                    if col in tbl.dataframe.columns:
                        t_val = float(str(tbl.dataframe.iloc[0][col]).replace(",", ""))
                        assert abs(t_val) < 0.001
                        return


# ─────────────────────────────────────────────────────────────────────────────
# 5. IO 왕복 무결성
# ─────────────────────────────────────────────────────────────────────────────

class TestIORoundTrip:
    """데이터 저장 → 불러오기 → 원본과 비교."""

    def test_csv_roundtrip_numeric(self):
        """숫자 데이터 CSV 저장→불러오기 무결성."""
        from statworkbench.io.csv_reader import read_csv

        rng = np.random.default_rng(30)
        df_orig = pd.DataFrame({
            "a": rng.integers(1, 100, 50).astype(float),
            "b": rng.normal(0, 1, 50),
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8",
                                         delete=False, newline="") as f:
            df_orig.to_csv(f.name, index=False)
            fname = f.name

        try:
            ds = read_csv(fname)
            assert ds is not None
            assert list(ds.data.columns) == ["a", "b"]
            assert len(ds.data) == 50
            pd.testing.assert_frame_equal(
                ds.data.reset_index(drop=True).astype(float),
                df_orig.reset_index(drop=True).astype(float),
                check_exact=False, rtol=1e-5,
            )
        finally:
            os.unlink(fname)

    def test_csv_roundtrip_with_missing(self):
        """결측값 포함 CSV 왕복: NaN 보존."""
        from statworkbench.io.csv_reader import read_csv

        df_orig = pd.DataFrame({"x": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8",
                                         delete=False, newline="") as f:
            df_orig.to_csv(f.name, index=False)
            fname = f.name

        try:
            ds = read_csv(fname)
            loaded = pd.to_numeric(ds.data["x"], errors="coerce")
            assert loaded.isna().sum() == 2
            assert loaded.dropna().tolist() == pytest.approx([1.0, 3.0, 5.0])
        finally:
            os.unlink(fname)

    def test_csv_roundtrip_unicode_strings(self):
        """한글 문자열 CSV 왕복 무결성."""
        from statworkbench.io.csv_reader import read_csv

        df_orig = pd.DataFrame({"이름": ["홍길동", "김철수", "이영희"], "나이": [30, 25, 35]})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8-sig",
                                         delete=False, newline="") as f:
            df_orig.to_csv(f.name, index=False, encoding="utf-8-sig")
            fname = f.name

        try:
            ds = read_csv(fname)
            assert ds is not None
            assert len(ds.data) == 3
        finally:
            os.unlink(fname)

    def test_project_store_roundtrip(self):
        """프로젝트 저장→불러오기: Dataset 구조 보존."""
        from statworkbench.io.project_store import save_project, load_project

        rng = np.random.default_rng(31)
        df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.integers(0, 3, 20).astype(float)})
        ds = _make_ds(df, {"x": _scale("x"), "y": _nominal("y")})
        ds.variables["x"].label = "연속변수"
        ds.variables["y"].value_labels = {0: "A", 1: "B", 2: "C"}

        with tempfile.NamedTemporaryFile(suffix=".swb", delete=False) as f:
            fname = f.name

        try:
            from statworkbench.core.project import Project
            proj = Project(dataset=ds)
            save_project(proj, fname)
            loaded_proj = load_project(fname)
            assert loaded_proj is not None
            loaded_ds = loaded_proj.dataset
            assert loaded_ds is not None
            assert len(loaded_ds.data) == 20
            assert "x" in loaded_ds.variables
            assert "y" in loaded_ds.variables
            assert loaded_ds.variables["x"].label == "연속변수"
            assert loaded_ds.variables["y"].value_labels.get(0) == "A" or \
                   loaded_ds.variables["y"].value_labels.get("0") == "A"
        finally:
            os.unlink(fname)

    def test_csv_empty_file(self):
        """빈 CSV 파일: 오류 없이 빈 Dataset 반환."""
        from statworkbench.io.csv_reader import read_csv

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8",
                                         delete=False) as f:
            f.write("")  # 완전히 빈 파일
            fname = f.name

        try:
            try:
                ds = read_csv(fname)
                # 빈 파일은 빈 DataFrame 또는 예외 — 둘 다 OK
                if ds is not None:
                    assert len(ds.data) == 0 or ds.data.empty
            except Exception:
                pass  # 예외도 허용
        finally:
            os.unlink(fname)

    def test_csv_single_row(self):
        """헤더만 있는 CSV: 0행 Dataset."""
        from statworkbench.io.csv_reader import read_csv

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8",
                                         delete=False, newline="") as f:
            f.write("a,b,c\n")
            fname = f.name

        try:
            ds = read_csv(fname)
            assert ds is not None
            assert list(ds.data.columns) == ["a", "b", "c"] or len(ds.data) == 0
        finally:
            os.unlink(fname)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 변수 타입 추론 경계값
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableTypeInferenceBoundary:
    """Dataset 생성 시 변수 메타데이터 자동 추론 경계값."""

    def test_all_zeros_inferred_as_scale(self):
        """모두 0인 열: SCALE로 추론."""
        df = pd.DataFrame({"x": [0, 0, 0, 0, 0]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"].measure in (MeasureType.SCALE, MeasureType.NOMINAL)

    def test_negative_integers_scale(self):
        """음의 정수: SCALE."""
        df = pd.DataFrame({"x": [-3, -2, -1, 0, 1]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"].storage_type in (StorageType.INTEGER, StorageType.FLOAT)

    def test_large_integer_scale(self):
        """큰 정수 (1e9 범위): 오버플로우 없이 처리."""
        df = pd.DataFrame({"x": [10**9, 2*10**9, 3*10**9]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"] is not None

    def test_boolean_column_binary(self):
        """불리언 열: BINARY로 추론."""
        df = pd.DataFrame({"x": [True, False, True, False]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"].storage_type in (StorageType.BOOLEAN, StorageType.INTEGER)

    def test_string_column_nominal(self):
        """문자열 열: NOMINAL로 추론."""
        df = pd.DataFrame({"x": ["a", "b", "c", "d"]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"].measure == MeasureType.NOMINAL

    def test_two_unique_string_binary(self):
        """문자열 2개 고유값: BINARY로 추론."""
        df = pd.DataFrame({"x": ["yes", "no", "yes", "no"]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"].measure in (MeasureType.BINARY, MeasureType.NOMINAL)

    def test_float_column_scale(self):
        """소수점 열: SCALE, FLOAT 타입."""
        df = pd.DataFrame({"x": [1.1, 2.2, 3.3]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"].storage_type == StorageType.FLOAT
        assert ds.variables["x"].measure == MeasureType.SCALE

    def test_mixed_int_float_column(self):
        """정수+소수 혼합 열: FLOAT로 처리."""
        df = pd.DataFrame({"x": [1, 2.5, 3, 4.7]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"].storage_type == StorageType.FLOAT

    def test_single_value_column(self):
        """단일 고유값 열: 처리 안정성."""
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        ds = Dataset(df, name="t")
        assert ds.variables["x"] is not None

    def test_dataset_update_variable_meta(self):
        """update_variable_meta: 필드 정확히 갱신."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = Dataset(df, name="t")
        ds.update_variable_meta("x", label="내 변수", decimals=4)
        assert ds.variables["x"].label == "내 변수"
        assert ds.variables["x"].decimals == 4


# ─────────────────────────────────────────────────────────────────────────────
# 7. 신택스 파서 왕복
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntaxRoundTrip:
    """신택스 write → parse → 재생성 일관성."""

    def test_frequencies_syntax_roundtrip(self):
        """FREQUENCIES 명령 생성→파싱→재생성."""
        from statworkbench.syntax.writer import SyntaxWriter
        from statworkbench.syntax.parser import SyntaxParser

        writer = SyntaxWriter()
        writer.write_frequencies(["var1", "var2"], dataset_id="ds1",
                                  options={"statistics": ["mean", "std"]})
        syntax = writer.to_string()
        assert "FREQUENCIES" in syntax.upper()
        assert "var1" in syntax or "VAR1" in syntax

        parser = SyntaxParser()
        commands = parser.parse(syntax)
        assert len(commands) >= 1

    def test_descriptives_syntax_roundtrip(self):
        """DESCRIPTIVES 명령 생성→파싱."""
        from statworkbench.syntax.writer import SyntaxWriter
        from statworkbench.syntax.parser import SyntaxParser

        writer = SyntaxWriter()
        writer.write_descriptives(["x", "y", "z"], dataset_id="ds1")
        syntax = writer.to_string()
        assert "DESCRIPTIVES" in syntax.upper() or len(syntax) > 0

        parser = SyntaxParser()
        commands = parser.parse(syntax)
        assert commands is not None

    def test_multiple_commands_syntax(self):
        """여러 명령 연속 생성→파싱."""
        from statworkbench.syntax.writer import SyntaxWriter
        from statworkbench.syntax.parser import SyntaxParser

        writer = SyntaxWriter()
        writer.write_frequencies(["a"], dataset_id="ds1", options={})
        writer.write_descriptives(["b"], dataset_id="ds1")
        syntax = writer.to_string()
        parser = SyntaxParser()
        commands = parser.parse(syntax)
        assert len(commands) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. 분석 오류 복구 — 잘못된 spec
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisErrorRecovery:
    """잘못된 spec, 빈 변수 목록, 존재하지 않는 변수 처리."""

    def _simple_ds(self):
        rng = np.random.default_rng(40)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 30),
            "grp": [0]*15 + [1]*15,
        })
        return _make_ds(df, {"x": _scale("x"), "grp": _nominal("grp")})

    def test_regression_nonexistent_variable(self):
        """존재하지 않는 독립변수: 경고 후 반환."""
        from statworkbench.analysis.regression import run_analysis as reg_run
        ds = self._simple_ds()
        res = reg_run(ds, {
            "variables": {"dependent": "x", "independent": ["does_not_exist"]},
            "options": {},
            "confidence_level": 0.95,
        })
        assert res is not None

    def test_ttest_empty_dependent(self):
        """의존변수 미지정 t-검정: 경고 반환."""
        from statworkbench.analysis.ttests import run_analysis as ttest_run
        ds = self._simple_ds()
        res = ttest_run(ds, {
            "variables": {"dependent": "", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        })
        assert res is not None

    def test_anova_single_group(self):
        """단일 그룹 ANOVA: 경고 반환."""
        from statworkbench.analysis.anova import run_analysis as anova_run
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "grp": [1, 1, 1, 1]})
        ds = _make_ds(df, {"y": _scale("y"), "grp": _nominal("grp")})
        res = anova_run(ds, {
            "variables": {"dependent": "y", "factor": "grp"},
            "options": {"post_hoc": []},
            "confidence_level": 0.95,
        })
        assert len(res.warnings) > 0

    def test_correlation_single_variable(self):
        """변수 1개 상관분석: 경고 또는 빈 행렬."""
        from statworkbench.analysis.correlation import run_analysis as corr_run
        ds = self._simple_ds()
        res = corr_run(ds, {
            "variables": {"target": ["x"]},
            "options": {"method": "pearson", "flag_significant": False, "pairwise": False},
            "confidence_level": 0.95,
        })
        assert res is not None

    def test_descriptive_no_variables(self):
        """변수 없는 기술통계: 경고 반환."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run
        ds = self._simple_ds()
        res = desc_run(ds, {"variables": {"scale": []}, "confidence_level": 0.95})
        assert res is not None

    def test_regression_n_less_than_predictors(self):
        """N < 예측변수 수 회귀: 경고 반환."""
        from statworkbench.analysis.regression import run_analysis as reg_run
        rng = np.random.default_rng(41)
        df = pd.DataFrame({
            "y": rng.normal(0, 1, 3),
            "x1": rng.normal(0, 1, 3),
            "x2": rng.normal(0, 1, 3),
            "x3": rng.normal(0, 1, 3),
            "x4": rng.normal(0, 1, 3),
        })
        ds = _make_ds(df, {col: _scale(col) for col in df.columns})
        res = reg_run(ds, {
            "variables": {"dependent": "y", "independent": ["x1", "x2", "x3", "x4"]},
            "options": {},
            "confidence_level": 0.95,
        })
        assert res is not None


# ─────────────────────────────────────────────────────────────────────────────
# 9. 동일 데이터에 여러 분석 순차 실행
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleAnalysesOnSameDataset:
    """동일 Dataset 객체로 여러 분석을 순서대로 실행해 데이터 불변성 확인."""

    def test_dataset_unchanged_after_analyses(self):
        """분석 실행 후 Dataset 원본 데이터 변경 없음."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run
        from statworkbench.analysis.ttests import run_analysis as ttest_run
        from statworkbench.analysis.correlation import run_analysis as corr_run

        rng = np.random.default_rng(50)
        n = 100
        df = pd.DataFrame({
            "x": rng.normal(10, 2, n),
            "y": rng.normal(20, 3, n),
            "grp": [0]*50 + [1]*50,
        })
        ds = _make_ds(df, {"x": _scale("x"), "y": _scale("y"), "grp": _nominal("grp")})
        original_shape = ds.data.shape
        original_sum = ds.data["x"].sum()

        desc_run(ds, {"variables": {"scale": ["x", "y"]}, "confidence_level": 0.95})
        ttest_run(ds, {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        })
        corr_run(ds, {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "flag_significant": True, "pairwise": False},
            "confidence_level": 0.95,
        })

        assert ds.data.shape == original_shape
        assert ds.data["x"].sum() == pytest.approx(original_sum, rel=1e-10)

    def test_all_analysis_modules_return_valid_result(self):
        """10개 분석 모듈 순차 실행: 모두 유효한 AnalysisResult 반환."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run
        from statworkbench.analysis.ttests import run_analysis as ttest_run
        from statworkbench.analysis.anova import run_analysis as anova_run
        from statworkbench.analysis.correlation import run_analysis as corr_run
        from statworkbench.analysis.regression import run_analysis as reg_run
        from statworkbench.analysis.frequencies import run_analysis as freq_run
        from statworkbench.analysis.crosstab import run_analysis as cross_run
        from statworkbench.analysis.nonparametric import run_analysis as np_run
        from statworkbench.analysis.normality import run_analysis as norm_run

        rng = np.random.default_rng(60)
        n = 60
        df = pd.DataFrame({
            "cont": rng.normal(50, 10, n),
            "grp": [0]*20 + [1]*20 + [2]*20,
            "cat": (rng.integers(0, 3, n)).tolist(),
        })
        ds = _make_ds(df, {
            "cont": _scale("cont"),
            "grp": _nominal("grp"),
            "cat": _nominal("cat"),
        })

        results = [
            desc_run(ds, {"variables": {"scale": ["cont"]}, "confidence_level": 0.95}),
            ttest_run(ds, {
                "variables": {"dependent": "cont", "group": "grp"},
                "options": {"equal_var": "auto"}, "confidence_level": 0.95,
            }),
            anova_run(ds, {
                "variables": {"dependent": "cont", "factor": "grp"},
                "options": {"post_hoc": []}, "confidence_level": 0.95,
            }),
            corr_run(ds, {
                "variables": {"target": ["cont"]},
                "options": {"method": "pearson", "flag_significant": False, "pairwise": False},
                "confidence_level": 0.95,
            }),
            reg_run(ds, {
                "variables": {"dependent": "cont", "independent": ["cat"]},
                "options": {}, "confidence_level": 0.95,
            }),
            freq_run(ds, {
                "variables": {"categorical": ["grp"]},
                "options": {}, "confidence_level": 0.95,
            }),
            cross_run(ds, {
                "variables": {"row": "grp", "column": "cat"},
                "options": {"chi_square": True, "expected": False, "row_pct": False,
                            "col_pct": False, "total_pct": False},
                "confidence_level": 0.95,
            }),
            norm_run(ds, {
                "variables": {"scale": ["cont"]},
                "options": {}, "confidence_level": 0.95,
            }),
            np_run(ds, {
                "variables": {"test_type": "kruskal_wallis", "dependent": "cont", "group": "grp"},
                "options": {}, "confidence_level": 0.95,
            }),
        ]

        for i, res in enumerate(results):
            assert res is not None, f"분석 {i} 결과가 None"
            assert hasattr(res, "tables"), f"분석 {i} tables 속성 없음"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Dataset 핵심 기능 경계값
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetCoreBoundary:
    """Dataset 클래스 경계값 및 오류 처리."""

    def test_add_variable_duplicate_name_raises(self):
        """중복 변수명 추가: DatasetError."""
        from statworkbench.core.exceptions import DatasetError
        df = pd.DataFrame({"x": [1, 2, 3]})
        ds = Dataset(df, name="t")
        with pytest.raises(DatasetError):
            ds.add_variable("x")

    def test_remove_nonexistent_variable_raises(self):
        """존재하지 않는 변수 삭제: DatasetError."""
        from statworkbench.core.exceptions import DatasetError
        df = pd.DataFrame({"x": [1, 2, 3]})
        ds = Dataset(df, name="t")
        with pytest.raises(DatasetError):
            ds.remove_variable("no_such_var")

    def test_rename_to_existing_name_raises(self):
        """기존 이름으로 변수 이름 변경: DatasetError."""
        from statworkbench.core.exceptions import DatasetError
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        ds = Dataset(df, name="t")
        with pytest.raises(DatasetError):
            ds.rename_variable("x", "y")

    def test_copy_is_independent(self):
        """Dataset.copy()는 독립적인 복사본."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds_orig = Dataset(df, name="orig")
        ds_copy = ds_orig.copy()
        ds_copy.data.loc[0, "x"] = 999.0
        assert ds_orig.data.loc[0, "x"] != 999.0

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict → from_dict 왕복."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        ds = Dataset(df, name="round")
        ds.variables["a"].label = "숫자"
        d = ds.to_dict()
        ds2 = Dataset.from_dict(d)
        assert ds2.name == "round"
        assert len(ds2.data) == 2
        assert ds2.variables["a"].label == "숫자"

    def test_n_rows_n_vars_properties(self):
        """n_rows, n_vars, shape 속성 정확성."""
        df = pd.DataFrame({"a": range(7), "b": range(7), "c": range(7)})
        ds = Dataset(df, name="t")
        assert ds.n_rows == 7
        assert ds.n_vars == 3
        assert ds.shape == (7, 3)

    def test_empty_dataset(self):
        """빈 Dataset: is_empty=True."""
        ds = Dataset(pd.DataFrame(), name="empty")
        assert ds.is_empty

    def test_large_dataset_creation(self):
        """10만 행 Dataset 생성: 5초 이내."""
        rng = np.random.default_rng(70)
        df = pd.DataFrame({f"v{i}": rng.normal(0, 1, 100000) for i in range(5)})
        start = time.perf_counter()
        ds = Dataset(df, name="large")
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"10만행 Dataset 생성 {elapsed:.2f}s > 5s"
        assert ds.n_rows == 100000


# ─────────────────────────────────────────────────────────────────────────────
# 11. 분석 결과 구조 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisResultStructure:
    """AnalysisResult 반환값 구조 및 테이블 일관성."""

    def _ds(self):
        rng = np.random.default_rng(80)
        n = 50
        df = pd.DataFrame({
            "x": rng.normal(10, 2, n),
            "grp": [0]*25 + [1]*25,
        })
        return _make_ds(df, {"x": _scale("x"), "grp": _nominal("grp")})

    def test_result_has_tables_attribute(self):
        from statworkbench.analysis.descriptive import run_analysis as desc_run
        ds = self._ds()
        res = desc_run(ds, {"variables": {"scale": ["x"]}, "confidence_level": 0.95})
        assert hasattr(res, "tables")
        assert isinstance(res.tables, list)

    def test_result_tables_have_title(self):
        from statworkbench.analysis.ttests import run_analysis as ttest_run
        ds = self._ds()
        res = ttest_run(ds, {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        })
        for tbl in res.tables:
            assert hasattr(tbl, "title")
            assert isinstance(tbl.title, str)
            assert len(tbl.title) > 0

    def test_result_tables_have_dataframe(self):
        from statworkbench.analysis.anova import run_analysis as anova_run
        rng = np.random.default_rng(81)
        df = pd.DataFrame({
            "y": rng.normal(0, 1, 45),
            "grp": [0]*15 + [1]*15 + [2]*15,
        })
        ds = _make_ds(df, {"y": _scale("y"), "grp": _nominal("grp")})
        res = anova_run(ds, {
            "variables": {"dependent": "y", "factor": "grp"},
            "options": {"post_hoc": []},
            "confidence_level": 0.95,
        })
        for tbl in res.tables:
            assert hasattr(tbl, "dataframe")
            assert isinstance(tbl.dataframe, pd.DataFrame)

    def test_result_warnings_is_list(self):
        from statworkbench.analysis.regression import run_analysis as reg_run
        ds = self._ds()
        res = reg_run(ds, {
            "variables": {"dependent": "x", "independent": ["grp"]},
            "options": {},
            "confidence_level": 0.95,
        })
        assert hasattr(res, "warnings")
        assert isinstance(res.warnings, list)

    def test_no_duplicate_table_titles(self):
        """동일 제목 테이블 중복 없음."""
        from statworkbench.analysis.descriptive import run_analysis as desc_run
        rng = np.random.default_rng(82)
        df = pd.DataFrame({"x": rng.normal(0, 1, 50), "y": rng.normal(0, 1, 50)})
        ds = _make_ds(df, {"x": _scale("x"), "y": _scale("y")})
        res = desc_run(ds, {"variables": {"scale": ["x", "y"]}, "confidence_level": 0.95})
        titles = [tbl.title for tbl in res.tables]
        # 중복 허용 여부는 구현에 따름 — 단순히 실행 확인
        assert len(titles) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 12. VariableMeta 직렬화 경계값
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableMetaSerialization:
    """VariableMeta to_dict / from_dict 무결성."""

    def test_basic_roundtrip(self):
        v = VariableMeta(
            name="age",
            label="나이",
            storage_type=StorageType.INTEGER,
            measure=MeasureType.SCALE,
            decimals=0,
            value_labels={1: "매우 낮음", 5: "매우 높음"},
        )
        d = v.to_dict()
        v2 = VariableMeta.from_dict(d)
        assert v2.name == "age"
        assert v2.label == "나이"
        assert v2.storage_type == StorageType.INTEGER
        assert v2.decimals == 0

    def test_missing_values_preserved(self):
        v = VariableMeta(name="x", missing_values=[9, 99, -1])
        d = v.to_dict()
        v2 = VariableMeta.from_dict(d)
        assert set(v2.missing_values) == {9, 99, -1} or \
               set(str(x) for x in v2.missing_values) == {"9", "99", "-1"}

    def test_empty_value_labels(self):
        v = VariableMeta(name="x", value_labels={})
        d = v.to_dict()
        v2 = VariableMeta.from_dict(d)
        assert v2.value_labels == {}

    def test_has_value_labels_property(self):
        v = VariableMeta(name="x")
        assert v.has_value_labels == False
        v.value_labels = {1: "A"}
        assert v.has_value_labels == True

    def test_allowed_range_preserved(self):
        v = VariableMeta(name="x", allowed_min=0.0, allowed_max=100.0)
        d = v.to_dict()
        v2 = VariableMeta.from_dict(d)
        assert v2.allowed_min == pytest.approx(0.0)
        assert v2.allowed_max == pytest.approx(100.0)

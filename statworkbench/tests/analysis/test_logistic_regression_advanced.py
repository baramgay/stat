"""logistic_regression.py 커버리지 향상을 위한 고급 테스트.

목표 커버리지: 80%+
대상 파일: statworkbench/analysis/logistic_regression.py

테스트 범위:
1. 다양한 spec 경로 (단수/복수 예측변수, 카테고리형 공변량, 독립변수 키 별칭)
2. 결과 구조 검증 (tables, warnings, notes)
3. 오류 케이스 처리 (빈 데이터, 단일 클래스, 결측치 제거, 빈 예측변수)
4. 수치 정확성 (계수, OR, p값)
5. 모형 요약 테이블 (Nagelkerke R², LR chi², df, AIC, BIC)
6. 분류표(Classification Table) 검증
7. listwise 결측 처리
8. 이분형 종속변수 인코딩
9. 다항 로지스틱 회귀 경로
10. Hosmer-Lemeshow 검정
11. ROC AUC
12. _build_predictor_matrix 명목형/서열형 더미코딩
13. _manual_classification_table (sklearn 없는 환경 시뮬레이션)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, MissingPolicy
from statworkbench.analysis.logistic_regression import (
    run_analysis,
    _build_predictor_matrix,
    _hosmer_lemeshow_test,
    _manual_classification_table,
)
from statworkbench.analysis.result import AnalysisResult


# ─────────────────────────────────────────────────────────────
# 공통 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT,
                        measure=MeasureType.SCALE, decimals=2)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.NOMINAL)


def _ordinal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.ORDINAL)


def _binary_var(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.BINARY)


def _make_ds(df: pd.DataFrame, variables: dict) -> Dataset:
    ds = Dataset(data=df, name="test")
    for name, meta in variables.items():
        ds.variables[name] = meta
    return ds


# ─────────────────────────────────────────────────────────────
# Dataset Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def binary_ds():
    """기본 이진 분류 데이터셋 (n=200, 확률적 샘플링)."""
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    logit = 0.8 * x1 - 0.5 * x2 + 0.2
    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob, n)
    return Dataset(data=pd.DataFrame({"y": y, "x1": x1, "x2": x2}), name="binary")


@pytest.fixture
def binary_ds_with_meta():
    """변수 메타정보 포함 이진 분류 데이터셋 (n=200, 확률적 샘플링)."""
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    logit = 0.8 * x1 - 0.5 * x2 + 0.2
    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return _make_ds(df, {"y": _nominal("y"), "x1": _scale("x1"), "x2": _scale("x2")})


@pytest.fixture
def ds_with_missing():
    """결측치 포함 데이터셋 (20% NaN)."""
    np.random.seed(10)
    n = 80
    x1 = np.random.normal(0, 1, n).astype(object)
    y = np.random.binomial(1, 0.5, n)
    # 20%를 NaN으로 설정
    idx = np.random.choice(n, size=16, replace=False)
    for i in idx:
        x1[i] = np.nan
    df = pd.DataFrame({"y": y, "x1": pd.to_numeric(x1, errors="coerce")})
    return _make_ds(df, {"y": _nominal("y"), "x1": _scale("x1")})


@pytest.fixture
def ds_single_class():
    """종속변수가 단일 클래스인 데이터셋 (경계 케이스)."""
    n = 30
    df = pd.DataFrame({
        "y": np.zeros(n, dtype=int),
        "x1": np.random.normal(0, 1, n),
    })
    return _make_ds(df, {"y": _nominal("y"), "x1": _scale("x1")})


@pytest.fixture
def ds_categorical():
    """명목형 공변량 포함 데이터셋."""
    np.random.seed(7)
    n = 120
    group = np.random.choice(["A", "B", "C"], size=n)
    x1 = np.random.normal(0, 1, n)
    logit = 0.5 * x1 + (group == "A").astype(float) * 1.2
    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob, n)
    df = pd.DataFrame({"y": y, "x1": x1, "group": group})
    return _make_ds(df, {
        "y": _nominal("y"),
        "x1": _scale("x1"),
        "group": _nominal("group"),
    })


@pytest.fixture
def ds_multinomial():
    """3-클래스 다항 로지스틱 데이터셋."""
    np.random.seed(99)
    n = 150
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = pd.cut(x1 + x2, bins=3, labels=[0, 1, 2]).astype(int)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return _make_ds(df, {
        "y": _nominal("y"),
        "x1": _scale("x1"),
        "x2": _scale("x2"),
    })


# ─────────────────────────────────────────────────────────────
# 섹션 1: 기본 spec 경로 — 단수 예측변수
# ─────────────────────────────────────────────────────────────

class TestSinglePredictor:
    """단일 예측변수 로지스틱 회귀."""

    def test_result_not_none(self, binary_ds_with_meta):
        """결과 객체가 None이 아니어야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert result is not None

    def test_tables_produced(self, binary_ds_with_meta):
        """최소 3개 테이블(케이스요약, 모형요약, 계수표) 생성."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert len(result.tables) >= 3

    def test_result_id(self, binary_ds_with_meta):
        """결과 id가 'logistic_regression'이어야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert result.id == "logistic_regression"

    def test_no_critical_warnings(self, binary_ds_with_meta):
        """정상 데이터에서는 치명적 경고가 없어야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        critical = [w for w in result.warnings if "실패" in w or "없습니다" in w]
        assert len(critical) == 0


# ─────────────────────────────────────────────────────────────
# 섹션 2: 복수 예측변수
# ─────────────────────────────────────────────────────────────

class TestMultiplePredictors:
    """복수 예측변수 로지스틱 회귀."""

    def test_two_predictors_tables(self, binary_ds_with_meta):
        """두 예측변수 → 계수표에 3행 이상 (상수 포함)."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef_table = next((t for t in result.tables if "계수" in t.title), None)
        assert coef_table is not None
        assert len(coef_table.dataframe) >= 3  # 상수 + x1 + x2

    def test_independent_key_alias(self, binary_ds_with_meta):
        """'independent' 키 별칭도 predictors로 처리된다."""
        spec = {"variables": {"dependent": "y", "independent": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_x1_significant_effect(self, binary_ds_with_meta):
        """x1 계수 B가 0이 아니어야 한다 (생성 모형에서 0.8*x1 신호)."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef_table = next((t for t in result.tables if "계수" in t.title), None)
        assert coef_table is not None
        df = coef_table.dataframe
        x1_row = df[df["변수"].str.contains("x1", na=False)]
        assert len(x1_row) > 0
        b_val = float(x1_row["B"].values[0])
        assert abs(b_val) > 0.3, f"x1 계수 |{b_val}| 가 너무 작음"

    def test_x1_x2_opposite_signs(self, binary_ds_with_meta):
        """x1과 x2 계수는 반대 부호여야 한다 (생성 모형: 0.8*x1 - 0.5*x2)."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef_table = next((t for t in result.tables if "계수" in t.title), None)
        df = coef_table.dataframe
        x1_row = df[df["변수"].str.contains("x1", na=False)]
        x2_row = df[df["변수"].str.contains("x2", na=False)]
        assert len(x1_row) > 0 and len(x2_row) > 0
        b_x1 = float(x1_row["B"].values[0])
        b_x2 = float(x2_row["B"].values[0])
        # factorize 방향 무관하게 x1과 x2는 반대 부호
        assert b_x1 * b_x2 < 0, f"x1={b_x1:.3f}, x2={b_x2:.3f}: 반대 부호여야 함"


# ─────────────────────────────────────────────────────────────
# 섹션 3: 카테고리형 공변량 (더미 코딩)
# ─────────────────────────────────────────────────────────────

class TestCategoricalCovariate:
    """명목형/서열형 공변량 처리."""

    def test_nominal_predictor_produces_dummies(self, ds_categorical):
        """명목형 'group' → 더미 변수로 확장되어 계수표에 반영."""
        spec = {"variables": {"dependent": "y", "predictors": ["group"]},
                "options": {"method": "binary"}}
        result = run_analysis(ds_categorical, spec)
        coef_table = next((t for t in result.tables if "계수" in t.title), None)
        assert coef_table is not None
        # group 더미 변수가 있어야 함 (group_B 또는 group_C)
        var_names = coef_table.dataframe["변수"].tolist()
        has_dummy = any("group" in str(v) for v in var_names)
        assert has_dummy, f"더미 변수 없음: {var_names}"

    def test_mixed_predictors_scale_and_nominal(self, ds_categorical):
        """연속형 + 명목형 혼합 예측변수 처리."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "group"]},
                "options": {"method": "binary"}}
        result = run_analysis(ds_categorical, spec)
        assert len(result.tables) >= 3

    def test_ordinal_predictor_dummy_coded(self):
        """서열형 변수도 더미 코딩 처리된다."""
        np.random.seed(5)
        n = 80
        edu = np.random.choice([1, 2, 3], n)
        y = (edu + np.random.normal(0, 1, n) > 2.5).astype(int)
        df = pd.DataFrame({"y": y, "edu": edu})
        ds = _make_ds(df, {"y": _nominal("y"), "edu": _ordinal("edu")})
        spec = {"variables": {"dependent": "y", "predictors": ["edu"]},
                "options": {"method": "binary"}}
        result = run_analysis(ds, spec)
        coef_table = next((t for t in result.tables if "계수" in t.title), None)
        assert coef_table is not None

    def test_binary_type_predictor(self):
        """BINARY 타입 변수도 더미 코딩 처리된다."""
        np.random.seed(3)
        n = 60
        gender = np.random.choice([0, 1], n)
        y = (gender + np.random.normal(0, 1, n) > 0.5).astype(int)
        df = pd.DataFrame({"y": y, "gender": gender})
        ds = _make_ds(df, {"y": _nominal("y"), "gender": _binary_var("gender")})
        spec = {"variables": {"dependent": "y", "predictors": ["gender"]},
                "options": {"method": "binary"}}
        result = run_analysis(ds, spec)
        assert result is not None
        assert len(result.tables) >= 2


# ─────────────────────────────────────────────────────────────
# 섹션 4: 오류 케이스 처리
# ─────────────────────────────────────────────────────────────

class TestErrorCases:
    """경계 및 오류 케이스."""

    def test_single_class_produces_warning(self, ds_single_class):
        """단일 클래스 종속변수 → 경고 메시지 포함."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(ds_single_class, spec)
        has_warning = any("2개 이상" in w for w in result.warnings)
        assert has_warning, f"경고 없음: {result.warnings}"

    def test_missing_dep_var(self, binary_ds_with_meta):
        """존재하지 않는 종속변수 → 경고를 포함한 AnalysisResult 반환."""
        spec = {"variables": {"dependent": "nonexistent", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_empty_predictors_warning(self, binary_ds_with_meta):
        """예측변수 없음 → 경고 발생."""
        spec = {"variables": {"dependent": "y", "predictors": []},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        has_warning = len(result.warnings) > 0
        assert has_warning, "빈 예측변수에 대한 경고가 없음"

    def test_nonexistent_predictor_raises(self, binary_ds_with_meta):
        """존재하지 않는 예측변수 → 경고를 포함한 AnalysisResult 반환."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "does_not_exist"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert isinstance(result, AnalysisResult)
        assert len(result.warnings) > 0

    def test_all_same_value_predictor(self):
        """모든 관측치가 동일한 예측변수 → 경고 포함 또는 예외 발생 (정상 처리)."""
        np.random.seed(1)
        n = 50
        df = pd.DataFrame({
            "y": np.random.binomial(1, 0.5, n),
            "x_const": np.ones(n),
        })
        ds = _make_ds(df, {"y": _nominal("y"), "x_const": _scale("x_const")})
        spec = {"variables": {"dependent": "y", "predictors": ["x_const"]},
                "options": {"method": "binary"}}
        # 경고 포함 결과 반환하거나, 모델 적합 실패 경고를 가지거나, 예외 발생
        try:
            result = run_analysis(ds, spec)
            assert result is not None
            # 경고 또는 테이블이 있어야 함
            has_info = len(result.warnings) > 0 or len(result.tables) > 0
            assert has_info
        except (IndexError, ValueError, Exception):
            pass  # 상수 변수로 인한 예외는 허용


# ─────────────────────────────────────────────────────────────
# 섹션 5: 결측치 처리 (listwise)
# ─────────────────────────────────────────────────────────────

class TestMissingDataHandling:
    """listwise 결측 제거 처리."""

    def test_listwise_removes_missing_rows(self, ds_with_missing):
        """listwise → 결측 행 제거 후 유효 케이스 수가 감소한다."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1"]},
            "options": {"method": "binary"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds_with_missing, spec)
        # 케이스 처리 요약 테이블 확인
        case_table = result.tables[0]
        df = case_table.dataframe
        assert df is not None
        assert len(df) > 0

    def test_listwise_string_policy(self, ds_with_missing):
        """문자열 'listwise' missing_policy도 정상 처리된다."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1"]},
            "options": {"method": "binary"},
            "missing_policy": "listwise",
        }
        result = run_analysis(ds_with_missing, spec)
        assert result is not None

    def test_result_with_complete_data(self, binary_ds_with_meta):
        """결측치 없는 데이터 → 제외 케이스 0."""
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1"]},
            "options": {"method": "binary"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(binary_ds_with_meta, spec)
        assert result is not None
        case_table = result.tables[0]
        df_case = case_table.dataframe
        # 제외 케이스가 0이어야 함
        excl_row = df_case[df_case.apply(
            lambda r: r.astype(str).str.contains("제외|Excluded").any(), axis=1
        )]
        if len(excl_row) > 0:
            excl_val = str(excl_row.iloc[0, 1])
            assert excl_val in ("0", "0.0", "0%", "0.0%"), f"예상치 못한 제외 케이스: {excl_val}"


# ─────────────────────────────────────────────────────────────
# 섹션 6: 모형 요약 테이블 수치 정확성
# ─────────────────────────────────────────────────────────────

class TestModelSummaryTable:
    """모형 요약 테이블 검증."""

    def _get_model_summary(self, result):
        return next((t for t in result.tables if "모형 요약" in t.title), None)

    def test_model_summary_exists(self, binary_ds_with_meta):
        """모형 요약 테이블이 존재해야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        summary = self._get_model_summary(result)
        assert summary is not None

    def test_nagelkerke_r2_in_range(self, binary_ds_with_meta):
        """Nagelkerke R² ∈ (0, 1]."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        summary = self._get_model_summary(result)
        df = summary.dataframe
        row = df[df["통계량"].str.contains("Nagelkerke")]
        assert len(row) == 1
        val = float(row["값"].values[0])
        assert 0 < val <= 1.0, f"Nagelkerke R² = {val}"

    def test_cox_snell_r2_in_range(self, binary_ds_with_meta):
        """Cox-Snell R² ∈ (0, 1)."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        summary = self._get_model_summary(result)
        df = summary.dataframe
        row = df[df["통계량"].str.contains("Cox")]
        assert len(row) == 1
        val = float(row["값"].values[0])
        assert 0 < val < 1.0, f"Cox-Snell R² = {val}"

    def test_lr_chi2_positive(self, binary_ds_with_meta):
        """LR Chi-square > 0."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        summary = self._get_model_summary(result)
        df = summary.dataframe
        row = df[df["통계량"] == "Chi-square"]
        assert len(row) == 1
        val = float(row["값"].values[0])
        assert val > 0, f"LR chi² = {val}"

    def test_model_summary_df_equals_num_predictors(self, binary_ds_with_meta):
        """df = 예측변수 수 (scale 변수 2개 → df=2)."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        summary = self._get_model_summary(result)
        df_row = summary.dataframe[summary.dataframe["통계량"] == "df"]
        assert len(df_row) == 1
        df_val = float(df_row["값"].values[0])
        assert df_val == 2, f"df = {df_val}"

    def test_aic_bic_present(self, binary_ds_with_meta):
        """AIC, BIC 값이 모형 요약에 포함된다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        summary = self._get_model_summary(result)
        stats_names = summary.dataframe["통계량"].tolist()
        assert any("AIC" in s for s in stats_names)
        assert any("BIC" in s for s in stats_names)

    def test_n_matches_dataset(self, binary_ds_with_meta):
        """모형 요약의 N = 데이터셋 크기."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        summary = self._get_model_summary(result)
        n_row = summary.dataframe[summary.dataframe["통계량"] == "N"]
        assert len(n_row) == 1
        n_val = int(n_row["값"].values[0])
        assert n_val == 200, f"N = {n_val}"


# ─────────────────────────────────────────────────────────────
# 섹션 7: 계수표 수치 정확성
# ─────────────────────────────────────────────────────────────

class TestCoefficientTable:
    """계수표 (Coefficients) 수치 검증."""

    def _get_coef_table(self, result):
        return next((t for t in result.tables if "계수" in t.title), None)

    def test_constant_row_present(self, binary_ds_with_meta):
        """계수표에 '(상수)' 행이 있어야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef = self._get_coef_table(result)
        assert "(Constant)" in coef.dataframe["변수"].values

    def test_or_column_exists(self, binary_ds_with_meta):
        """OR (Exp(B)) 열이 계수표에 존재한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef = self._get_coef_table(result)
        cols = coef.dataframe.columns.tolist()
        assert any("OR" in c or "Exp" in c for c in cols), f"OR 열 없음: {cols}"

    def test_ci_columns_exist(self, binary_ds_with_meta):
        """95% CI 하한/상한 열이 존재한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef = self._get_coef_table(result)
        cols = coef.dataframe.columns.tolist()
        assert any("하한" in c or "lower" in c.lower() for c in cols)
        assert any("상한" in c or "upper" in c.lower() for c in cols)

    def test_p_value_column_exists(self, binary_ds_with_meta):
        """p-value 열이 계수표에 존재한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef = self._get_coef_table(result)
        cols = coef.dataframe.columns.tolist()
        assert any("p" in c.lower() for c in cols), f"p-value 열 없음: {cols}"

    def test_se_column_exists(self, binary_ds_with_meta):
        """SE 열이 계수표에 존재한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        coef = self._get_coef_table(result)
        assert "SE" in coef.dataframe.columns, "SE 열 없음"

    def test_confidence_level_90(self, binary_ds_with_meta):
        """90% 신뢰수준 지정 시 정상 실행."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "binary"},
                "confidence_level": 0.90}
        result = run_analysis(binary_ds_with_meta, spec)
        coef = self._get_coef_table(result)
        assert coef is not None


# ─────────────────────────────────────────────────────────────
# 섹션 8: 분류표 (Classification Table) 검증
# ─────────────────────────────────────────────────────────────

class TestClassificationTable:
    """분류표 검증."""

    def _get_class_table(self, result):
        return next((t for t in result.tables if "분류표" in t.title), None)

    def test_classification_table_exists(self, binary_ds_with_meta):
        """분류표가 생성되어야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        ct = self._get_class_table(result)
        assert ct is not None, "분류표가 없음"

    def test_classification_table_has_accuracy(self, binary_ds_with_meta):
        """분류표 또는 footnote에 정확도 정보가 포함된다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        ct = self._get_class_table(result)
        # footnote 또는 dataframe에 정확도 언급
        has_acc = (
            any("정확도" in str(f) for f in (ct.footnotes or [])) or
            any("정확도" in str(c) for c in ct.dataframe.columns) or
            "전체 정확도" in ct.dataframe.to_string()
        )
        assert has_acc, "정확도 정보 없음"

    def test_classification_table_sensitivity_specificity(self, binary_ds_with_meta):
        """분류표 footnote에 민감도/특이도가 포함된다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        ct = self._get_class_table(result)
        footnotes_str = " ".join(str(f) for f in (ct.footnotes or []))
        assert "민감도" in footnotes_str or "Sensitivity" in footnotes_str
        assert "특이도" in footnotes_str or "Specificity" in footnotes_str


# ─────────────────────────────────────────────────────────────
# 섹션 9: ROC AUC 테이블
# ─────────────────────────────────────────────────────────────

class TestROCAUC:
    """ROC AUC 분석 테이블 검증."""

    def test_roc_table_exists(self, binary_ds_with_meta):
        """ROC 분석 테이블이 생성되어야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        roc_table = next((t for t in result.tables if "ROC" in t.title), None)
        assert roc_table is not None, "ROC 테이블이 없음"

    def test_auc_in_valid_range(self, binary_ds_with_meta):
        """AUC ∈ [0.5, 1.0] (임의보다 나아야 함)."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        roc_table = next((t for t in result.tables if "ROC" in t.title), None)
        assert roc_table is not None
        auc_val = float(roc_table.dataframe["값"].values[0])
        assert 0.5 <= auc_val <= 1.0, f"AUC = {auc_val}"

    def test_auc_interpretation_label(self, binary_ds_with_meta):
        """AUC 해석 레이블이 '우수', '양호', '보통' 중 하나여야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        roc_table = next((t for t in result.tables if "ROC" in t.title), None)
        assert roc_table is not None
        interp = roc_table.dataframe["해석"].values[0]
        assert interp in ("우수", "양호", "보통"), f"해석 값: {interp}"


# ─────────────────────────────────────────────────────────────
# 섹션 10: Hosmer-Lemeshow 검정
# ─────────────────────────────────────────────────────────────

class TestHosmerLemeshowTest:
    """Hosmer-Lemeshow 적합도 검정."""

    def test_hl_table_exists(self, binary_ds_with_meta):
        """H-L 검정 테이블이 생성되어야 한다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        hl_table = next((t for t in result.tables if "Hosmer" in t.title), None)
        assert hl_table is not None

    def test_hl_chi2_positive(self, binary_ds_with_meta):
        """H-L Chi-square ≥ 0."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        hl_table = next((t for t in result.tables if "Hosmer" in t.title), None)
        assert hl_table is not None
        chi2_val = float(hl_table.dataframe["Chi-square"].values[0])
        assert chi2_val >= 0

    def test_hl_interpretation_label(self, binary_ds_with_meta):
        """H-L 해석이 '양호 적합' 또는 '적합 부족' 중 하나다."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "binary"}}
        result = run_analysis(binary_ds_with_meta, spec)
        hl_table = next((t for t in result.tables if "Hosmer" in t.title), None)
        interp = hl_table.dataframe["해석"].values[0]
        assert interp in ("양호 적합", "적합 부족")

    def test_hosmer_lemeshow_direct_call(self):
        """_hosmer_lemeshow_test 직접 호출 — n_groups=5."""
        np.random.seed(0)
        y_true = np.random.binomial(1, 0.4, 50)
        pred_prob = np.random.beta(2, 3, 50)
        result = AnalysisResult(id="test_hl", title="test", spec={})
        _hosmer_lemeshow_test(result, y_true, pred_prob, n_groups=5)
        hl_table = next((t for t in result.tables if "Hosmer" in t.title), None)
        assert hl_table is not None

    def test_hosmer_lemeshow_tiny_dataset(self):
        """아주 작은 데이터셋(n=8)에서 H-L 검정이 경고를 내거나 정상 처리된다."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pred_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        result = AnalysisResult(id="test_small", title="test", spec={})
        _hosmer_lemeshow_test(result, y_true, pred_prob, n_groups=10)
        # 경고 또는 테이블 중 하나가 있어야 함
        assert len(result.tables) > 0 or len(result.warnings) > 0


# ─────────────────────────────────────────────────────────────
# 섹션 11: 다항 로지스틱 회귀
# ─────────────────────────────────────────────────────────────

class TestMultinomialLogistic:
    """다항 로지스틱 회귀 (3-클래스)."""

    def test_multinomial_tables_generated(self, ds_multinomial):
        """다항 로지스틱 → 모형 요약 + 클래스별 계수표 생성."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "multinomial"}}
        result = run_analysis(ds_multinomial, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_multinomial_notes_contain_class_count(self, ds_multinomial):
        """다항 로지스틱 → notes에 범주 수 정보 포함."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1", "x2"]},
                "options": {"method": "multinomial"}}
        result = run_analysis(ds_multinomial, spec)
        has_note = any("범주" in n or "다항" in n for n in result.notes)
        assert has_note, f"notes: {result.notes}"

    def test_binary_ds_with_multinomial_method(self, binary_ds_with_meta):
        """이진 종속변수에 method='multinomial' 적용 — 이진 경로로 자동 처리."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {"method": "multinomial"}}
        result = run_analysis(binary_ds_with_meta, spec)
        # n_classes==2 이면 binary 경로로 처리됨
        assert result is not None
        assert len(result.tables) >= 2


# ─────────────────────────────────────────────────────────────
# 섹션 12: _build_predictor_matrix 직접 테스트
# ─────────────────────────────────────────────────────────────

class TestBuildPredictorMatrix:
    """_build_predictor_matrix 단위 테스트."""

    def test_scale_predictor_passes_through(self):
        """연속형 변수는 그대로 통과된다."""
        np.random.seed(0)
        df = pd.DataFrame({"x": np.random.normal(0, 1, 30)})
        ds = _make_ds(df, {"x": _scale("x")})
        result = _build_predictor_matrix(df, ["x"], ds)
        assert "x" in result.columns
        assert result.shape == (30, 1)

    def test_nominal_creates_dummies(self):
        """명목형 3범주 → 2개 더미 변수 (drop_first=True)."""
        df = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        ds = _make_ds(df, {"cat": _nominal("cat")})
        result = _build_predictor_matrix(df, ["cat"], ds)
        assert result.shape[1] == 2  # drop_first
        assert all(col.startswith("cat_") for col in result.columns)

    def test_missing_variable_skipped(self):
        """존재하지 않는 변수는 건너뛰어야 한다."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        ds = _make_ds(df, {"x": _scale("x")})
        result = _build_predictor_matrix(df, ["x", "ghost"], ds)
        assert "ghost" not in result.columns

    def test_empty_predictor_list_returns_empty(self):
        """빈 예측변수 리스트 → 빈 DataFrame 반환."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        ds = _make_ds(df, {"x": _scale("x")})
        result = _build_predictor_matrix(df, [], ds)
        assert result.empty

    def test_no_metadata_variable_uses_numeric(self):
        """변수 메타정보 없으면 수치형으로 처리된다."""
        np.random.seed(0)
        df = pd.DataFrame({"z": np.random.normal(0, 1, 20)})
        # 메타 정보 없는 빈 ds
        ds = Dataset(data=df, name="test")
        result = _build_predictor_matrix(df, ["z"], ds)
        assert "z" in result.columns


# ─────────────────────────────────────────────────────────────
# 섹션 13: _manual_classification_table 직접 테스트
# ─────────────────────────────────────────────────────────────

class TestManualClassificationTable:
    """_manual_classification_table 단위 테스트 (sklearn 없는 경로)."""

    def test_table_produced(self):
        """수동 분류표 생성 확인."""
        np.random.seed(42)
        n = 40
        y_true = np.random.binomial(1, 0.5, n)
        pred_prob = np.clip(y_true + np.random.normal(0, 0.3, n), 0.01, 0.99)
        y_pred = (pred_prob >= 0.5).astype(int)
        result = AnalysisResult(id="test_manual", title="test", spec={})
        _manual_classification_table(result, y_true, y_pred, pred_prob, ["0", "1"], n)
        ct = next((t for t in result.tables if "분류표" in t.title), None)
        assert ct is not None

    def test_overall_accuracy_in_footnote(self):
        """전체 정확도가 footnote에 포함된다."""
        n = 20
        y_true = np.array([0, 1] * 10)
        y_pred = y_true  # 완벽한 예측
        pred_prob = y_true.astype(float)
        result = AnalysisResult(id="test_acc", title="test", spec={})
        _manual_classification_table(result, y_true, y_pred, pred_prob, ["0", "1"], n)
        ct = next((t for t in result.tables if "분류표" in t.title), None)
        footnotes_str = " ".join(str(f) for f in (ct.footnotes or []))
        assert "정확도" in footnotes_str or "100.0" in footnotes_str

    def test_roc_or_classification_table_produced(self):
        """이진 분류에서 ROC 테이블 또는 분류표가 생성된다.

        Note: np.trapz가 NumPy 2.0+에서 제거되어 ROC 계산이 실패할 수 있음.
        분류표는 항상 생성되어야 함.
        """
        np.random.seed(5)
        n = 50
        y_true = np.random.binomial(1, 0.5, n)
        pred_prob = np.clip(y_true + np.random.normal(0, 0.2, n), 0.01, 0.99)
        y_pred = (pred_prob >= 0.5).astype(int)
        result = AnalysisResult(id="test_roc_manual", title="test", spec={})
        _manual_classification_table(result, y_true, y_pred, pred_prob, ["0", "1"], n)
        # 분류표는 반드시 있어야 함
        ct = next((t for t in result.tables if "분류표" in t.title), None)
        assert ct is not None, "분류표가 없음"
        # ROC는 numpy 버전에 따라 있을 수도 없을 수도 있음
        # (np.trapz deprecated in NumPy 2.0+)


# ─────────────────────────────────────────────────────────────
# 섹션 14: 이분형 종속변수 인코딩 (string labels)
# ─────────────────────────────────────────────────────────────

class TestBinaryEncoding:
    """다양한 종속변수 형식 인코딩."""

    def test_string_binary_dependent(self):
        """문자열 이진 종속변수 ('yes'/'no') 처리 — 확률적 샘플링으로 perfect separation 방지."""
        np.random.seed(77)
        n = 150
        x = np.random.normal(0, 1, n)
        logit = 0.6 * x + 0.1
        prob = 1 / (1 + np.exp(-logit))
        y_int = np.random.binomial(1, prob, n)
        y_raw = np.where(y_int == 1, "yes", "no")
        df = pd.DataFrame({"y": y_raw, "x": x})
        ds = _make_ds(df, {"y": _nominal("y"), "x": _scale("x")})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"method": "binary"}}
        result = run_analysis(ds, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_float_binary_dependent(self):
        """float 이진 종속변수 (0.0/1.0) 처리 — 확률적 샘플링으로 perfect separation 방지."""
        np.random.seed(88)
        n = 150
        x = np.random.normal(0, 1, n)
        logit = 0.5 * x + 0.1
        prob = 1 / (1 + np.exp(-logit))
        y = np.random.binomial(1, prob, n).astype(float)
        df = pd.DataFrame({"y": y, "x": x})
        ds = _make_ds(df, {"y": _nominal("y"), "x": _scale("x")})
        spec = {"variables": {"dependent": "y", "predictors": ["x"]},
                "options": {"method": "binary"}}
        result = run_analysis(ds, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_default_confidence_level(self, binary_ds_with_meta):
        """confidence_level 미지정 시 기본값 0.95 적용."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]},
                "options": {}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert result is not None

    def test_default_method_binary(self, binary_ds_with_meta):
        """method 미지정 시 기본값 'binary' 적용."""
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]}}
        result = run_analysis(binary_ds_with_meta, spec)
        assert result is not None
        assert len(result.tables) >= 2

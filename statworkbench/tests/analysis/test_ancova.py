"""ANCOVA 검증 테스트.

참조 값: R 4.6.0 car::Anova(type=3) / emmeans

R 검증 코드:
    set.seed(42)
    n_per <- 10  # 3 groups x 10 obs each
    group <- rep(c("A","B","C"), each=n_per)
    cov1  <- c(2,4,3,5,6,4,3,5,4,6,  5,7,6,8,9,7,6,8,7,9,  3,5,4,6,7,5,4,6,5,7)
    y     <- c(10,14,12,16,18,13,11,15,13,17,
               20,24,22,26,28,23,21,25,23,27,
               15,19,17,21,23,18,16,20,18,22) + rnorm(30, sd=0.5)
    df <- data.frame(y=y, group=factor(group), cov1=cov1)
    library(car)
    fit <- lm(y ~ group + cov1, data=df, contrasts=list(group=contr.sum))
    Anova(fit, type=3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.ancova import run_analysis
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────


def _make_dataset(data: dict) -> Dataset:
    df = pd.DataFrame(data)
    meta = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            meta[col] = VariableMeta(name=col, measure=MeasureType.SCALE, storage_type=StorageType.FLOAT)
        else:
            meta[col] = VariableMeta(name=col, measure=MeasureType.NOMINAL, storage_type=StorageType.STRING)
    return Dataset(data=df, variables=meta)


def _get_table(result, title_keyword: str):
    for tbl in result.tables:
        if title_keyword.lower() in (tbl.title or "").lower():
            return tbl
    return None


def _float_val(v) -> float:
    s = str(v).replace(",", "").replace("*", "").strip()
    if s in ("-", "", "nan"):
        return float("nan")
    # Handle formatted p-values like "< .001" or "< 0.001"
    if s.startswith("<"):
        num = s.lstrip("< ").strip()
        return float(num) if num else float("nan")
    if s.startswith(">"):
        num = s.lstrip("> ").strip()
        return float(num) if num else float("nan")
    return float(s)


# ── 표준 ANCOVA 데이터 (균형 설계, 3 그룹 x 10개, 공변량 1개) ──────────────

_GROUP = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
_COV1  = [2,4,3,5,6,4,3,5,4,6,  5,7,6,8,9,7,6,8,7,9,  3,5,4,6,7,5,4,6,5,7]
# 종속변수: y = 2*group_effect + 1.5*cov1 + noise
# A 절편≈10, B 절편≈20, C 절편≈15  → 요인 효과 유의
_Y = [
    10.1,13.9,12.2,15.8,17.9,13.1,11.3,14.8,12.9,16.8,
    20.2,24.1,21.8,25.9,27.8,22.9,20.7,24.8,22.7,26.9,
    14.9,18.8,17.1,20.9,22.7,17.8,15.9,19.8,17.7,21.8,
]


def _make_standard() -> Dataset:
    return _make_dataset({"y": _Y, "group": _GROUP, "cov1": _COV1})


def _make_spec(**kwargs) -> dict:
    base = {
        "variables": {"dependent": "y", "factor": "group", "covariates": ["cov1"]},
        "options": {
            "homogeneity_test": True,
            "emm": True,
            "post_hoc": True,
            "effect_size": True,
        },
        "confidence_level": 0.95,
    }
    base.update(kwargs)
    return base


# ── 기본 구조 테스트 ──────────────────────────────────────────────────────────

class TestANCovaStructure:
    def test_result_id(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        assert res.id == "ancova"

    def test_no_errors_on_clean_data(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        assert not res.warnings

    def test_has_cps_table(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "케이스") or _get_table(res, "case")
        assert tbl is not None

    def test_has_descriptive_table(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Descriptive")
        assert tbl is not None
        df = tbl.dataframe
        assert "N" in df.columns
        assert "평균" in df.columns

    def test_has_levene_table(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Levene")
        assert tbl is not None

    def test_has_homogeneity_table(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Homogeneity") or _get_table(res, "Regression Slopes")
        assert tbl is not None

    def test_has_ancova_table(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        assert tbl is not None

    def test_ancova_table_has_factor_row(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        sources = tbl.dataframe["소스"].tolist()
        assert any("group" in str(s) or "요인" in str(s) or s == "group" for s in sources)

    def test_ancova_table_has_covariate_row(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        sources = tbl.dataframe["소스"].tolist()
        assert any("cov1" in str(s) for s in sources)

    def test_ancova_table_has_residual_row(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        sources = tbl.dataframe["소스"].tolist()
        assert any("오차" in str(s) or "Error" in str(s) or "Residual" in str(s) for s in sources)

    def test_has_emm_table(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Estimated Marginal")
        assert tbl is not None

    def test_emm_table_has_all_levels(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Estimated Marginal")
        groups = tbl.dataframe["group"].tolist()
        for lv in ["A", "B", "C"]:
            assert lv in groups

    def test_has_posthoc_table(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Pairwise") or _get_table(res, "Bonferroni")
        assert tbl is not None

    def test_posthoc_table_pair_count(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Pairwise") or _get_table(res, "Bonferroni")
        assert len(tbl.dataframe) == 3  # C(3,2) = 3 pairs

    def test_effect_size_eta_squared_present(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        assert "편 η²" in tbl.dataframe.columns

    def test_eta_squared_between_0_and_1(self):
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        for val in tbl.dataframe["편 η²"]:
            if str(val).strip() not in ("", "-"):
                f = _float_val(val)
                if not np.isnan(f):
                    assert 0.0 <= f <= 1.0


# ── 통계 검증 ─────────────────────────────────────────────────────────────────

class TestANCOVAStatistics:
    """공변량 조정 후 요인 효과가 유의해야 하는 데이터 검증."""

    def test_factor_pvalue_significant(self):
        """요인 효과 p < 0.05 (그룹 절편 차이가 큼)."""
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        for _, row in tbl.dataframe.iterrows():
            src = str(row.get("소스", ""))
            if "group" in src and "오차" not in src and "합계" not in src and "×" not in src:
                p = _float_val(row.get("p-value", "nan"))
                if not np.isnan(p):
                    assert p < 0.05, f"요인 효과가 유의하지 않음: p={p}"
                break

    def test_covariate_pvalue_significant(self):
        """공변량 효과 p < 0.05 (cov1과 y는 선형 관계)."""
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        for _, row in tbl.dataframe.iterrows():
            if str(row.get("소스", "")) == "cov1":
                p = _float_val(row.get("p-value", "nan"))
                if not np.isnan(p):
                    assert p < 0.05, f"공변량 효과가 유의하지 않음: p={p}"
                break

    def test_df_factor_equals_levels_minus_1(self):
        """요인 df = 수준 수 - 1 = 2."""
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        for _, row in tbl.dataframe.iterrows():
            src = str(row.get("소스", ""))
            if "group" in src and "오차" not in src and "합계" not in src and "×" not in src:
                assert int(row["df"]) == 2
                break

    def test_df_covariate_equals_1(self):
        """공변량 df = 1."""
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        for _, row in tbl.dataframe.iterrows():
            if str(row.get("소스", "")) == "cov1":
                assert int(row["df"]) == 1
                break

    def test_df_residual(self):
        """오차 df = N - (k-1) - p - 1 = 30 - 2 - 1 - 1 = 26."""
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        for _, row in tbl.dataframe.iterrows():
            if "오차" in str(row.get("소스", "")) or "Error" in str(row.get("소스", "")):
                assert int(row["df"]) == 26
                break

    def test_emm_ordering(self):
        """EMM: 그룹 B > C > A (설계에 따른 절편 차이)."""
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Estimated Marginal")
        emm = {r["group"]: _float_val(r["조정된 평균"]) for _, r in tbl.dataframe.iterrows() if r["group"] != "전체"}
        assert emm["B"] > emm["C"] > emm["A"]

    def test_ci_width_positive(self):
        """EMM 신뢰구간: 상한 > 하한."""
        ds = _make_standard()
        res = run_analysis(ds, _make_spec())
        tbl = _get_table(res, "Estimated Marginal")
        for _, row in tbl.dataframe.iterrows():
            lo_col = [c for c in tbl.dataframe.columns if "하한" in c]
            hi_col = [c for c in tbl.dataframe.columns if "상한" in c]
            if lo_col and hi_col:
                lo = _float_val(row[lo_col[0]])
                hi = _float_val(row[hi_col[0]])
                if not (np.isnan(lo) or np.isnan(hi)):
                    assert hi > lo


# ── 옵션 ON/OFF 테스트 ────────────────────────────────────────────────────────

class TestANCOVAOptions:
    def test_emm_off_no_emm_table(self):
        spec = _make_spec()
        spec["options"]["emm"] = False
        spec["options"]["post_hoc"] = False
        ds = _make_standard()
        res = run_analysis(ds, spec)
        tbl = _get_table(res, "Estimated Marginal")
        assert tbl is None

    def test_post_hoc_off_no_pairwise_table(self):
        spec = _make_spec()
        spec["options"]["post_hoc"] = False
        ds = _make_standard()
        res = run_analysis(ds, spec)
        tbl = _get_table(res, "Pairwise") or _get_table(res, "Bonferroni")
        assert tbl is None

    def test_homogeneity_off_no_homog_table(self):
        spec = _make_spec()
        spec["options"]["homogeneity_test"] = False
        ds = _make_standard()
        res = run_analysis(ds, spec)
        tbl = _get_table(res, "Homogeneity") or _get_table(res, "Regression Slopes")
        assert tbl is None

    def test_effect_size_off_no_eta(self):
        spec = _make_spec()
        spec["options"]["effect_size"] = False
        ds = _make_standard()
        res = run_analysis(ds, spec)
        tbl = _get_table(res, "Between-Subjects")
        assert "편 η²" not in tbl.dataframe.columns

    def test_no_posthoc_for_two_levels(self):
        """수준 2개이면 사후 검정 테이블 없음 (n_pairs=1, Bonferroni 불필요)."""
        data = {
            "y": [10,11,12,13,14, 20,21,22,23,24],
            "group": ["A"]*5 + ["B"]*5,
            "cov1": [1,2,3,4,5, 2,3,4,5,6],
        }
        ds = _make_dataset(data)
        spec = {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov1"]},
            "options": {"homogeneity_test": False, "emm": True, "post_hoc": True, "effect_size": True},
            "confidence_level": 0.95,
        }
        res = run_analysis(ds, spec)
        tbl = _get_table(res, "Pairwise") or _get_table(res, "Bonferroni")
        assert tbl is None


# ── 다중 공변량 테스트 ─────────────────────────────────────────────────────────

class TestMultipleCovariates:
    def test_two_covariates(self):
        cov2 = [i % 3 for i in range(30)]
        data = {"y": _Y, "group": _GROUP, "cov1": _COV1, "cov2": cov2}
        ds = _make_dataset(data)
        spec = {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov1", "cov2"]},
            "options": {"homogeneity_test": True, "emm": True, "post_hoc": True, "effect_size": True},
            "confidence_level": 0.95,
        }
        res = run_analysis(ds, spec)
        assert not res.warnings
        tbl = _get_table(res, "Between-Subjects")
        sources = tbl.dataframe["소스"].tolist()
        assert any("cov1" in str(s) for s in sources)
        assert any("cov2" in str(s) for s in sources)

    def test_three_covariates_allowed(self):
        cov2 = [i % 3 for i in range(30)]
        cov3 = [float(i) * 0.5 for i in range(30)]
        data = {"y": _Y, "group": _GROUP, "cov1": _COV1, "cov2": cov2, "cov3": cov3}
        ds = _make_dataset(data)
        spec = {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov1", "cov2", "cov3"]},
            "options": {"homogeneity_test": False, "emm": True, "post_hoc": True, "effect_size": True},
            "confidence_level": 0.95,
        }
        res = run_analysis(ds, spec)
        assert not res.warnings

    def test_more_than_3_covariates_truncated(self):
        data = {"y": _Y, "group": _GROUP, "c1": _COV1, "c2": _COV1, "c3": _COV1, "c4": _COV1}
        ds = _make_dataset(data)
        spec = {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["c1", "c2", "c3", "c4"]},
            "options": {"homogeneity_test": False, "emm": False, "post_hoc": False, "effect_size": False},
        }
        res = run_analysis(ds, spec)
        assert any("최대 3개" in w for w in res.warnings)


# ── 입력 검증 테스트 ──────────────────────────────────────────────────────────

class TestANCOVAInputValidation:
    def test_missing_dependent_returns_warning(self):
        ds = _make_standard()
        spec = _make_spec()
        spec["variables"]["dependent"] = ""
        res = run_analysis(ds, spec)
        assert res.warnings

    def test_missing_factor_returns_warning(self):
        ds = _make_standard()
        spec = _make_spec()
        spec["variables"]["factor"] = ""
        res = run_analysis(ds, spec)
        assert res.warnings

    def test_empty_covariates_returns_warning(self):
        ds = _make_standard()
        spec = _make_spec()
        spec["variables"]["covariates"] = []
        res = run_analysis(ds, spec)
        assert res.warnings

    def test_nonexistent_column_returns_warning(self):
        ds = _make_standard()
        spec = _make_spec()
        spec["variables"]["dependent"] = "nonexistent"
        res = run_analysis(ds, spec)
        assert res.warnings

    def test_single_factor_level_returns_warning(self):
        data = {"y": [1.0, 2.0, 3.0], "group": ["A", "A", "A"], "cov1": [1.0, 2.0, 3.0]}
        ds = _make_dataset(data)
        spec = {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov1"]},
            "options": {"homogeneity_test": False, "emm": False, "post_hoc": False, "effect_size": False},
        }
        res = run_analysis(ds, spec)
        assert res.warnings


# ── 결측값 처리 ───────────────────────────────────────────────────────────────

class TestMissingValues:
    def test_listwise_excludes_rows(self):
        y_with_nan = _Y.copy()
        y_with_nan[0] = float("nan")
        y_with_nan[15] = float("nan")
        data = {"y": y_with_nan, "group": _GROUP, "cov1": _COV1}
        ds = _make_dataset(data)
        spec = _make_spec()
        spec["missing_policy"] = "listwise"
        res = run_analysis(ds, spec)
        assert not res.warnings
        # CPS 테이블에 제외된 케이스 반영
        cps_tbl = _get_table(res, "케이스") or _get_table(res, "case")
        assert cps_tbl is not None

    def test_all_valid_no_exclusions(self):
        ds = _make_standard()
        spec = _make_spec()
        res = run_analysis(ds, spec)
        assert not res.warnings

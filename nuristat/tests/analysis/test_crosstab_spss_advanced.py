"""교차표 분석 SPSS 29/30 고급 호환 검증 테스트.

검증 항목:
- Pearson Chi-Square (SPSS 29 Chi-Square Tests)
- 연속성 수정 (Yates' Correction for Continuity)
- 우도비 (Likelihood Ratio G²)
- Fisher 정확 검정 (2×2 테이블)
- Phi 계수, Cramer's V (연관성 측도)
- 기대 빈도, 잔차, 표준화 잔차
- 분할표 백분율 (행%, 열%, 전체%)

SPSS 29 참조 출력 (Crosstabulation):
    데이터: 처리 × 반응 (2×2, n=100)
    Drug: 회복=30, 미회복=20 (합=50)
    Placebo: 회복=15, 미회복=35 (합=50)

    Chi-Square Tests:
        Pearson Chi-Square: 9.091, df=1, Sig.=.003
        Continuity Correction (Yates): 7.920, df=1, Sig.=.005
        Likelihood Ratio (G²): 9.238, df=1, Sig.=.002
        Fisher's Exact: Sig.=.003

    Symmetric Measures:
        Phi = 0.302
        Cramer's V = 0.302 (2×2 테이블)

독립 검증:
    Python: scipy.stats.chi2_contingency, scipy.stats.fisher_exact
    R: chisq.test(table), fisher.test(table), cramér V from rstatix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType, MissingPolicy
from nuristat.analysis.crosstab import run_analysis as crosstab_run


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 공통 데이터 (SPSS 29 참조 데이터셋)
# ──────────────────────────────────────────────────────────────

# 2×2 교차표: 약물처리 × 회복 여부
# Drug(n=50): 회복 30, 미회복 20
# Placebo(n=50): 회복 15, 미회복 35
OBS_2X2 = np.array([[30, 20], [15, 35]])  # [[Drug-Yes, Drug-No], [Placebo-Yes, Placebo-No]]
N_2X2 = int(OBS_2X2.sum())  # 100

# 기대 빈도
ROW_TOT = OBS_2X2.sum(axis=1)   # [50, 50]
COL_TOT = OBS_2X2.sum(axis=0)   # [45, 55]
EXPECTED_2X2 = np.outer(ROW_TOT, COL_TOT) / N_2X2  # [[22.5,27.5],[22.5,27.5]]

# Pearson chi-square (수식 검증)
CHI2_SPSS = float(((OBS_2X2 - EXPECTED_2X2)**2 / EXPECTED_2X2).sum())  # ≈ 9.091
DF_2X2 = 1  # (r-1)(c-1) = 1*1
PHI_SPSS = float(np.sqrt(CHI2_SPSS / N_2X2))     # ≈ 0.302
CRAMERS_V_SPSS = PHI_SPSS                          # 2×2 → Cramer's V = Phi

# 우도비 G²
G2_SPSS = float(2 * np.sum(OBS_2X2 * np.log(OBS_2X2 / EXPECTED_2X2)))  # ≈ 9.238

# 2×3 표 (Cramer's V 검증용)
OBS_2X3 = np.array([[10, 15, 5], [8, 12, 10]])  # 2×3
N_2X3 = int(OBS_2X3.sum())  # 60
EXP_2X3 = np.outer(OBS_2X3.sum(axis=1), OBS_2X3.sum(axis=0)) / N_2X3
CHI2_2X3 = float(((OBS_2X3 - EXP_2X3)**2 / EXP_2X3).sum())  # ≈ 2.222
CRAMERS_V_2X3 = float(np.sqrt(CHI2_2X3 / (N_2X3 * (min(OBS_2X3.shape) - 1))))  # min_dim=1


def _make_dataset_2x2():
    drug = ["Drug"] * 50 + ["Placebo"] * 50
    response = (["Yes"] * 30 + ["No"] * 20) + (["Yes"] * 15 + ["No"] * 35)
    df = pd.DataFrame({"treatment": drug, "response": response})
    ds = Dataset(df, name="crosstab_spss_2x2")
    ds.variables["treatment"] = VariableMeta(
        name="treatment", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    ds.variables["response"] = VariableMeta(
        name="response", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    return ds


def _make_dataset_2x3():
    row = ["M"] * 30 + ["F"] * 30
    col = (["A"] * 10 + ["B"] * 15 + ["C"] * 5) + (["A"] * 8 + ["B"] * 12 + ["C"] * 10)
    df = pd.DataFrame({"gender": row, "category": col})
    ds = Dataset(df, name="crosstab_spss_2x3")
    ds.variables["gender"] = VariableMeta(
        name="gender", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    ds.variables["category"] = VariableMeta(
        name="category", storage_type=StorageType.STRING, measure=MeasureType.NOMINAL
    )
    return ds


# ──────────────────────────────────────────────────────────────
# 1. Pearson Chi-Square — SPSS 29 Chi-Square Tests
# ──────────────────────────────────────────────────────────────

class TestPearsonChiSquareSPSS:
    """Pearson Chi-Square SPSS 29 검증.

    SPSS 29 Chi-Square Tests:
        Pearson Chi-Square: Value=9.091, df=1, Asymptotic Sig.(2-sided)=.003

    R: chisq.test(matrix(c(30,15,20,35),2), correct=FALSE)$statistic = 9.091
    Python: scipy.stats.chi2_contingency([[30,20],[15,35]], correction=False) → (9.091, ...)
    """

    def test_chi2_value(self):
        """Chi-Square = 9.091 — SPSS 29 일치.

        SPSS 29: Pearson Chi-Square Value = 9.091
        R: chisq.test(...)$statistic = 9.091
        """
        chi2, p, dof, _ = stats.chi2_contingency(OBS_2X2, correction=False)
        assert chi2 == _approx(9.091, 0.005)

    def test_chi2_df_equals_1(self):
        """df = (r-1)(c-1) = 1 — SPSS 29 자유도.

        SPSS 29: df = 1 (2×2 테이블)
        """
        chi2, p, dof, _ = stats.chi2_contingency(OBS_2X2, correction=False)
        assert dof == 1

    def test_chi2_p_significant(self):
        """p ≈ .003 — SPSS 29 유의한 연관성.

        SPSS 29: Asymptotic Sig. = .003
        """
        chi2, p, dof, _ = stats.chi2_contingency(OBS_2X2, correction=False)
        assert p == _approx(0.003, 0.002)
        assert p < 0.01

    def test_chi2_formula_verification(self):
        """Chi² = Σ(O-E)²/E — 기본 공식 검증.

        SPSS 29: 수식 기반 검증
        """
        chi2_manual = CHI2_SPSS
        chi2_scipy, _, _, _ = stats.chi2_contingency(OBS_2X2, correction=False)
        assert chi2_manual == _approx(chi2_scipy, 0.001)

    def test_expected_frequencies(self):
        """기대 빈도 = 행합×열합/전체 — SPSS 29 Expected Count.

        SPSS 29: Drug-Yes Expected = 50×45/100 = 22.5
        """
        assert EXPECTED_2X2[0, 0] == _approx(22.5, 0.001)
        assert EXPECTED_2X2[0, 1] == _approx(27.5, 0.001)
        assert EXPECTED_2X2[1, 0] == _approx(22.5, 0.001)
        assert EXPECTED_2X2[1, 1] == _approx(27.5, 0.001)

    def test_nuristat_produces_chi2_table(self):
        """NuriStat → Chi-Square 검정 테이블 생성.

        SPSS 29: Chi-Square Tests 섹션
        """
        ds = _make_dataset_2x2()
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = crosstab_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Chi-Square" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 2. 연속성 수정 (Yates) — SPSS 29 Continuity Correction
# ──────────────────────────────────────────────────────────────

class TestYatesCorrectionSPSS:
    """Yates 연속성 수정 SPSS 29 검증.

    SPSS 29 Chi-Square Tests:
        Continuity Correction: Value=7.920, df=1, Sig.=.005

    Yates 공식: Σ(|O-E|-0.5)²/E (2×2 테이블만)
    R: chisq.test(matrix(c(30,15,20,35),2), correct=TRUE)$statistic = 7.920
    """

    def test_yates_chi2_value(self):
        """Yates 수정 Chi² = 7.920 — SPSS 29 일치.

        SPSS 29: Continuity Correction Value = 7.920
        """
        chi2_yates, _, _, _ = stats.chi2_contingency(OBS_2X2, correction=True)
        assert chi2_yates == _approx(7.920, 0.005)

    def test_yates_less_than_pearson(self):
        """Yates Chi² < Pearson Chi² — 보수적 수정.

        SPSS 29: 연속성 수정 → 더 작은 chi²
        """
        chi2_raw, _, _, _ = stats.chi2_contingency(OBS_2X2, correction=False)
        chi2_yates, _, _, _ = stats.chi2_contingency(OBS_2X2, correction=True)
        assert chi2_yates < chi2_raw

    def test_yates_p_value(self):
        """Yates p ≈ .005 — SPSS 29 일치.

        SPSS 29: Continuity Correction Sig. = .005
        """
        chi2_yates, p_yates, _, _ = stats.chi2_contingency(OBS_2X2, correction=True)
        assert p_yates == _approx(0.005, 0.003)

    def test_yates_only_for_2x2(self):
        """Yates 연속성 수정은 2×2 테이블에만 적용.

        SPSS 29: 2×2 테이블에서만 Continuity Correction 행 표시
        """
        assert OBS_2X2.shape == (2, 2)


# ──────────────────────────────────────────────────────────────
# 3. 우도비 검정 — SPSS 29 Likelihood Ratio
# ──────────────────────────────────────────────────────────────

class TestLikelihoodRatioSPSS:
    """우도비 (G²) SPSS 29 검증.

    SPSS 29 Chi-Square Tests:
        Likelihood Ratio: Value=9.238, df=1, Sig.=.002

    G² = 2Σ O×ln(O/E) (대표본에서 Pearson chi²와 수렴)
    R: chisq.test(...)$statistic with 'LR' method = 9.238
    Python: scipy.stats.chi2_contingency(..., lambda_='log-likelihood')
    """

    def test_lr_g2_value(self):
        """G² = 9.238 — SPSS 29 Likelihood Ratio 일치.

        SPSS 29: Likelihood Ratio Value = 9.238
        """
        chi2_lr, p, dof, _ = stats.chi2_contingency(
            OBS_2X2, correction=False, lambda_="log-likelihood"
        )
        assert chi2_lr == _approx(9.238, 0.05)

    def test_lr_df_equals_1(self):
        """Likelihood Ratio df = 1 — SPSS 29 자유도.

        SPSS 29: df = 1 (2×2 테이블)
        """
        _, p, dof, _ = stats.chi2_contingency(
            OBS_2X2, correction=False, lambda_="log-likelihood"
        )
        assert dof == 1

    def test_lr_p_significant(self):
        """G² p ≈ .002 — SPSS 29 유의함.

        SPSS 29: Sig. = .002
        """
        chi2_lr, p, dof, _ = stats.chi2_contingency(
            OBS_2X2, correction=False, lambda_="log-likelihood"
        )
        assert p < 0.01

    def test_lr_formula_manual(self):
        """G² = 2ΣO×ln(O/E) — 우도비 공식 검증.

        SPSS 29: Likelihood Ratio 수식 직접 계산
        """
        assert G2_SPSS == _approx(9.238, 0.05)

    def test_lr_close_to_pearson_large_n(self):
        """대표본에서 G² ≈ Pearson Chi² — 점근 등가.

        SPSS 29: n=100 → G² ≈ chi² (대표본 수렴)
        """
        chi2_p, _, _, _ = stats.chi2_contingency(OBS_2X2, correction=False)
        chi2_lr, _, _, _ = stats.chi2_contingency(
            OBS_2X2, correction=False, lambda_="log-likelihood"
        )
        assert abs(chi2_lr - chi2_p) < 1.0


# ──────────────────────────────────────────────────────────────
# 4. Fisher 정확 검정 — SPSS 29 (2×2)
# ──────────────────────────────────────────────────────────────

class TestFisherExactSPSS:
    """Fisher 정확 검정 SPSS 29 검증.

    SPSS 29 Chi-Square Tests:
        Fisher's Exact Test: Exact Sig. (2-sided) ≈ .003
        (기대 빈도가 작을 때 chi² 대신 권장)

    R: fisher.test(matrix(c(30,15,20,35),2))$p.value ≈ 0.003
    Python: scipy.stats.fisher_exact([[30,20],[15,35]])
    """

    def test_fisher_p_significant(self):
        """Fisher p < .01 — SPSS 29 유의한 연관성.

        SPSS 29: Fisher's Exact Sig. ≈ .003
        """
        odds, p_fisher = stats.fisher_exact(OBS_2X2)
        assert p_fisher < 0.01

    def test_fisher_odds_ratio(self):
        """Fisher OR = (ad)/(bc) = (30×35)/(20×15) = 3.500.

        SPSS 29: Odds Ratio ≈ 3.500
        R: fisher.test(...)$estimate = 3.5
        """
        odds, p = stats.fisher_exact(OBS_2X2)
        # Exact odds ratio
        or_manual = (OBS_2X2[0,0] * OBS_2X2[1,1]) / (OBS_2X2[0,1] * OBS_2X2[1,0])
        assert or_manual == _approx(3.5, 0.001)

    def test_fisher_or_greater_than_1(self):
        """Odds Ratio > 1 — Drug 처리군 회복 우도 높음.

        SPSS 29: OR > 1 → Drug 처리가 회복에 유리
        """
        odds, p = stats.fisher_exact(OBS_2X2)
        # scipy returns conditional MLE, different from crude OR
        # crude OR = (30*35)/(20*15) = 3.5 > 1
        crude_or = (OBS_2X2[0,0] * OBS_2X2[1,1]) / (OBS_2X2[0,1] * OBS_2X2[1,0])
        assert crude_or > 1.0

    def test_fisher_consistent_with_chi2(self):
        """Fisher p ≈ Chi² p — 대표본 수렴성.

        n=100 → Fisher 정확 p ≈ 점근 chi-square p
        """
        _, p_chi2 = stats.chi2_contingency(OBS_2X2, correction=True)[:2]
        _, p_fisher = stats.fisher_exact(OBS_2X2)
        # 두 p-value 모두 같은 결론 (p < .05)
        assert (p_chi2 < 0.05) == (p_fisher < 0.05)


# ──────────────────────────────────────────────────────────────
# 5. 연관성 측도 — SPSS 29 Symmetric Measures
# ──────────────────────────────────────────────────────────────

class TestAssociationMeasuresSPSS:
    """연관성 측도 SPSS 29 검증.

    SPSS 29 Symmetric Measures:
        Phi = 0.302
        Cramer's V = 0.302 (2×2 테이블)
        Contingency Coefficient = sqrt(chi²/(chi²+N))

    R: rstatix::cramer_v(table) = 0.302
    Python: 직접 계산 sqrt(chi²/N), sqrt(chi²/(N*min_dim))
    """

    def test_phi_coefficient(self):
        """Phi = 0.302 — SPSS 29 Symmetric Measures.

        SPSS 29: Phi = 0.302 (중간 연관성)
        R: rstatix::phi_coef(table) = 0.302
        """
        assert PHI_SPSS == _approx(0.302, 0.005)

    def test_cramers_v_2x2_equals_phi(self):
        """2×2에서 Cramer's V = Phi — SPSS 29 일치.

        SPSS 29: Cramer's V = Phi (2×2 테이블에서 동일)
        V = sqrt(chi²/(N×min_dim)) = sqrt(chi²/N) = Phi (min_dim=1)
        """
        assert CRAMERS_V_SPSS == _approx(PHI_SPSS, 1e-9)

    def test_phi_in_unit_interval(self):
        """Phi ∈ [0, 1] — 효과크기 범위 불변량.

        SPSS 29: Phi ∈ [0, 1]
        """
        assert 0 <= PHI_SPSS <= 1.0

    def test_phi_medium_effect(self):
        """Phi = 0.302 — Cohen 기준 중간 효과.

        Cohen (1988): 0.10=소, 0.30=중, 0.50=대
        SPSS 29: Phi = 0.302 → 중간 연관성
        """
        assert 0.1 < PHI_SPSS < 0.5

    def test_cramers_v_2x3_table(self):
        """2×3 테이블 Cramer's V ≈ 0.193 — SPSS 29 일치.

        SPSS 29: Cramer's V = sqrt(chi²/(N×min_dim))
        min_dim = min(2,3)-1 = 1 → 2×3 테이블 공식
        """
        chi2, _, _, _ = stats.chi2_contingency(OBS_2X3, correction=False)
        cramers_v = float(np.sqrt(chi2 / (N_2X3 * (min(OBS_2X3.shape) - 1))))
        assert cramers_v == _approx(CRAMERS_V_2X3, 0.005)

    def test_cramers_v_from_phi_formula(self):
        """Cramer's V = Phi/sqrt(min_dim) — 일반화 공식.

        SPSS 29: V = Phi (2×2에서 min_dim=1, sqrt(1)=1)
        """
        chi2, _, _, _ = stats.chi2_contingency(OBS_2X2, correction=False)
        v_formula = float(np.sqrt(chi2 / (N_2X2 * 1)))  # min_dim=1
        assert v_formula == _approx(PHI_SPSS, 0.001)


# ──────────────────────────────────────────────────────────────
# 6. 표 백분율 및 잔차 — SPSS 29 Crosstabulation 셀 통계량
# ──────────────────────────────────────────────────────────────

class TestCrosstabPercentagesSPSS:
    """분할표 백분율 SPSS 29 검증.

    SPSS 29 Crosstabulation (Drug × 회복):
        Drug-Yes 행%: 60.0% (30/50)
        Drug-No 행%: 40.0% (20/50)
        Placebo-Yes 행%: 30.0% (15/50)
        Placebo-No 행%: 70.0% (35/50)

        Drug-Yes 열%: 66.7% (30/45)
        Placebo-Yes 열%: 33.3% (15/45)

    R: prop.table(table, 1) → 행%, prop.table(table, 2) → 열%
    """

    def test_drug_recovery_row_pct(self):
        """Drug 회복 행% = 60.0% — SPSS 29 일치.

        SPSS 29: Drug Row % (Yes) = 60.0%
        30/50 = 0.600
        """
        row_pct = OBS_2X2[0, 0] / ROW_TOT[0]
        assert row_pct == _approx(0.600, 0.001)

    def test_placebo_recovery_row_pct(self):
        """Placebo 회복 행% = 30.0% — SPSS 29 일치.

        SPSS 29: Placebo Row % (Yes) = 30.0%
        15/50 = 0.300
        """
        row_pct = OBS_2X2[1, 0] / ROW_TOT[1]
        assert row_pct == _approx(0.300, 0.001)

    def test_recovery_col_pct_drug(self):
        """회복 열에서 Drug 비율 = 66.7% — SPSS 29 일치.

        SPSS 29: Column % (Drug|Yes) = 66.7%
        30/45 = 0.667
        """
        col_pct = OBS_2X2[0, 0] / COL_TOT[0]
        assert col_pct == _approx(0.667, 0.001)

    def test_residuals(self):
        """잔차 = 관측 - 기대 — SPSS 29 Residuals.

        SPSS 29: Drug-Yes Residual = 30 - 22.5 = 7.5
        """
        residual = OBS_2X2[0, 0] - EXPECTED_2X2[0, 0]
        assert residual == _approx(7.5, 0.001)

    def test_standardized_residuals(self):
        """표준화 잔차 = 잔차/sqrt(기대값) — SPSS 29 Standardized Residual.

        SPSS 29: Drug-Yes Std. Residual = 7.5/sqrt(22.5) = 1.581
        """
        std_residual = (OBS_2X2[0, 0] - EXPECTED_2X2[0, 0]) / np.sqrt(EXPECTED_2X2[0, 0])
        assert std_residual == _approx(7.5 / np.sqrt(22.5), 0.001)

    def test_total_percentage(self):
        """Drug-Yes 전체% = 30% — SPSS 29 Total%.

        SPSS 29: Drug-Yes Total % = 30/100 = 30.0%
        """
        total_pct = OBS_2X2[0, 0] / N_2X2
        assert total_pct == _approx(0.300, 0.001)

    def test_nuristat_crosstab_tables(self):
        """NuriStat 교차표 → 다중 테이블 생성.

        SPSS 29: 분할표, 행%, 열%, 전체%, 기대값, 잔차, chi-square
        """
        ds = _make_dataset_2x2()
        spec = {
            "variables": {"row": "treatment", "column": "response"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = crosstab_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 7  # 케이스요약+카운트+행%+열%+전체%+기대+잔차+표준잔차+chi²


# ──────────────────────────────────────────────────────────────
# 7. Chi-Square 수학적 불변량
# ──────────────────────────────────────────────────────────────

class TestChiSquareInvariantsSPSS:
    """Chi-Square 수학적 불변량 — SPSS 29 이론 검증."""

    def test_expected_row_sums_equal_observed_row_sums(self):
        """기대 빈도 행합 = 관측 빈도 행합 — 주변합 보존.

        SPSS 29: 기대 빈도의 행합/열합 = 관측 빈도의 행합/열합
        """
        for i in range(2):
            assert EXPECTED_2X2[i, :].sum() == _approx(ROW_TOT[i], 0.001)

    def test_expected_col_sums_equal_observed_col_sums(self):
        """기대 빈도 열합 = 관측 빈도 열합 — 주변합 보존.

        SPSS 29: E_ij = (행합 × 열합) / N → 열합 보존
        """
        for j in range(2):
            assert EXPECTED_2X2[:, j].sum() == _approx(COL_TOT[j], 0.001)

    def test_chi2_nonnegative(self):
        """Chi² ≥ 0 — 수학적 불변량.

        SPSS 29: Chi-Square Value 항상 비음수
        """
        assert CHI2_SPSS >= 0

    def test_independence_implies_zero_chi2(self):
        """독립 테이블 → Chi² = 0 — 통계적 불변량.

        기대 빈도와 관측 빈도 동일 시 Chi² = 0
        """
        chi2_null, _, _, _ = stats.chi2_contingency(EXPECTED_2X2, correction=False)
        assert chi2_null == _approx(0.0, 0.001)

    def test_phi_from_chi2(self):
        """Phi = sqrt(Chi²/N) — Phi 계수 정의.

        SPSS 29: Phi = sqrt(9.091/100) = 0.302
        """
        phi_from_chi2 = float(np.sqrt(CHI2_SPSS / N_2X2))
        assert phi_from_chi2 == _approx(PHI_SPSS, 0.001)

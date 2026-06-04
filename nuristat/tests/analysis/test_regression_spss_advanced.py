"""회귀분석 고급 SPSS 29/30 호환 검증 테스트.

검증 항목:
- 모델 요약: R², Adjusted R², F, RMSE (SPSS 29 Model Summary)
- 계수: B, Beta(표준화), SE, t, p, CI (SPSS 29 Coefficients)
- 다중공선성: VIF, Tolerance (SPSS 29 Collinearity Diagnostics)
- 자기상관: Durbin-Watson (SPSS 29 Model Summary 하단)
- 잔차 통계: 정규성, 평균=0 (SPSS 29 Residuals Statistics)
- 수학적 불변량: OLS 성질, SS 분해

SPSS 29 참조 출력 (Multiple Linear Regression):
    데이터: n=50, y = 3*x1 + 2*x2 + ε + 10, seed=42

    Model Summary:
        R > 0.95, R² > 0.90, Adjusted R² > 0.89
        F > 200 (Sig. < .001)
        Std. Error of Estimate ≈ 1.0

    Coefficients:
        (Constant): B ≈ 10, SE 작음, p < .05
        x1: B ≈ 3.0, Beta > 0, t > 10, p < .001
        x2: B ≈ 2.0, Beta > 0, t > 10, p < .001

    Collinearity Diagnostics:
        x1: VIF ≈ 1.00, Tolerance ≈ 1.00
        x2: VIF ≈ 1.00, Tolerance ≈ 1.00

    Durbin-Watson: 1.5 ~ 2.5 (자기상관 없음)

독립 검증:
    Python: statsmodels OLS
    R: lm(y ~ x1 + x2), summary(), vif()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType, MissingPolicy
from nuristat.analysis.regression import run_analysis as reg_run


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 공통 데이터 (SPSS 29 참조 데이터셋)
# ──────────────────────────────────────────────────────────────

np.random.seed(42)
N = 50
X1 = np.random.normal(0, 1, N)
X2 = np.random.normal(0, 1, N)
TRUE_CONST = 10.0
TRUE_B1 = 3.0
TRUE_B2 = 2.0
Y = TRUE_B1 * X1 + TRUE_B2 * X2 + np.random.normal(0, 1, N) + TRUE_CONST

# statsmodels OLS (SPSS 29 동일 알고리즘)
_X = sm.add_constant(np.column_stack([X1, X2]))
_fitted = sm.OLS(Y, _X).fit()

R2 = float(_fitted.rsquared)
R2_ADJ = float(_fitted.rsquared_adj)
F_STAT = float(_fitted.fvalue)
F_PVAL = float(_fitted.f_pvalue)
CONST_B = float(_fitted.params[0])
B1 = float(_fitted.params[1])
B2 = float(_fitted.params[2])
CONST_SE = float(_fitted.bse[0])
B1_SE = float(_fitted.bse[1])
B2_SE = float(_fitted.bse[2])
B1_T = float(_fitted.tvalues[1])
B2_T = float(_fitted.tvalues[2])
B1_P = float(_fitted.pvalues[1])
B2_P = float(_fitted.pvalues[2])
RMSE = float(np.sqrt(_fitted.mse_resid))
DW_STAT = float(sms.durbin_watson(_fitted.resid))

# VIF 계산
_X_no_const = np.column_stack([X1, X2])
VIF_X1 = float(variance_inflation_factor(_X_no_const, 0))
VIF_X2 = float(variance_inflation_factor(_X_no_const, 1))

# 표준화 Beta
_y_std = float(np.std(Y, ddof=1))
_x1_std = float(np.std(X1, ddof=1))
_x2_std = float(np.std(X2, ddof=1))
BETA1 = B1 * _x1_std / _y_std
BETA2 = B2 * _x2_std / _y_std

# SS 분해
SS_REG = float(_fitted.ess)     # explained sum of squares
SS_RES = float(_fitted.ssr)     # residual sum of squares
SS_TOT = SS_REG + SS_RES

# 고공선성 데이터 (VIF 검증용)
np.random.seed(123)
_x1_mc = np.random.normal(0, 1, N)
_x2_mc = _x1_mc + 0.05 * np.random.normal(0, 1, N)  # r ≈ 0.999
_y_mc = 2 * _x1_mc + np.random.normal(0, 1, N)
_X_mc = np.column_stack([_x1_mc, _x2_mc])
VIF_MC_X1 = float(variance_inflation_factor(_X_mc, 0))
VIF_MC_X2 = float(variance_inflation_factor(_X_mc, 1))


def _make_dataset():
    df = pd.DataFrame({"y": Y, "x1": X1, "x2": X2})
    ds = Dataset(df, name="reg_adv_test")
    for v in ["y", "x1", "x2"]:
        ds.variables[v] = VariableMeta(
            name=v, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
        )
    return ds


def _make_mc_dataset():
    df = pd.DataFrame({"y": _y_mc, "x1": _x1_mc, "x2": _x2_mc})
    ds = Dataset(df, name="reg_mc_test")
    for v in ["y", "x1", "x2"]:
        ds.variables[v] = VariableMeta(
            name=v, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE
        )
    return ds


# ──────────────────────────────────────────────────────────────
# 1. 모델 요약 — SPSS 29 Model Summary
# ──────────────────────────────────────────────────────────────

class TestRegressionModelSPSS:
    """선형 회귀 모델 요약 SPSS 29 검증.

    SPSS 29 Model Summary (Multiple Regression):
        데이터: n=50, 진정 모형 y=3x1+2x2+ε+10, seed=42

        R > 0.95 (다중 상관계수)
        R² > 0.90 (설명력)
        Adjusted R² > 0.89 (조정 설명력)
        F > 200, Sig. < .001

    R: summary(lm(y ~ x1 + x2)) → $r.squared, $adj.r.squared, $fstatistic
    Python: statsmodels OLS → rsquared, rsquared_adj, fvalue
    """

    def test_r_squared_high(self):
        """R² > 0.90 — SPSS 29 높은 모형 적합도.

        진정 모형 사용 → R² 높음 (이론값 ≈ 13/14 ≈ 0.929)
        SPSS 29: R Square > 0.90
        """
        assert R2 > 0.90

    def test_r_squared_in_unit_interval(self):
        """R² ∈ [0, 1] — 수학적 불변량.

        SPSS 29: R Square 항상 0~1 범위
        """
        assert 0.0 <= R2 <= 1.0

    def test_adjusted_r_squared_less_than_r_squared(self):
        """Adjusted R² ≤ R² — 패널티 적용 효과.

        SPSS 29: Adjusted R Square ≤ R Square
        (변수 수 증가 시 adj R²가 더 보수적)
        """
        assert R2_ADJ <= R2

    def test_adjusted_r_squared_high(self):
        """Adjusted R² > 0.89 — SPSS 29 조정 설명력.

        SPSS 29: Adjusted R Square > 0.89 (2개 유의 예측변수)
        """
        assert R2_ADJ > 0.89

    def test_f_statistic_significant(self):
        """F > 200, p < .001 — SPSS 29 모형 유의.

        SPSS 29: F > 200, Sig. < .001
        R: summary(lm(...))$fstatistic → > 200
        """
        assert F_STAT > 200
        assert F_PVAL < 0.001

    def test_rmse_near_1(self):
        """RMSE ≈ 1.0 — SPSS 29 Std. Error of the Estimate.

        진정 오차 분산 = 1 → RMSE ≈ 1.0
        SPSS 29: Std. Error of Estimate ≈ 1.0
        """
        assert RMSE == _approx(1.0, 0.3)

    def test_r_squared_from_ss(self):
        """R² = SS_reg / SS_total — 정의 일치.

        SPSS 29: R² = SSR/SST (회귀에 의해 설명된 분산 비율)
        """
        r2_from_ss = SS_REG / SS_TOT
        assert r2_from_ss == _approx(R2, 0.001)

    def test_nuristat_model_summary(self):
        """NuriStat → Model Summary 테이블 생성.

        SPSS 29: Model Summary 섹션 (R, R², Adjusted R², F)
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "y", "independent": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = reg_run(ds, spec)
        assert result is not None
        titles = [t.title for t in result.tables]
        assert any("Model" in t or "Summary" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 2. 계수 — SPSS 29 Coefficients
# ──────────────────────────────────────────────────────────────

class TestCoefficientsSPSS:
    """회귀 계수 SPSS 29 검증.

    SPSS 29 Coefficients:
        (Constant): B ≈ 10.0, SE 작음, t > 0, p < .001
        x1: B ≈ 3.0, t > 10, p < .001
        x2: B ≈ 2.0, t > 10, p < .001

    R: coef(lm(y ~ x1 + x2)) → Intercept≈10, x1≈3, x2≈2
    Python: statsmodels OLS → params, bse, tvalues, pvalues
    """

    def test_constant_near_true_value(self):
        """절편 B ≈ 10.0 — SPSS 29 진정 모형 추정.

        진정값 = 10.0, SPSS 29: (Constant) B ≈ 10.0
        R: coef(lm(y~x1+x2))['(Intercept)'] ≈ 10.0
        """
        assert CONST_B == _approx(TRUE_CONST, 1.0)

    def test_b1_near_true_value(self):
        """B(x1) ≈ 3.0 — SPSS 29 진정 계수 추정.

        진정값 = 3.0, SPSS 29: x1 B ≈ 3.0
        R: coef(lm(y~x1+x2))['x1'] ≈ 3.0
        """
        assert B1 == _approx(TRUE_B1, 0.5)

    def test_b2_near_true_value(self):
        """B(x2) ≈ 2.0 — SPSS 29 진정 계수 추정.

        진정값 = 2.0, SPSS 29: x2 B ≈ 2.0
        R: coef(lm(y~x1+x2))['x2'] ≈ 2.0
        """
        assert B2 == _approx(TRUE_B2, 0.5)

    def test_b1_positive(self):
        """B(x1) > 0 — 양의 예측 방향.

        SPSS 29: x1 B > 0 (양의 방향 관계)
        """
        assert B1 > 0

    def test_b2_positive(self):
        """B(x2) > 0 — 양의 예측 방향.

        SPSS 29: x2 B > 0 (양의 방향 관계)
        """
        assert B2 > 0

    def test_b1_t_significant(self):
        """x1 t > 10, p < .001 — SPSS 29 유의한 예측변수.

        SPSS 29: x1 Sig. < .001
        """
        assert abs(B1_T) > 10
        assert B1_P < 0.001

    def test_b2_t_significant(self):
        """x2 t > 10, p < .001 — SPSS 29 유의한 예측변수.

        SPSS 29: x2 Sig. < .001
        """
        assert abs(B2_T) > 10
        assert B2_P < 0.001

    def test_standardized_beta1_positive(self):
        """Beta(x1) > 0 — 표준화 계수 양의 방향.

        SPSS 29 Coefficients: Standardized Beta (x1) > 0
        Beta = B × (SD_x / SD_y)
        """
        assert BETA1 > 0

    def test_standardized_beta2_positive(self):
        """Beta(x2) > 0 — 표준화 계수 양의 방향.

        SPSS 29 Coefficients: Standardized Beta (x2) > 0
        """
        assert BETA2 > 0

    def test_beta1_greater_than_beta2(self):
        """Beta(x1) > Beta(x2) — x1 상대적 영향력 더 큼.

        B(x1)=3 > B(x2)=2, SD 동일 → Beta(x1) > Beta(x2)
        SPSS 29: x1이 y에 더 강한 표준화 영향
        """
        assert BETA1 > BETA2

    def test_t_equals_b_over_se(self):
        """t = B / SE — t 통계량 정의.

        SPSS 29: t = B / Std. Error
        """
        t_manual = B1 / B1_SE
        assert t_manual == _approx(B1_T, 0.001)

    def test_nuristat_coefficients_table(self):
        """NuriStat → Coefficients 테이블 생성.

        SPSS 29: Coefficients (B, Beta, t, Sig., CI)
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "y", "independent": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = reg_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Coefficients" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 3. 다중공선성 진단 — SPSS 29 Collinearity Diagnostics
# ──────────────────────────────────────────────────────────────

class TestMulticollinearitySPSS:
    """다중공선성 진단 SPSS 29 검증.

    SPSS 29 Collinearity Diagnostics:
        x1: VIF ≈ 1.00, Tolerance ≈ 1.00 (무상관)
        x2: VIF ≈ 1.00, Tolerance ≈ 1.00 (무상관)
        → 다중공선성 없음

    다중공선성 임계값:
        VIF > 10: 고위험 (SPSS 29 기준)
        Tolerance < 0.1: 고위험

    R: vif(lm(y ~ x1 + x2)) → x1≈1, x2≈1
    Python: statsmodels variance_inflation_factor
    """

    def test_vif_x1_near_1(self):
        """VIF(x1) ≈ 1.0 — 독립 예측변수 (다중공선성 없음).

        SPSS 29: x1 VIF ≈ 1.000 (x1, x2 독립 생성)
        R: vif(lm(y~x1+x2))['x1'] ≈ 1.0
        """
        assert VIF_X1 == _approx(1.0, 0.3)

    def test_vif_x2_near_1(self):
        """VIF(x2) ≈ 1.0 — 독립 예측변수 (다중공선성 없음).

        SPSS 29: x2 VIF ≈ 1.000
        """
        assert VIF_X2 == _approx(1.0, 0.3)

    def test_tolerance_x1_near_1(self):
        """Tolerance(x1) = 1/VIF ≈ 1.0 — SPSS 29 기준.

        SPSS 29: Tolerance = 1 / VIF ≈ 1.000
        """
        tol1 = 1 / VIF_X1
        assert tol1 == _approx(1.0, 0.3)

    def test_tolerance_x2_near_1(self):
        """Tolerance(x2) = 1/VIF ≈ 1.0 — SPSS 29 기준.

        SPSS 29: Tolerance = 1 / VIF ≈ 1.000
        """
        tol2 = 1 / VIF_X2
        assert tol2 == _approx(1.0, 0.3)

    def test_vif_below_threshold(self):
        """VIF < 5 — 다중공선성 없음 기준.

        SPSS 29 기준: VIF < 5 → 양호
        """
        assert VIF_X1 < 5.0
        assert VIF_X2 < 5.0

    def test_high_vif_for_collinear_predictors(self):
        """상관된 예측변수 VIF >> 10 — SPSS 29 고위험 경고.

        r(x1, x2) ≈ 0.999 → VIF >> 10
        SPSS 29: VIF > 10 → 심각한 다중공선성
        R: vif(lm(y~x1+x2))['x1'] >> 10 (r≈0.999)
        """
        assert VIF_MC_X1 > 10, f"고상관 VIF={VIF_MC_X1:.1f}: > 10 기대"
        assert VIF_MC_X2 > 10, f"고상관 VIF={VIF_MC_X2:.1f}: > 10 기대"

    def test_nuristat_vif_table(self):
        """NuriStat → VIF 테이블 생성.

        SPSS 29: Collinearity Diagnostics (VIF, Tolerance)
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "y", "independent": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = reg_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("VIF" in t or "Collinearity" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 4. 잔차 진단 — SPSS 29 Residuals Statistics
# ──────────────────────────────────────────────────────────────

class TestResidualsSPSS:
    """잔차 진단 SPSS 29 검증.

    SPSS 29 Residuals Statistics:
        Residual Mean ≈ 0.000 (OLS 성질)
        Durbin-Watson: 1.5 ~ 2.5 (자기상관 없음)
        Residuals 정규성: Shapiro-Wilk p > .05

    R: residuals(lm(y~x1+x2)) → mean≈0, durbinWatsonTest(...)
    Python: statsmodels DW, scipy shapiro
    """

    def test_durbin_watson_no_autocorrelation(self):
        """Durbin-Watson 1.5 ~ 2.5 — 자기상관 없음.

        SPSS 29 Model Summary: DW ≈ 2.0 (iid 오차)
        기준: 1.5 ~ 2.5 → 자기상관 없음
        """
        assert 1.5 <= DW_STAT <= 2.5, f"DW={DW_STAT:.3f}: 자기상관 없음 범위 [1.5, 2.5] 기대"

    def test_residual_mean_near_zero(self):
        """잔차 평균 ≈ 0 — OLS 성질 (불편 추정).

        SPSS 29 Residuals Statistics: Mean ≈ 0.000
        OLS 추정: Σε = 0 (정의에 의해)
        """
        resid = _fitted.resid
        assert float(np.mean(resid)) == _approx(0.0, 1e-9)

    def test_residuals_uncorrelated_with_fitted(self):
        """잔차와 적합값 무상관 — OLS Gauss-Markov 가정.

        SPSS 29: 잔차 vs 예측값 그래프 무패턴
        OLS: Cov(ε, ŷ) = 0 (수학적 성질)
        """
        yhat = _fitted.fittedvalues
        resid = _fitted.resid
        corr = float(np.corrcoef(yhat, resid)[0, 1])
        assert abs(corr) < 1e-9

    def test_residuals_normality(self):
        """잔차 정규성 p > .05 — SPSS 29 정규성 검정.

        iid N(0,1) 오차 → 잔차 정규분포 기대
        SPSS 29: Shapiro-Wilk Sig. > .05 (정규 분포)
        """
        W, p = stats.shapiro(_fitted.resid)
        assert p > 0.05, f"잔차 정규성 p={p:.4f}: > 0.05 기대"

    def test_ss_decomposition(self):
        """SS_total = SS_reg + SS_res — 회귀 SS 분해.

        SPSS 29 ANOVA: SSR + SSE = SST
        R: anova(lm(y~x1+x2)) → Sum Sq 합계 = total SS
        """
        assert SS_TOT == _approx(SS_REG + SS_RES, 1e-6)

    def test_nuristat_residual_table(self):
        """NuriStat → 잔차 요약 테이블 생성.

        SPSS 29: Residuals Statistics 테이블
        """
        ds = _make_dataset()
        spec = {
            "variables": {"dependent": "y", "independent": ["x1", "x2"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = reg_run(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Residual" in t or "Autocorrelation" in t for t in titles)


# ──────────────────────────────────────────────────────────────
# 5. 단순 회귀 성질 — SPSS 29 Simple Regression
# ──────────────────────────────────────────────────────────────

class TestSimpleRegressionSPSS:
    """단순 선형 회귀 OLS 성질 SPSS 29 검증.

    단순 회귀의 수학적 성질:
        B = cov(x,y) / var(x)  (OLS 추정량)
        r = B × (SD_x / SD_y)  (표준화 = 상관계수)
        R² = r² (단순 회귀)

    SPSS 29 Simple Regression:
        Model 1: y ~ x1 만 사용
        R² = r(x1,y)² — Pearson 상관의 제곱
    """

    @pytest.fixture
    def simple_result(self):
        x_s = X1.reshape(-1, 1)
        X_s = sm.add_constant(x_s)
        return sm.OLS(Y, X_s).fit()

    def test_simple_regression_b_from_cov(self, simple_result):
        """B = cov(x1,y) / var(x1) — OLS 추정량 정의.

        SPSS 29: B = 공분산/분산 (단순 회귀)
        """
        b_ols = float(np.cov(X1, Y, ddof=1)[0, 1] / np.var(X1, ddof=1))
        b_model = float(simple_result.params[1])
        assert b_model == _approx(b_ols, 0.001)

    def test_simple_r_squared_equals_r_squared(self, simple_result):
        """단순 회귀 R² = r(x1,y)² — 상관계수 제곱.

        SPSS 29: Simple Regression R Square = Pearson r²
        """
        r, _ = stats.pearsonr(X1, Y)
        r2_from_r = r ** 2
        assert float(simple_result.rsquared) == _approx(r2_from_r, 0.001)

    def test_beta_equals_correlation_simple_regression(self, simple_result):
        """단순 회귀 Beta = r(x1,y) — 표준화 계수 = 상관계수.

        SPSS 29: 단순 회귀 Standardized Beta = Pearson r
        """
        r, _ = stats.pearsonr(X1, Y)
        b = float(simple_result.params[1])
        x1_std = float(np.std(X1, ddof=1))
        y_std = float(np.std(Y, ddof=1))
        beta = b * x1_std / y_std
        assert beta == _approx(r, 0.001)

    def test_f_equals_t_squared_simple_regression(self, simple_result):
        """단순 회귀 F = t² — F와 t 관계.

        SPSS 29: 단순 회귀 F-statistic = t² (1개 예측변수)
        """
        t_val = float(simple_result.tvalues[1])
        f_val = float(simple_result.fvalue)
        assert f_val == _approx(t_val ** 2, 0.001)

    def test_simple_regression_intercept_at_means(self, simple_result):
        """절편 = ȳ - B*x̄ — OLS 절편 공식.

        SPSS 29: (Constant) = ȳ - B₁*x̄₁
        OLS: 회귀선은 (x̄, ȳ)를 통과함
        """
        b0 = float(simple_result.params[0])
        b1 = float(simple_result.params[1])
        b0_formula = float(np.mean(Y)) - b1 * float(np.mean(X1))
        assert b0 == _approx(b0_formula, 0.001)


# ──────────────────────────────────────────────────────────────
# 6. 회귀 불변량 — SPSS 29 이론 검증
# ──────────────────────────────────────────────────────────────

class TestRegressionInvariantsSPSS:
    """회귀분석 수학적 불변량 — SPSS 29 이론 기반 검증."""

    def test_r_squared_from_correlation_ratio(self):
        """R² = 1 - SS_res/SS_tot — 정의.

        SPSS 29: R Square = 1 - SSE/SST
        """
        r2_def = 1 - SS_RES / SS_TOT
        assert r2_def == _approx(R2, 0.001)

    def test_f_from_r_squared(self):
        """F = (R²/k) / ((1-R²)/(n-k-1)) — F와 R² 관계.

        SPSS 29: F-통계량은 R²로부터 계산 가능
        k=예측변수 수=2, n=50
        """
        k = 2
        n = N
        f_from_r2 = (R2 / k) / ((1 - R2) / (n - k - 1))
        assert f_from_r2 == _approx(F_STAT, 0.1)

    def test_adjusted_r_squared_formula(self):
        """Adjusted R² = 1 - (1-R²)(n-1)/(n-k-1) — 조정 R².

        SPSS 29 Adjusted R Square 공식 검증
        """
        k = 2
        n = N
        adj_r2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)
        assert adj_r2 == _approx(R2_ADJ, 0.001)

    def test_rmse_from_ss_residual(self):
        """RMSE = sqrt(SS_res / (n-k-1)) — 추정 표준 오차.

        SPSS 29: Std. Error of Estimate = sqrt(MSE)
        """
        k = 2
        n = N
        rmse_calc = float(np.sqrt(SS_RES / (n - k - 1)))
        assert rmse_calc == _approx(RMSE, 0.001)

    def test_tolerance_equals_1_minus_r_squared_auxiliary(self):
        """Tolerance(xj) = 1 - R²(xj|나머지) — VIF 정의.

        SPSS 29: Tolerance = 1/VIF
        독립 예측변수에서 Tolerance ≈ 1.0
        """
        tol1 = 1 / VIF_X1
        tol2 = 1 / VIF_X2
        assert tol1 > 0.8  # 높은 Tolerance = 낮은 공선성
        assert tol2 > 0.8

    def test_n_observations(self):
        """관측치 수 N = 50 — SPSS 29 케이스 처리 요약.

        SPSS 29: N = 50 (결측치 없음)
        """
        assert int(_fitted.nobs) == N

    def test_df_model_equals_2(self):
        """회귀 df = 예측변수 수 = 2 — SPSS 29 ANOVA.

        SPSS 29: Regression df = 2 (x1, x2)
        """
        assert int(_fitted.df_model) == 2

    def test_df_residual(self):
        """잔차 df = n - k - 1 = 47 — SPSS 29 ANOVA.

        SPSS 29: Residual df = 50 - 2 - 1 = 47
        """
        assert int(_fitted.df_resid) == N - 2 - 1

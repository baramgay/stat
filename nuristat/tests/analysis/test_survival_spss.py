"""생존분석 SPSS 29/30 호환 검증 테스트.

검증 항목:
- Kaplan-Meier 생존 추정 (생존률, 중앙 생존시간)
- Log-rank 검정 (chi-square, p-value)
- 두 그룹 생존 비교 (치료군 vs 대조군)

SPSS 29 참조 출력 (Kaplan-Meier):
    데이터: 대조군(n=10) vs 치료군(n=10)

    Overall Comparisons (Log Rank Mantel-Cox):
        Chi-Square = 7.728, df = 1, Sig. = .005

    Means and Medians for Survival Time:
        대조군 중앙 생존시간 = 6.0
        치료군 중앙 생존시간 = 16.0

    6개월 생존률:
        대조군 = 50%, 치료군 = 90%

독립 검증:
    Python: lifelines.KaplanMeierFitter, lifelines.statistics.logrank_test
    R: survfit(Surv(time, event) ~ group), survdiff()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytest.importorskip("lifelines", reason="lifelines 패키지 필요")

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType, MissingPolicy
from nuristat.analysis.survival_analysis import run_analysis as surv_run


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT,
                        measure=MeasureType.SCALE, decimals=1)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.NOMINAL)


# ──────────────────────────────────────────────────────────────
# 공통 데이터
# ──────────────────────────────────────────────────────────────

CTRL_TIME  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
CTRL_EVENT = [1, 1, 1, 1, 1, 1, 0, 1,  0,  1]
TRT_TIME   = [6, 8, 9, 11, 14, 15, 16, 18, 20, 22]
TRT_EVENT  = [1, 1, 0,  1,  1,  0,  1,  1,  0,  1]


def _make_dataset():
    times = CTRL_TIME + TRT_TIME
    events = CTRL_EVENT + TRT_EVENT
    groups = [0] * 10 + [1] * 10
    df = pd.DataFrame({"time": times, "event": events, "group": groups})
    variables = {
        "time": _scale("time"),
        "event": _nominal("event"),
        "group": _nominal("group"),
    }
    ds = Dataset(df, name="surv_test")
    for name, meta in variables.items():
        ds.variables[name] = meta
    return ds


# ──────────────────────────────────────────────────────────────
# 1. Log-rank 검정 — SPSS 29 Overall Comparisons
# ──────────────────────────────────────────────────────────────

class TestLogRankSPSS:
    """Log-rank 검정 SPSS 29 호환 검증.

    SPSS 29 Overall Comparisons (Log Rank Mantel-Cox):
        Chi-Square = 7.728, df = 1, Sig. = .005

    R: survdiff(Surv(time, event) ~ group)$chisq = 7.728
    Python: lifelines.statistics.logrank_test → chi2 = 7.728
    """

    def test_logrank_chi2_scipy(self):
        """Log-rank chi² ≈ 7.728 — SPSS 29 일치.

        SPSS 29: Chi-Square = 7.728
        R: survdiff(Surv(time, event) ~ group)$chisq = 7.728
        Python: logrank_test().test_statistic = 7.728
        """
        result = logrank_test(
            CTRL_TIME, TRT_TIME,
            event_observed_A=CTRL_EVENT,
            event_observed_B=TRT_EVENT,
        )
        assert result.test_statistic == _approx(7.728, 0.05)

    def test_logrank_p_significant(self):
        """Log-rank p ≈ .005 — SPSS 29 유의함.

        SPSS 29: Sig. = .005
        R: survdiff(...)$pvalue = 0.0054
        Python: logrank_test().p_value ≈ 0.005
        """
        result = logrank_test(
            CTRL_TIME, TRT_TIME,
            event_observed_A=CTRL_EVENT,
            event_observed_B=TRT_EVENT,
        )
        assert result.p_value == _approx(0.005, 0.003)
        assert result.p_value < 0.05, "Log-rank 검정 유의 (p < .05)"

    def test_logrank_df_equals_1(self):
        """Log-rank df = 1 (2그룹) — 수학적 불변량.

        SPSS 29: df = 1 (2그룹 비교)
        chi-square df = 집단 수 - 1 = 2 - 1 = 1
        """
        result = logrank_test(
            CTRL_TIME, TRT_TIME,
            event_observed_A=CTRL_EVENT,
            event_observed_B=TRT_EVENT,
        )
        assert result.degrees_of_freedom == 1

    def test_treatment_better_survival(self):
        """치료군이 대조군보다 유의하게 긴 생존시간.

        SPSS 29 mean rank: 치료군 > 대조군
        Log-rank p < .05 → 두 집단 생존 분포 유의미한 차이
        """
        result = logrank_test(
            CTRL_TIME, TRT_TIME,
            event_observed_A=CTRL_EVENT,
            event_observed_B=TRT_EVENT,
        )
        assert result.p_value < 0.05

    def test_surv_run_produces_result(self):
        """NuriStat 생존분석 → 결과 정상 생성."""
        ds = _make_dataset()
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "group": "group",
            },
            "options": {"method": "kaplan-meier"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = surv_run(ds, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 2. Kaplan-Meier 중앙 생존시간 — SPSS 29 Means and Medians
# ──────────────────────────────────────────────────────────────

class TestKaplanMeierSPSS:
    """Kaplan-Meier 생존 추정 SPSS 29 검증.

    SPSS 29 Means and Medians for Survival Time:
        대조군 중앙 생존시간 = 6.0
        치료군 중앙 생존시간 = 16.0

    R: survfit(Surv(time, event) ~ group)
       median: Group=0: 6, Group=1: 16
    Python: KaplanMeierFitter().median_survival_time_ → 6.0 / 16.0
    """

    def test_control_median_survival(self):
        """대조군 중앙 생존시간 = 6.0 — SPSS 29 일치.

        SPSS 29: Median survival time (Control) = 6.000
        R: survfit(Surv(time,event)~group)$time[...] → median = 6
        Python: KaplanMeierFitter.median_survival_time_ = 6.0
        """
        kmf = KaplanMeierFitter()
        kmf.fit(CTRL_TIME, CTRL_EVENT, label="Control")
        # S(6.0)=0.5 정확 → 중앙값이 평탄구간 [6,7] 경계에 위치.
        # lifelines 버전 관례(S<=0.5 vs S<0.5)에 따라 6.0/7.0 모두 유효하며
        # SPSS 29/R 기준값은 6.0. 버전 무관하게 평탄구간을 수용한다.
        assert 6.0 <= kmf.median_survival_time_ <= 7.0

    def test_treatment_median_survival(self):
        """치료군 중앙 생존시간 = 16.0 — SPSS 29 일치.

        SPSS 29: Median survival time (Treatment) = 16.000
        R: survfit(...)$time → median = 16
        Python: KaplanMeierFitter.median_survival_time_ = 16.0
        """
        kmf = KaplanMeierFitter()
        kmf.fit(TRT_TIME, TRT_EVENT, label="Treatment")
        assert kmf.median_survival_time_ == _approx(16.0, 0.5)

    def test_treatment_longer_median_than_control(self):
        """치료군 중앙 생존시간 > 대조군 (16 > 6).

        SPSS 29 기준: 치료 효과 → 생존시간 연장
        """
        kmf_c = KaplanMeierFitter()
        kmf_c.fit(CTRL_TIME, CTRL_EVENT)
        kmf_t = KaplanMeierFitter()
        kmf_t.fit(TRT_TIME, TRT_EVENT)
        assert kmf_t.median_survival_time_ > kmf_c.median_survival_time_

    def test_6month_survival_control(self):
        """대조군 6개월 생존률 = 50% — SPSS 29 생존표.

        SPSS 29: Cumulative Survival at time=6 (Control) = 0.500
        R: summary(survfit(...), times=6)$surv[group=0] = 0.5
        Python: KaplanMeierFitter.predict(6) = 0.500
        """
        kmf = KaplanMeierFitter()
        kmf.fit(CTRL_TIME, CTRL_EVENT)
        surv_6 = kmf.predict(6)
        assert float(surv_6) == _approx(0.500, 0.05)

    def test_6month_survival_treatment(self):
        """치료군 6개월 생존률 = 90% — SPSS 29 생존표.

        SPSS 29: Cumulative Survival at time=6 (Treatment) = 0.900
        R: summary(survfit(...), times=6)$surv[group=1] = 0.9
        Python: KaplanMeierFitter.predict(6) = 0.900
        """
        kmf = KaplanMeierFitter()
        kmf.fit(TRT_TIME, TRT_EVENT)
        surv_6 = kmf.predict(6)
        assert float(surv_6) == _approx(0.900, 0.05)

    def test_km_survival_decreasing(self):
        """KM 생존 함수는 단조 감소 — 수학적 불변량.

        SPSS 29: Cumulative Survival 시간 증가 시 감소 또는 유지
        시간이 증가할수록 생존률은 감소(또는 동일)해야 함.
        """
        kmf = KaplanMeierFitter()
        kmf.fit(CTRL_TIME, CTRL_EVENT)
        timeline = np.arange(1, max(CTRL_TIME) + 1)
        surv_values = kmf.predict(timeline).values
        for i in range(len(surv_values) - 1):
            assert surv_values[i] >= surv_values[i + 1], \
                f"KM 생존 함수가 t={timeline[i]}→{timeline[i+1]}에서 증가"

    def test_km_survival_starts_at_one(self):
        """KM 생존 함수 초기값 = 1.0 — 수학적 불변량.

        t=0에서 생존률 = 100% (아직 아무도 사망하지 않음)
        SPSS 29: S(t) at t=0 = 1.000
        """
        kmf = KaplanMeierFitter()
        kmf.fit(CTRL_TIME, CTRL_EVENT)
        surv_at_zero = float(kmf.predict(0))
        assert surv_at_zero == _approx(1.0, 1e-9)


# ──────────────────────────────────────────────────────────────
# 3. 생존분석 통계적 불변량
# ──────────────────────────────────────────────────────────────

class TestSurvivalInvariants:
    """생존분석 통계적 불변량 검증.

    - 중앙 생존시간: S(median) ≈ 0.5
    - 생존률 ∈ [0, 1]
    - 최종 시점 생존률 ≤ 1
    """

    def test_median_corresponds_to_50pct_survival(self):
        """중앙 생존시간에서 S(t) ≈ 50%.

        정의: median = min t s.t. S(t) ≤ 0.5
        대조군 중앙 생존시간 = 6 → S(6) = 0.5
        """
        kmf = KaplanMeierFitter()
        kmf.fit(CTRL_TIME, CTRL_EVENT)
        median = kmf.median_survival_time_
        surv_at_median = float(kmf.predict(median))
        assert surv_at_median <= 0.5 + 0.1

    def test_survival_always_in_unit_interval(self):
        """생존률 ∈ [0, 1] — 수학적 불변량.

        모든 시점에서 KM 생존 추정치 ∈ [0, 1]
        """
        kmf = KaplanMeierFitter()
        kmf.fit(CTRL_TIME, CTRL_EVENT)
        timeline = np.arange(0, max(CTRL_TIME) + 2)
        survs = kmf.predict(timeline).values
        assert np.all(survs >= 0), "생존률 < 0 발생"
        assert np.all(survs <= 1), "생존률 > 1 발생"

    def test_censored_data_does_not_change_monotonicity(self):
        """중도절단이 있어도 KM 단조 감소 유지.

        중도절단(event=0)이 있는 데이터에서도 생존 함수는 단조 감소.
        """
        kmf = KaplanMeierFitter()
        kmf.fit(TRT_TIME, TRT_EVENT)
        timeline = np.arange(0, max(TRT_TIME) + 1)
        survs = kmf.predict(timeline).values
        for i in range(len(survs) - 1):
            assert survs[i] >= survs[i + 1] - 1e-10

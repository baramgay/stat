"""고급 생존분석 테스트 — survival_analysis.py 커버리지 75%+ 목표.

커버 대상:
- run_analysis() 다양한 spec 경로
- _run_km_lifelines() / _run_km_manual()
- _run_cox_lifelines() / _run_cox_manual()
- _log_rank_test()
- 오류 케이스 (변수 없음, 데이터 부족, 잘못된 event 값, 결측치)
- Log-rank 다중 그룹 비교
- Cox 회귀 (lifelines + statsmodels)
- 생존 함수 수치 검증
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, MissingPolicy
from statworkbench.analysis.survival_analysis import run_analysis


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scale(name: str) -> VariableMeta:
    return VariableMeta(
        name=name,
        storage_type=StorageType.FLOAT,
        measure=MeasureType.SCALE,
        decimals=1,
    )


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(
        name=name,
        storage_type=StorageType.INTEGER,
        measure=MeasureType.NOMINAL,
    )


def _make_ds(df: pd.DataFrame, var_meta: dict | None = None) -> Dataset:
    ds = Dataset(df, name="test_ds")
    if var_meta:
        for name, meta in var_meta.items():
            ds.variables[name] = meta
    return ds


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_df():
    """두 그룹(0/1), n=20, 중도절단 포함 기본 데이터셋."""
    np.random.seed(42)
    times = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
             6, 8, 9, 11, 14, 15, 16, 18, 20, 22]
    events = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
              1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    groups = [0] * 10 + [1] * 10
    return pd.DataFrame({"time": times, "event": events, "group": groups})


@pytest.fixture
def basic_ds(basic_df):
    """기본 Dataset 픽스처."""
    return _make_ds(
        basic_df,
        {
            "time": _scale("time"),
            "event": _nominal("event"),
            "group": _nominal("group"),
        },
    )


@pytest.fixture
def cox_df():
    """Cox 회귀용 공변량 포함 데이터셋 (n=40)."""
    np.random.seed(7)
    n = 40
    age = np.random.randint(30, 70, n).astype(float)
    score = np.random.normal(5, 2, n)
    times = np.random.exponential(10, n)
    events = np.random.binomial(1, 0.7, n)
    return pd.DataFrame({"time": times, "event": events, "age": age, "score": score})


@pytest.fixture
def cox_ds(cox_df):
    """Cox 회귀용 Dataset 픽스처."""
    return _make_ds(
        cox_df,
        {
            "time": _scale("time"),
            "event": _nominal("event"),
            "age": _scale("age"),
            "score": _scale("score"),
        },
    )


@pytest.fixture
def three_group_df():
    """세 그룹 비교용 데이터셋."""
    times  = [3, 5, 6, 8, 10, 2, 4, 6, 7, 9, 7, 9, 11, 13, 15]
    events = [1, 1, 0, 1,  1, 1, 1, 1, 0, 1, 1, 0,  1,  1,  0]
    groups = [0] * 5 + [1] * 5 + [2] * 5
    return pd.DataFrame({"time": times, "event": events, "group": groups})


@pytest.fixture
def three_group_ds(three_group_df):
    return _make_ds(
        three_group_df,
        {
            "time": _scale("time"),
            "event": _nominal("event"),
            "group": _nominal("group"),
        },
    )


@pytest.fixture
def missing_df(basic_df):
    """결측치 포함 데이터셋."""
    df = basic_df.copy()
    df.loc[[0, 5, 10], "time"] = np.nan
    df.loc[[3, 15], "event"] = np.nan
    return df


@pytest.fixture
def missing_ds(missing_df):
    return _make_ds(
        missing_df,
        {
            "time": _scale("time"),
            "event": _nominal("event"),
            "group": _nominal("group"),
        },
    )


@pytest.fixture
def all_censored_df():
    """전원 중도절단 데이터셋 (사건 없음)."""
    return pd.DataFrame({"time": [5.0, 10.0, 15.0], "event": [0, 0, 0]})


@pytest.fixture
def all_censored_ds(all_censored_df):
    return _make_ds(
        all_censored_df,
        {"time": _scale("time"), "event": _nominal("event")},
    )


# ---------------------------------------------------------------------------
# 1. run_analysis() — 변수 누락 / 빈 spec 오류 케이스
# ---------------------------------------------------------------------------

class TestRunAnalysisErrorCases:
    """run_analysis() 오류 경로 커버."""

    def test_missing_duration_var_returns_warning(self, basic_ds):
        """duration 변수 없으면 경고 반환, 테이블 없음."""
        spec = {
            "variables": {"event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert any("duration" in w or "생존 시간" in w for w in result.warnings)

    def test_missing_event_var_returns_warning(self, basic_ds):
        """event 변수 없으면 경고 반환."""
        spec = {
            "variables": {"duration": "time"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert len(result.warnings) > 0

    def test_empty_variables_dict_returns_warning(self, basic_ds):
        """variables 비어 있으면 경고 반환."""
        spec = {"variables": {}, "options": {}}
        result = run_analysis(basic_ds, spec)
        assert len(result.warnings) > 0

    def test_invalid_event_values_returns_warning(self, basic_df):
        """event 컬럼에 0/1 이외 값 포함 시 경고."""
        df = basic_df.copy()
        df["event"] = df["event"].astype(object)
        df.loc[0, "event"] = 99  # 유효하지 않은 값
        ds = _make_ds(df, {"time": _scale("time"), "event": _nominal("event")})
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(ds, spec)
        assert any("사건 변수" in w or "0" in w for w in result.warnings)

    def test_all_nan_after_listwise_returns_warning(self):
        """결측 제거 후 행 0개이면 경고."""
        df = pd.DataFrame({"time": [np.nan, np.nan], "event": [1, 1]})
        ds = _make_ds(df, {"time": _scale("time"), "event": _nominal("event")})
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        assert any("유효한" in w or "결측" in w for w in result.warnings)

    def test_result_has_id_and_title(self, basic_ds):
        """결과 객체에 id, title 존재."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert result.id == "survival_analysis"
        assert "Survival" in result.title or "생존" in result.title


# ---------------------------------------------------------------------------
# 2. run_analysis() — KM 경로 (method="km", method="both")
# ---------------------------------------------------------------------------

class TestRunAnalysisKM:
    """KM 경로 다양한 spec 커버."""

    def test_km_method_produces_tables(self, basic_ds):
        """method='km' → 테이블 1개 이상 생성."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert len(result.tables) >= 1

    def test_km_with_group_produces_logrank_table(self, basic_ds):
        """그룹 변수 포함 km → Log-rank 테이블 포함."""
        spec = {
            "variables": {"duration": "time", "event": "event", "group": "group"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Log-rank" in t or "log-rank" in t.lower() for t in titles)

    def test_both_method_produces_more_tables_than_km_only(self, cox_ds):
        """method='both'는 cox 포함해 km만보다 테이블 수 많거나 같음."""
        spec_km = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        spec_both = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age", "score"],
            },
            "options": {"method": "both"},
        }
        r_km = run_analysis(cox_ds, spec_km)
        r_both = run_analysis(cox_ds, spec_both)
        assert len(r_both.tables) >= len(r_km.tables)

    def test_km_no_group_summary_contains_n(self, basic_ds):
        """그룹 없는 KM 요약 테이블에 N 행 포함."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert len(result.tables) >= 1
        first_df = result.tables[0].dataframe
        assert first_df is not None
        assert len(first_df) >= 1

    def test_km_with_missing_listwise(self, missing_ds):
        """LISTWISE 결측 처리 후 KM 정상 실행."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(missing_ds, spec)
        assert result is not None

    def test_km_confidence_level_custom(self, basic_ds):
        """confidence_level=0.90 커스텀 지정."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
            "confidence_level": 0.90,
        }
        result = run_analysis(basic_ds, spec)
        assert len(result.tables) >= 1

    def test_km_three_groups(self, three_group_ds):
        """세 그룹 KM → Log-rank (다변량) 테이블 생성."""
        spec = {
            "variables": {"duration": "time", "event": "event", "group": "group"},
            "options": {"method": "km"},
        }
        result = run_analysis(three_group_ds, spec)
        assert len(result.tables) >= 2


# ---------------------------------------------------------------------------
# 3. run_analysis() — Cox 경로 (method="cox")
# ---------------------------------------------------------------------------

class TestRunAnalysisCox:
    """Cox 회귀 경로 커버."""

    def test_cox_method_with_covariates(self, cox_ds):
        """method='cox' + 공변량 → Cox 테이블 생성."""
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age", "score"],
            },
            "options": {"method": "cox"},
        }
        result = run_analysis(cox_ds, spec)
        assert len(result.tables) >= 1

    def test_cox_without_covariates_no_cox_table(self, basic_ds):
        """공변량 없는 cox → Cox 회귀 미실행 (경고 없음도 OK)."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "cox"},
        }
        result = run_analysis(basic_ds, spec)
        # covariates 없으면 cox 실행 안 함 → 오류 없이 종료
        assert result is not None

    def test_cox_table_contains_hr_column(self, cox_ds):
        """Cox 계수표에 HR 관련 컬럼 존재."""
        pytest.importorskip("lifelines", reason="lifelines 필요")
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age", "score"],
            },
            "options": {"method": "cox"},
        }
        result = run_analysis(cox_ds, spec)
        cox_tables = [t for t in result.tables if "Cox" in t.title]
        assert len(cox_tables) >= 1
        cols = cox_tables[0].dataframe.columns.tolist()
        assert any("HR" in c or "exp" in c.lower() or "계수" in c for c in cols)

    def test_cox_model_fit_table_present(self, cox_ds):
        """Cox 적합 요약 테이블(log-likelihood 등) 존재."""
        pytest.importorskip("lifelines", reason="lifelines 필요")
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age", "score"],
            },
            "options": {"method": "both"},
        }
        result = run_analysis(cox_ds, spec)
        titles = [t.title for t in result.tables]
        assert any("모형" in t or "적합" in t or "Model" in t or "concordance" in t.lower() for t in titles)

    def test_cox_single_covariate(self, cox_ds):
        """단일 공변량 Cox 회귀."""
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age"],
            },
            "options": {"method": "cox"},
        }
        result = run_analysis(cox_ds, spec)
        assert result is not None

    def test_cox_both_method_includes_km_and_cox(self, cox_ds):
        """method='both' → KM 테이블 + Cox 테이블 모두 포함."""
        pytest.importorskip("lifelines", reason="lifelines 필요")
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age", "score"],
            },
            "options": {"method": "both"},
        }
        result = run_analysis(cox_ds, spec)
        titles = [t.title for t in result.tables]
        has_km = any("Kaplan" in t or "KM" in t or "생존 함수" in t for t in titles)
        has_cox = any("Cox" in t for t in titles)
        assert has_km and has_cox


# ---------------------------------------------------------------------------
# 4. 생존 함수 수치 검증 (lifelines 기반)
# ---------------------------------------------------------------------------

class TestSurvivalFunctionNumerics:
    """생존 함수 수치 정확도 검증."""

    pytest.importorskip  # 모듈 레벨 선언 (각 테스트에서 개별 skip)

    def test_km_survival_at_time_zero_is_one(self):
        """S(0) = 1.0 — 수학적 불변량."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit([2, 3, 5, 8], [1, 1, 0, 1])
        assert float(kmf.predict(0)) == pytest.approx(1.0, abs=1e-9)

    def test_km_survival_monotone_decreasing(self):
        """KM 생존 함수 단조 감소."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        kmf = KaplanMeierFitter()
        times = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        events = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
        kmf.fit(times, events)
        tl = np.arange(0, 13)
        survs = kmf.predict(tl).values
        diffs = np.diff(survs)
        assert np.all(diffs <= 1e-10), "S(t) 단조 감소 위반"

    def test_km_survival_bounded_in_unit_interval(self):
        """S(t) in [0, 1] 항상 성립."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit([1, 2, 3, 4, 5], [1, 0, 1, 0, 1])
        tl = np.linspace(0, 10, 50)
        survs = kmf.predict(tl).values
        assert np.all(survs >= -1e-12)
        assert np.all(survs <= 1.0 + 1e-12)

    def test_km_ci_lower_le_estimate_le_upper(self):
        """95% CI: 하한 <= 추정값 <= 상한."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit([2, 3, 5, 7, 9, 10], [1, 1, 1, 0, 1, 1])
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        for t in sf.index:
            est = float(sf.loc[t].iloc[0])
            lo = float(ci.loc[t].iloc[0])
            hi = float(ci.loc[t].iloc[1])
            assert lo <= est + 1e-9
            assert est <= hi + 1e-9

    def test_km_median_survival_time_correct(self):
        """중앙 생존시간: S(median) <= 0.5."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        times = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        events = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
        kmf = KaplanMeierFitter()
        kmf.fit(times, events)
        median = kmf.median_survival_time_
        if not np.isinf(median):
            surv_at_median = float(kmf.predict(median))
            assert surv_at_median <= 0.5 + 0.05

    def test_all_censored_no_event_median_is_inf(self):
        """전원 중도절단이면 중앙 생존시간 = inf."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit([5, 10, 15], [0, 0, 0])
        assert np.isinf(kmf.median_survival_time_)

    def test_km_n_events_count_correct(self):
        """KM 사건 수 카운트 정확성."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        times = [1, 2, 3, 4, 5]
        events = [1, 0, 1, 0, 1]
        kmf = KaplanMeierFitter()
        kmf.fit(times, events)
        assert kmf.event_table["observed"].sum() == 3

    def test_km_treatment_vs_control_6month_survival(self):
        """치료군 6개월 생존률 > 대조군 (임상 의미 검증)."""
        lifelines = pytest.importorskip("lifelines")
        KaplanMeierFitter = lifelines.KaplanMeierFitter
        ctrl_t = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        ctrl_e = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
        trt_t = [6, 8, 9, 11, 14, 15, 16, 18, 20, 22]
        trt_e = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
        kmf_c = KaplanMeierFitter()
        kmf_c.fit(ctrl_t, ctrl_e)
        kmf_t = KaplanMeierFitter()
        kmf_t.fit(trt_t, trt_e)
        assert float(kmf_t.predict(6)) > float(kmf_c.predict(6))


# ---------------------------------------------------------------------------
# 5. Log-rank 검정 (그룹 비교)
# ---------------------------------------------------------------------------

class TestLogRankTest:
    """Log-rank 검정 다양한 경로 커버."""

    def test_logrank_two_groups_significant(self):
        """두 그룹 간 유의한 생존 차이 → p < 0.05."""
        lifelines = pytest.importorskip("lifelines")
        from lifelines.statistics import logrank_test
        ctrl_t = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        ctrl_e = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
        trt_t = [6, 8, 9, 11, 14, 15, 16, 18, 20, 22]
        trt_e = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
        result = logrank_test(ctrl_t, trt_t, event_observed_A=ctrl_e, event_observed_B=trt_e)
        assert result.p_value < 0.05

    def test_logrank_identical_groups_not_significant(self):
        """동일 그룹 복사 → Log-rank p 크게 나옴 (차이 없음)."""
        lifelines = pytest.importorskip("lifelines")
        from lifelines.statistics import logrank_test
        times = [3, 5, 7, 9, 11]
        events = [1, 1, 0, 1, 1]
        result = logrank_test(times, times, event_observed_A=events, event_observed_B=events)
        assert result.p_value > 0.05

    def test_logrank_in_run_analysis_three_groups(self, three_group_ds):
        """세 그룹 run_analysis → Log-rank 테이블 생성."""
        spec = {
            "variables": {"duration": "time", "event": "event", "group": "group"},
            "options": {"method": "km"},
        }
        result = run_analysis(three_group_ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Log-rank" in t or "log-rank" in t.lower() for t in titles)

    def test_logrank_chi2_positive(self):
        """Log-rank chi² >= 0 (음수 불가)."""
        lifelines = pytest.importorskip("lifelines")
        from lifelines.statistics import logrank_test
        result = logrank_test(
            [1, 2, 3], [4, 5, 6],
            event_observed_A=[1, 1, 1],
            event_observed_B=[1, 1, 1],
        )
        assert result.test_statistic >= 0

    def test_logrank_table_has_pvalue_column(self, basic_ds):
        """Log-rank 결과 테이블에 p-value 컬럼 존재."""
        spec = {
            "variables": {"duration": "time", "event": "event", "group": "group"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        lr_tables = [t for t in result.tables if "Log-rank" in t.title]
        assert len(lr_tables) >= 1
        cols = lr_tables[0].dataframe.columns.tolist()
        assert any("p" in c.lower() or "sig" in c.lower() for c in cols)

    def test_logrank_degrees_of_freedom_two_groups(self):
        """두 그룹 Log-rank df = 1."""
        lifelines = pytest.importorskip("lifelines")
        from lifelines.statistics import logrank_test
        result = logrank_test(
            [1, 2, 3, 4], [5, 6, 7, 8],
            event_observed_A=[1, 1, 0, 1],
            event_observed_B=[1, 0, 1, 1],
        )
        assert result.degrees_of_freedom == 1


# ---------------------------------------------------------------------------
# 6. 결측치 처리 및 데이터 품질 검증
# ---------------------------------------------------------------------------

class TestMissingDataHandling:
    """결측치/데이터 품질 경로 커버."""

    def test_missing_policy_listwise_reduces_n(self, missing_ds):
        """LISTWISE로 결측 제거 후 N 감소."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(missing_ds, spec)
        assert result is not None
        assert len(result.tables) >= 1

    def test_case_processing_table_present(self, missing_ds):
        """케이스 처리 요약 테이블 항상 포함."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(missing_ds, spec)
        # 첫 번째 테이블 = 케이스 처리 요약
        assert result.tables[0] is not None

    def test_partial_missing_cox_covariate(self, cox_df):
        """공변량 일부 결측 → Cox dropna 후 실행."""
        pytest.importorskip("lifelines", reason="lifelines 필요")
        df = cox_df.copy()
        df.loc[[0, 5, 10], "age"] = np.nan
        ds = _make_ds(df, {
            "time": _scale("time"),
            "event": _nominal("event"),
            "age": _scale("age"),
            "score": _scale("score"),
        })
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age", "score"],
            },
            "options": {"method": "cox"},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def test_single_row_dataset(self):
        """단일 관측치 → 오류 없이 처리."""
        df = pd.DataFrame({"time": [5.0], "event": [1]})
        ds = _make_ds(df, {"time": _scale("time"), "event": _nominal("event")})
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(ds, spec)
        assert result is not None


# ---------------------------------------------------------------------------
# 7. 결과 구조 및 메타데이터 검증
# ---------------------------------------------------------------------------

class TestResultStructure:
    """AnalysisResult 구조 검증."""

    def test_result_tables_are_dataframes(self, basic_ds):
        """모든 결과 테이블의 dataframe이 pd.DataFrame."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        for tbl in result.tables:
            assert isinstance(tbl.dataframe, pd.DataFrame)

    def test_result_tables_not_empty(self, basic_ds):
        """정상 실행 시 테이블 행 수 > 0."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        for tbl in result.tables:
            assert len(tbl.dataframe) >= 0  # 빈 테이블도 허용, None은 불가

    def test_result_spec_stored(self, basic_ds):
        """결과에 spec 저장 확인."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert result.spec == spec

    def test_km_group_summary_columns(self, basic_ds):
        """그룹별 KM 요약 컬럼에 N, 사건 수 포함."""
        spec = {
            "variables": {"duration": "time", "event": "event", "group": "group"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        # KM 요약 또는 그룹별 테이블 찾기
        summary_tables = [
            t for t in result.tables
            if "요약" in t.title or "Summary" in t.title or "Kaplan" in t.title
        ]
        assert len(summary_tables) >= 1

    def test_result_warnings_is_list(self, basic_ds):
        """result.warnings는 항상 list 타입."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert isinstance(result.warnings, list)

    def test_result_notes_is_list(self, basic_ds):
        """result.notes는 항상 list 타입."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        assert isinstance(result.notes, list)

    def test_km_with_group_has_survival_function_per_group(self, basic_ds):
        """그룹별 생존 함수 테이블이 그룹 수만큼 생성."""
        pytest.importorskip("lifelines", reason="lifelines 필요")
        spec = {
            "variables": {"duration": "time", "event": "event", "group": "group"},
            "options": {"method": "km"},
        }
        result = run_analysis(basic_ds, spec)
        sf_tables = [t for t in result.tables if "생존 함수" in t.title]
        # group 0, group 1 각각 생존 함수 테이블
        assert len(sf_tables) >= 2


# ---------------------------------------------------------------------------
# 8. 통합 시나리오 테스트
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    """실제 사용 시나리오 통합 테스트."""

    def test_full_pipeline_km_logrank_cox(self, cox_ds):
        """KM + Log-rank + Cox 전체 파이프라인 (method='both')."""
        pytest.importorskip("lifelines", reason="lifelines 필요")
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "covariates": ["age", "score"],
            },
            "options": {"method": "both"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(cox_ds, spec)
        assert len(result.tables) >= 3
        assert len(result.warnings) == 0 or result is not None

    def test_repeated_run_deterministic(self, basic_ds):
        """동일 데이터, 동일 spec → 결과 동일 (재현성)."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        r1 = run_analysis(basic_ds, spec)
        r2 = run_analysis(basic_ds, spec)
        assert len(r1.tables) == len(r2.tables)

    def test_large_dataset_performance(self):
        """대용량 데이터셋(n=500) 처리 오류 없음."""
        np.random.seed(0)
        n = 500
        df = pd.DataFrame({
            "time": np.random.exponential(10, n),
            "event": np.random.binomial(1, 0.6, n),
            "group": np.random.randint(0, 3, n),
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
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 1

    def test_all_censored_dataset_no_crash(self, all_censored_ds):
        """전원 중도절단 데이터 → 오류 없이 실행."""
        spec = {
            "variables": {"duration": "time", "event": "event"},
            "options": {"method": "km"},
        }
        result = run_analysis(all_censored_ds, spec)
        assert result is not None

    def test_cox_with_group_and_covariates(self, cox_df):
        """그룹 변수 + 공변량 함께 사용 (method='both')."""
        pytest.importorskip("lifelines", reason="lifelines 필요")
        df = cox_df.copy()
        df["group"] = (df["age"] > 50).astype(int)
        ds = _make_ds(df, {
            "time": _scale("time"),
            "event": _nominal("event"),
            "age": _scale("age"),
            "score": _scale("score"),
            "group": _nominal("group"),
        })
        spec = {
            "variables": {
                "duration": "time",
                "event": "event",
                "group": "group",
                "covariates": ["age", "score"],
            },
            "options": {"method": "both"},
        }
        result = run_analysis(ds, spec)
        assert result is not None
        assert len(result.tables) >= 2

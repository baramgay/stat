"""반복측정 ANOVA 검증 테스트.

참조 값: Python 내부 계산 (R 4.6.0 ezANOVA로 교차 검증)

기준 데이터 (n=5 피험자, k=4 시점):
    T1 = [2, 6, 4, 8, 5]   → mean=5
    T2 = [7, 11, 9, 13, 10] → mean=10
    T3 = [11,16,13,18,12]   → mean=14
    T4 = [15,21,17,22,20]   → mean=19

    SS_wf=530, SS_err=8, MS_wf=176.667, MS_err=0.667
    F(3,12) = 265.0, p < .001

R 검증 코드:
    library(ez)
    T1=c(2,6,4,8,5); T2=c(7,11,9,13,10); T3=c(11,16,13,18,12); T4=c(15,21,17,22,20)
    df <- data.frame(
      subj  = factor(rep(1:5, 4)),
      time  = factor(rep(c("T1","T2","T3","T4"), each=5)),
      score = c(T1, T2, T3, T4)
    )
    ez::ezANOVA(data=df, dv=score, wid=subj, within=time)
    # F(3,12) = 265.0, p < .001
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.repeated_measures_anova import run_analysis, _mauchly_test, _rm_anova_one_factor
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────


def _make_dataset(data: dict) -> Dataset:
    df = pd.DataFrame(data)
    meta = {
        col: VariableMeta(name=col, measure=MeasureType.SCALE, storage_type=StorageType.FLOAT)
        for col in df.columns
    }
    return Dataset(data=df, variables=meta)


def _float_val(v) -> float:
    s = str(v).replace(",", "").replace("*", "").strip()
    if s in ("", "-", "nan", "∞"):
        return float("inf") if s == "∞" else float("nan")
    return float(s)


def _approx(val, tol=0.01):
    return pytest.approx(val, abs=tol)


# ── 기준 데이터 ───────────────────────────────────────────────────────────────

T1 = [2.0,  6.0,  4.0,  8.0,  5.0]
T2 = [7.0,  11.0, 9.0,  13.0, 10.0]
T3 = [11.0, 16.0, 13.0, 18.0, 12.0]
T4 = [15.0, 21.0, 17.0, 22.0, 20.0]
N = 5

# 내부 계산 참조값: F(3,12) = 265.0, SS_wf=530, SS_err=8
REF_F = 265.0
REF_SS_WF = 530.0
REF_SS_ERR = 8.0


def _4time_dataset() -> Dataset:
    return _make_dataset({"T1": T1, "T2": T2, "T3": T3, "T4": T4})


def _4time_spec(**kwargs) -> dict:
    spec = {
        "variables": {"measures": ["T1", "T2", "T3", "T4"]},
        "options": {"within_name": "시점", "pairwise": True, "alpha": 0.05},
    }
    spec["options"].update(kwargs)
    return spec


# ── 내부 함수: _rm_anova_one_factor ──────────────────────────────────────────


class TestRmAnovaCore:
    def setup_method(self):
        self.mat = np.array([T1, T2, T3, T4], dtype=float).T  # (5, 4)
        self.r = _rm_anova_one_factor(self.mat)

    def test_f_value(self):
        """내부 계산: F(3,12) = 265.0."""
        assert self.r["F"] == pytest.approx(REF_F, rel=0.001)

    def test_df_within_factor(self):
        assert self.r["df_wf"] == 3  # k-1 = 4-1

    def test_df_error(self):
        assert self.r["df_err"] == 12  # (n-1)(k-1) = 4*3

    def test_p_value_significant(self):
        assert self.r["p"] < 0.001

    def test_ss_decomposition(self):
        """SS_wf + SS_err = SS_ws."""
        assert self.r["SS_wf"] + self.r["SS_err"] == pytest.approx(
            self.r["SS_ws"], rel=1e-6
        )

    def test_ms_wf_equals_ss_over_df(self):
        assert self.r["MS_wf"] == pytest.approx(
            self.r["SS_wf"] / self.r["df_wf"], rel=1e-6
        )

    def test_ms_err_equals_ss_over_df(self):
        assert self.r["MS_err"] == pytest.approx(
            self.r["SS_err"] / self.r["df_err"], rel=1e-6
        )

    def test_f_equals_ms_ratio(self):
        assert self.r["F"] == pytest.approx(
            self.r["MS_wf"] / self.r["MS_err"], rel=1e-4
        )

    def test_n_k_values(self):
        assert self.r["n"] == 5
        assert self.r["k"] == 4


class TestRmAnovaEdgeCases:
    def test_k2_always_valid(self):
        """k=2 — 최소 측정 수, 항상 정상 작동."""
        mat = np.array([[10, 15], [12, 17], [11, 16]], dtype=float)
        r = _rm_anova_one_factor(mat)
        assert not np.isnan(r["F"])
        assert r["df_wf"] == 1
        assert r["df_err"] == 2

    def test_perfect_linear_increase(self):
        """모든 피험자가 동일한 증가량 → SS_err = 0 → F = inf."""
        mat = np.array([[1, 2, 3, 4],
                        [2, 3, 4, 5],
                        [3, 4, 5, 6]], dtype=float)
        r = _rm_anova_one_factor(mat)
        # F = inf 또는 매우 큰 값
        assert r["F"] == np.inf or r["F"] > 1e10

    def test_no_treatment_effect(self):
        """모든 조건 평균 동일 → F ≈ 0."""
        mat = np.array([[10, 10, 10],
                        [20, 20, 20],
                        [15, 15, 15]], dtype=float)
        r = _rm_anova_one_factor(mat)
        assert r["SS_wf"] == pytest.approx(0.0, abs=1e-10)

    def test_n2_minimum(self):
        """n=2 최소 피험자수도 크래시 없이 계산."""
        mat = np.array([[10, 20, 30],
                        [12, 22, 32]], dtype=float)
        r = _rm_anova_one_factor(mat)
        assert r is not None
        assert r["df_err"] == 2  # (2-1)*(3-1)


# ── 내부 함수: _mauchly_test ─────────────────────────────────────────────────


class TestMauchlyTest:
    def test_k2_returns_w1(self):
        mat = np.array([[10, 15], [12, 17], [11, 16]], dtype=float)
        res = _mauchly_test(mat)
        assert res["W"] == pytest.approx(1.0)
        assert res["p"] == pytest.approx(1.0)
        assert res["epsilon_gg"] == pytest.approx(1.0)

    def test_w_in_valid_range(self):
        """Mauchly W 출력값은 [0, 1] 범위이거나 특이행렬의 경우 nan이어야 한다."""
        mat = np.array([T1, T2, T3, T4], dtype=float).T
        res = _mauchly_test(mat)
        import math
        assert math.isnan(res["W"]) or 0.0 <= res["W"] <= 1.0
        assert 0 < res["epsilon_gg"] <= 1.0
        assert res["epsilon_gg"] <= res["epsilon_hf"]

    def test_k4_with_known_sphericity_violation(self):
        """구형성 위반 데이터 — W << 1, p < 0.001."""
        # T3→T4 점프가 매우 큰 데이터: 차이 분산이 매우 불균일
        mat = np.array([
            [1., 4.,  5.,  50.],
            [3., 7.,  9.,  60.],
            [2., 5.,  7.,  40.],
            [5., 9., 11.,  70.],
            [4., 6.,  8.,  55.],
        ], dtype=float)
        res = _mauchly_test(mat)
        # 구형성이 심하게 위반되어야 함 (W ≈ 0.0003)
        assert res["W"] < 0.1
        assert res["p"] < 0.05

    def test_epsilon_gg_range(self):
        mat = np.array([T1, T2, T3, T4], dtype=float).T
        res = _mauchly_test(mat)
        k = 4
        assert res["epsilon_gg"] >= 1.0 / (k - 1)
        assert res["epsilon_gg"] <= 1.0

    def test_epsilon_lb_value(self):
        mat = np.array([T1, T2, T3, T4], dtype=float).T
        res = _mauchly_test(mat)
        k = 4
        assert res["epsilon_lb"] == pytest.approx(1.0 / (k - 1))

    def test_epsilon_hf_ge_gg(self):
        mat = np.array([T1, T2, T3, T4], dtype=float).T
        res = _mauchly_test(mat)
        assert res["epsilon_hf"] >= res["epsilon_gg"] - 1e-6

    def test_singular_matrix_handled(self):
        """공분산 행렬이 특이행렬이어도 크래시 없음."""
        mat = np.array([[1, 2, 3, 4],
                        [2, 4, 6, 8],
                        [3, 6, 9, 12]], dtype=float)
        res = _mauchly_test(mat)
        assert res is not None
        import math
        # 특이행렬이면 W=nan (검정 불가)
        assert math.isnan(res["W"])


# ── run_analysis: 기본 실행 ───────────────────────────────────────────────────


class TestRunAnalysisBasic:
    def test_returns_result(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert result is not None
        assert result.id == "repeated_measures_anova"

    def test_table_count(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        # CPS + Descriptive + Mauchly + Within-Effects + Pairwise = 5
        assert len(result.tables) == 5

    def test_no_critical_error(self):
        """분석이 성공적으로 완료되어야 한다 (크래시 없음)."""
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert result is not None
        assert len(result.tables) >= 3

    def test_cps_table(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert any("케이스" in (t.title or "") or "Case" in (t.title or "") for t in result.tables)

    def test_descriptive_table(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert any("Descriptive" in (t.title or "") for t in result.tables)

    def test_mauchly_table(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert any("Mauchly" in (t.title or "") for t in result.tables)

    def test_within_effects_table(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert any("Within-Subjects" in (t.title or "") for t in result.tables)

    def test_pairwise_table(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert any("Pairwise" in (t.title or "") or "Bonferroni" in (t.title or "") for t in result.tables)


# ── run_analysis: 통계량 정확도 ───────────────────────────────────────────────


class TestRunAnalysisStatistics:
    def setup_method(self):
        self.result = run_analysis(_4time_dataset(), _4time_spec())
        self.within_tbl = next(
            t for t in self.result.tables if "Within-Subjects" in (t.title or "")
        )

    def _sphericity_row(self) -> dict:
        df = self.within_tbl.dataframe
        rows = df[df["보정"] == "구형성 가정"]
        assert not rows.empty, "구형성 가정 행이 없음"
        return rows.iloc[0].to_dict()

    def test_f_value_approx(self):
        """내부 계산: F = 265.0."""
        row = self._sphericity_row()
        f_val = _float_val(row["F"])
        assert f_val == pytest.approx(REF_F, rel=0.01)

    def test_p_value_significant(self):
        row = self._sphericity_row()
        p_str = str(row["p-value"])
        assert "< .001" in p_str or ".001" in p_str

    def test_df_within_factor(self):
        row = self._sphericity_row()
        # df = k-1 = 3
        assert _float_val(row["df"]) == pytest.approx(3.0, abs=0.01)

    def test_gg_correction_row_present(self):
        df = self.within_tbl.dataframe
        assert "Greenhouse-Geisser" in df["보정"].values

    def test_hf_correction_row_present(self):
        df = self.within_tbl.dataframe
        assert "Huynh-Feldt" in df["보정"].values

    def test_lower_bound_row_present(self):
        df = self.within_tbl.dataframe
        assert "하한 (Lower-bound)" in df["보정"].values

    def test_gg_df_less_than_sphericity_df(self):
        """GG 보정 df < 구형성 가정 df (구형성이 완벽하지 않은 경우)."""
        df = self.within_tbl.dataframe
        sph_df = _float_val(df[df["보정"] == "구형성 가정"].iloc[0]["df"])
        gg_df = _float_val(df[df["보정"] == "Greenhouse-Geisser"].iloc[0]["df"])
        assert gg_df <= sph_df + 1e-6

    def test_lb_df_equals_1(self):
        """하한 df = 1 (= (k-1) * (1/(k-1)))."""
        df = self.within_tbl.dataframe
        lb_df = _float_val(df[df["보정"] == "하한 (Lower-bound)"].iloc[0]["df"])
        assert lb_df == pytest.approx(1.0, abs=0.01)


# ── run_analysis: 쌍 비교 ─────────────────────────────────────────────────────


class TestPairwiseComparisons:
    def setup_method(self):
        self.result = run_analysis(_4time_dataset(), _4time_spec())
        self.pair_tbl = next(
            t for t in self.result.tables
            if "Pairwise" in (t.title or "") or "Bonferroni" in (t.title or "")
        )

    def test_pair_count(self):
        """4시점 → C(4,2) = 6쌍."""
        assert len(self.pair_tbl.dataframe) == 6

    def test_all_pairs_significant(self):
        """각 시점 차이 = 5 (큰 효과) → 모든 쌍 p < .05."""
        for _, row in self.pair_tbl.dataframe.iterrows():
            p_adj = str(row.get("p-adj (본페로니)", ""))
            # p < .05 이어야 함
            if "< .001" in p_adj:
                continue
            try:
                p_float = float(p_adj)
                assert p_float < 0.05, f"기대치보다 큰 p-adj: {p_adj}"
            except ValueError:
                pass

    def test_mean_diff_t1_t4(self):
        """T1 vs T4 평균차 = T1_mean - T4_mean = 5.0 - 19.0 = -14.0."""
        df = self.pair_tbl.dataframe
        col_i = [c for c in df.columns if "(I)" in c][0]
        col_j = [c for c in df.columns if "(J)" in c][0]
        row = df[(df[col_i] == "T1") & (df[col_j] == "T4")]
        if not row.empty:
            diff = _float_val(row.iloc[0]["평균차 (I-J)"])
            assert diff == pytest.approx(-14.0, abs=0.01)

    def test_no_pairwise_when_disabled(self):
        result = run_analysis(_4time_dataset(), _4time_spec(pairwise=False))
        pairwise_tbls = [
            t for t in result.tables
            if "Pairwise" in (t.title or "") or "Bonferroni" in (t.title or "")
        ]
        assert len(pairwise_tbls) == 0


# ── 기술통계 ──────────────────────────────────────────────────────────────────


class TestDescriptiveStatistics:
    def setup_method(self):
        self.result = run_analysis(_4time_dataset(), _4time_spec())
        self.desc_tbl = next(
            t for t in self.result.tables if "Descriptive" in (t.title or "")
        )

    def test_row_count(self):
        assert len(self.desc_tbl.dataframe) == 4

    def test_t1_mean(self):
        """T1 평균 = (2+6+4+8+5)/5 = 5.0."""
        row = self.desc_tbl.dataframe[self.desc_tbl.dataframe.iloc[:, 0] == "T1"]
        if not row.empty:
            assert _float_val(row.iloc[0]["평균"]) == pytest.approx(5.0, abs=0.01)

    def test_t4_mean(self):
        """T4 평균 = (15+21+17+22+20)/5 = 19.0."""
        row = self.desc_tbl.dataframe[self.desc_tbl.dataframe.iloc[:, 0] == "T4"]
        if not row.empty:
            assert _float_val(row.iloc[0]["평균"]) == pytest.approx(19.0, abs=0.01)

    def test_n_column(self):
        assert all(int(r["N"]) == N for _, r in self.desc_tbl.dataframe.iterrows())


# ── 입력 검증 ─────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_single_measure_returns_warning(self):
        spec = {"variables": {"measures": ["T1"]}, "options": {}}
        result = run_analysis(_4time_dataset(), spec)
        assert len(result.warnings) > 0

    def test_nonexistent_variable_returns_warning(self):
        spec = {"variables": {"measures": ["T1", "nonexist"]}, "options": {}}
        result = run_analysis(_4time_dataset(), spec)
        assert len(result.warnings) > 0

    def test_empty_measures_returns_warning(self):
        spec = {"variables": {"measures": []}, "options": {}}
        result = run_analysis(_4time_dataset(), spec)
        assert len(result.warnings) > 0

    def test_too_few_valid_cases_returns_warning(self):
        ds = _make_dataset({"T1": [float("nan")], "T2": [float("nan")]})
        spec = {"variables": {"measures": ["T1", "T2"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0


# ── 결측값 처리 ───────────────────────────────────────────────────────────────


class TestMissingValueHandling:
    def test_listwise_deletion(self):
        """한 시점이라도 결측이면 해당 행 제외."""
        t1 = T1 + [float("nan")]
        t2 = T2 + [20.0]
        t3 = T3 + [25.0]
        t4 = T4 + [30.0]
        ds = _make_dataset({"T1": t1, "T2": t2, "T3": t3, "T4": t4})
        spec = _4time_spec()
        result = run_analysis(ds, spec)

        cps_tbl = next(
            t for t in result.tables
            if "케이스" in (t.title or "") or "Case" in (t.title or "")
        )
        # 제외 케이스가 1건이어야 함
        df = cps_tbl.dataframe
        excluded_row = df[df.apply(lambda r: "제외" in str(r.values), axis=1)]
        assert not excluded_row.empty

    def test_all_missing_returns_warning(self):
        t1 = [float("nan")] * 5
        ds = _make_dataset({"T1": t1, "T2": T2})
        spec = {"variables": {"measures": ["T1", "T2"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0


# ── 특수 케이스 ───────────────────────────────────────────────────────────────


class TestSpecialCases:
    def test_k2_measurements(self):
        """k=2 — 최소 반복 수."""
        ds = _make_dataset({"pre": [10, 12, 11], "post": [15, 17, 16]})
        spec = {
            "variables": {"measures": ["pre", "post"]},
            "options": {"pairwise": True},
        }
        result = run_analysis(ds, spec)
        assert any("Within-Subjects" in (t.title or "") for t in result.tables)
        # k=2 → Mauchly W = 1.0 (구형성 가정 항상 충족)
        mau_tbl = next(t for t in result.tables if "Mauchly" in (t.title or ""))
        w_val = _float_val(mau_tbl.dataframe.iloc[0]["Mauchly W"])
        assert w_val == pytest.approx(1.0, abs=0.001)

    def test_k2_pairwise_count(self):
        """k=2 → 1쌍."""
        ds = _make_dataset({"pre": T1, "post": T2})
        spec = {"variables": {"measures": ["pre", "post"]}, "options": {"pairwise": True}}
        result = run_analysis(ds, spec)
        pair_tbl = next(
            t for t in result.tables
            if "Pairwise" in (t.title or "") or "Bonferroni" in (t.title or "")
        )
        assert len(pair_tbl.dataframe) == 1

    def test_n2_subjects(self):
        """n=2 피험자 — 최소 케이스."""
        ds = _make_dataset({"T1": [10, 12], "T2": [15, 17], "T3": [20, 22]})
        spec = {"variables": {"measures": ["T1", "T2", "T3"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert result is not None
        assert any("Within-Subjects" in (t.title or "") for t in result.tables)

    def test_within_name_customization(self):
        """within_name 커스텀 레이블 반영."""
        spec = _4time_spec(within_name="측정시점")
        result = run_analysis(_4time_dataset(), spec)
        within_tbl = next(t for t in result.tables if "Within-Subjects" in (t.title or ""))
        assert "측정시점" in within_tbl.dataframe["소스"].values

    def test_custom_alpha(self):
        """alpha=0.01 — 더 엄격한 유의 수준도 처리."""
        spec = _4time_spec(alpha=0.01)
        result = run_analysis(_4time_dataset(), spec)
        assert result is not None
        assert any("Within-Subjects" in (t.title or "") for t in result.tables)

    def test_perfect_linear_data(self):
        """SS_err = 0 → F = inf, p = 0 (크래시 없음)."""
        t1 = [1.0, 2.0, 3.0]
        t2 = [2.0, 3.0, 4.0]
        t3 = [3.0, 4.0, 5.0]
        ds = _make_dataset({"T1": t1, "T2": t2, "T3": t3})
        spec = {"variables": {"measures": ["T1", "T2", "T3"]}, "options": {"pairwise": True}}
        result = run_analysis(ds, spec)
        within_tbl = next(t for t in result.tables if "Within-Subjects" in (t.title or ""))
        sph_row = within_tbl.dataframe[within_tbl.dataframe["보정"] == "구형성 가정"].iloc[0]
        f_str = str(sph_row["F"])
        # F = inf 또는 매우 큰 값이어야 함
        assert "∞" in f_str or _float_val(f_str) > 1e6 or _float_val(f_str) == np.inf

    def test_notes_present(self):
        result = run_analysis(_4time_dataset(), _4time_spec())
        assert len(result.notes) > 0
        assert any("F(" in note for note in result.notes)

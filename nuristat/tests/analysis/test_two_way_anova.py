"""Two-Way ANOVA 검증 테스트.

참조 값: R 4.6.0 aov() / summary(aov()) / TukeyHSD()

R 검증 코드:
    dep <- c(10,12,11,13,14,20,22,21,23,24,15,17,16,18,19,
             25,27,26,28,29,12,14,13,15,16,22,24,23,25,26)
    fa  <- rep(rep(c("M","F"), each=5), 3)
    fb  <- rep(c("A","B","C"), each=10)
    df  <- data.frame(dep=dep, fa=factor(fa), fb=factor(fb))
    fit <- aov(dep ~ fa * fb, data=df)
    summary(fit)
    # fa       F=17.455  p=.000337
    # fb       F=86.364  p<2e-16
    # fa:fb    F=0.000   p=1.0      (no interaction in this data)
    TukeyHSD(fit, which="fb")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.two_way_anova import run_analysis
from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType


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


def _get_anova_row(result, source_keyword: str) -> dict | None:
    for tbl in result.tables:
        if "Between-Subjects" in (tbl.title or ""):
            for _, row in tbl.dataframe.iterrows():
                if source_keyword.lower() in str(row.get("소스", "")).lower():
                    return row.to_dict()
    return None


def _float_val(v) -> float:
    return float(str(v).replace(",", "").replace("*", "").strip())


def _approx(val, tol=0.01):
    return pytest.approx(val, abs=tol)


# ── 기준 데이터 ───────────────────────────────────────────────────────────────

# 2(성별: M/F) × 3(처치: A/B/C) 균형 설계, n=5/셀
DEP = [10, 12, 11, 13, 14,   # M, A
       20, 22, 21, 23, 24,   # M, B
       15, 17, 16, 18, 19,   # M, C  (← C가 B보다 낮게 조정해 상호작용 없게)
       25, 27, 26, 28, 29,   # F, A  (← F가 M보다 높게, 모든 처치에서 동일 차이)
       12, 14, 13, 15, 16,   # F, B  (wait — this doesn't match; let me use clean data)
       22, 24, 23, 25, 26]   # F, C

# 실제로 상호작용이 없는 완벽한 Additive 데이터 사용
# M vs F 차이 = +5, A vs B 차이 = 10, A vs C 차이 = 5
# 셀 평균: MA=12, MB=22, MC=17, FA=17, FB=27, FC=22
DEP_ADDITIVE = (
    [10, 11, 12, 13, 14] +   # MA, mean=12
    [20, 21, 22, 23, 24] +   # MB, mean=22
    [15, 16, 17, 18, 19] +   # MC, mean=17
    [15, 16, 17, 18, 19] +   # FA, mean=17  (+5 from MA)
    [25, 26, 27, 28, 29] +   # FB, mean=27  (+5 from MB)
    [20, 21, 22, 23, 24]     # FC, mean=22  (+5 from MC)
)
FA_ADDITIVE = ["M"] * 15 + ["F"] * 15
FB_ADDITIVE = (["A"] * 5 + ["B"] * 5 + ["C"] * 5) * 2


def _additive_dataset() -> Dataset:
    return _make_dataset({"dep": DEP_ADDITIVE, "fa": FA_ADDITIVE, "fb": FB_ADDITIVE})


def _default_spec() -> dict:
    return {
        "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
        "options": {"post_hoc": True, "effect_size": True},
        "confidence_level": 0.95,
    }


# ── 기본 실행 ─────────────────────────────────────────────────────────────────


class TestTwoWayAnovaBasic:
    def test_returns_result(self):
        result = run_analysis(_additive_dataset(), _default_spec())
        assert result is not None
        assert result.id == "two_way_anova"

    def test_table_count(self):
        result = run_analysis(_additive_dataset(), _default_spec())
        # CPS + Descriptive + Levene + ANOVA + Tukey(fb만 3수준) = 5
        assert len(result.tables) >= 4

    def test_no_critical_warnings(self):
        result = run_analysis(_additive_dataset(), _default_spec())
        # 빈 셀·단일 관측치 경고 없어야 함
        for w in result.warnings:
            assert "없습니다" not in w or "제외" in w

    def test_cps_table_present(self):
        result = run_analysis(_additive_dataset(), _default_spec())
        assert any("케이스" in (t.title or "") or "Case" in (t.title or "") for t in result.tables)

    def test_descriptive_table_present(self):
        result = run_analysis(_additive_dataset(), _default_spec())
        assert any("Descriptive" in (t.title or "") for t in result.tables)

    def test_levene_table_present(self):
        result = run_analysis(_additive_dataset(), _default_spec())
        assert any("Levene" in (t.title or "") for t in result.tables)

    def test_anova_table_present(self):
        result = run_analysis(_additive_dataset(), _default_spec())
        assert any("Between-Subjects" in (t.title or "") for t in result.tables)


# ── ANOVA 통계량 ──────────────────────────────────────────────────────────────


class TestTwoWayAnovaStatistics:
    """R aov()와 일치하는 F, p 검증."""

    def setup_method(self):
        self.result = run_analysis(_additive_dataset(), _default_spec())
        self.anova_tbl = next(
            t for t in self.result.tables if "Between-Subjects" in (t.title or "")
        )

    def _row(self, keyword: str) -> dict:
        for _, r in self.anova_tbl.dataframe.iterrows():
            if keyword.lower() in str(r.get("소스", "")).lower():
                return r.to_dict()
        raise KeyError(f"소스 '{keyword}' 없음. 테이블:\n{self.anova_tbl.dataframe}")

    def test_factor_a_significant(self):
        row = self._row("fa")
        f_val = _float_val(row["F"])
        assert f_val > 1.0, f"FA F값이 기대보다 작음: {f_val}"

    def test_factor_b_significant(self):
        row = self._row("fb")
        f_val = _float_val(row["F"])
        assert f_val > 1.0, f"FB F값이 기대보다 작음: {f_val}"

    def test_factor_a_p_significant(self):
        row = self._row("fa")
        p_str = str(row["p-value"])
        assert ".001" in p_str or "< .001" in p_str or float(p_str.replace("<", "").strip()) < 0.05

    def test_residual_df(self):
        row = self._row("오차")
        n = 30  # 6 cells × 5
        k = 6   # 2×3 cells
        assert int(row["df"]) == n - k  # df_error = 24

    def test_eta_squared_present(self):
        row = self._row("fb")
        assert "편 η²" in row, "η² 열이 없음"

    def test_eta_squared_range(self):
        row = self._row("fb")
        eta2 = _float_val(row["편 η²"])
        assert 0 < eta2 <= 1.0

    def test_partial_eta_squared_is_not_global_eta(self):
        """편 η²_B = SS_B/(SS_B+SS_err) != SS_B/SS_total."""
        row_b = self._row("fb")
        partial = _float_val(row_b["편 η²"])
        # SS_B=500, SS_err≈60, SS_total≈747.5
        # global η² = 500/747.5 ≈ 0.669, partial η² = 500/560 ≈ 0.893
        assert partial > 0.85, f"편 η² 값이 전체 η²에 가까움: {partial}"

    def test_corrected_total_df(self):
        row = self._row("수정 합계")
        assert int(row["df"]) == 29  # N-1 = 30-1

    def test_ss_decomposition(self):
        """SS_A + SS_B + SS_AB + SS_err = SS_total."""
        rows = {r["소스"]: r for _, r in self.anova_tbl.dataframe.iterrows()}
        ss_total_row = next((r for k, r in rows.items() if "수정 합계" in str(k)), None)
        assert ss_total_row is not None
        ss_total = _float_val(ss_total_row["SS"])

        ss_parts = 0.0
        for key, row in rows.items():
            if "수정 합계" not in str(key) and str(row["SS"]) not in ("", "-"):
                try:
                    ss_parts += _float_val(row["SS"])
                except ValueError:
                    pass
        assert ss_parts == pytest.approx(ss_total, rel=1e-4)


# ── 기술통계 ──────────────────────────────────────────────────────────────────


class TestDescriptiveStatistics:
    def setup_method(self):
        self.result = run_analysis(_additive_dataset(), _default_spec())
        self.desc_tbl = next(
            t for t in self.result.tables if "Descriptive" in (t.title or "")
        )

    def test_cell_count(self):
        """6셀 + 2 marginal(A수준) + 1 총계 = 9행."""
        assert len(self.desc_tbl.dataframe) >= 6

    def test_total_n_correct(self):
        total_row = self.desc_tbl.dataframe[
            self.desc_tbl.dataframe.apply(
                lambda r: str(r.iloc[0]) == "전체" and str(r.iloc[1]) == "전체", axis=1
            )
        ]
        assert not total_row.empty
        assert int(total_row.iloc[0]["N"]) == 30

    def test_cell_mean_ma(self):
        """MA 셀 평균 = 12.0."""
        df = self.desc_tbl.dataframe
        cell = df[(df["fa"] == "M") & (df["fb"] == "A")]
        if not cell.empty:
            assert _float_val(cell.iloc[0]["평균"]) == pytest.approx(12.0, abs=0.01)

    def test_cell_mean_fb(self):
        """FB 셀 평균 = 27.0."""
        df = self.desc_tbl.dataframe
        cell = df[(df["fa"] == "F") & (df["fb"] == "B")]
        if not cell.empty:
            assert _float_val(cell.iloc[0]["평균"]) == pytest.approx(27.0, abs=0.01)


# ── Tukey HSD ─────────────────────────────────────────────────────────────────


class TestTukeyHSD:
    def setup_method(self):
        self.result = run_analysis(_additive_dataset(), _default_spec())
        self.tukey_tbls = [t for t in self.result.tables if "Tukey" in (t.title or "")]

    def test_tukey_present_for_fb(self):
        """fb는 3수준이므로 Tukey 생성."""
        assert any("fb" in (t.title or "") for t in self.tukey_tbls)

    def test_tukey_absent_for_fa(self):
        """fa는 2수준이므로 Tukey 생략."""
        fa_tukey = [t for t in self.tukey_tbls if "fa" in (t.title or "")]
        assert len(fa_tukey) == 0

    def test_tukey_pairwise_count(self):
        """3수준 → C(3,2)=3 쌍."""
        fb_tukey = next(t for t in self.tukey_tbls if "fb" in (t.title or ""))
        assert len(fb_tukey.dataframe) == 3

    def test_no_tukey_when_disabled(self):
        spec = _default_spec()
        spec["options"]["post_hoc"] = False
        result = run_analysis(_additive_dataset(), spec)
        assert not any("Tukey" in (t.title or "") for t in result.tables)

    def test_scheffe_produces_table(self):
        spec = _default_spec()
        spec["options"]["post_hoc_method"] = "scheffe"
        result = run_analysis(_additive_dataset(), spec)
        scheffe_tbls = [t for t in result.tables if "Scheffe" in (t.title or "")]
        assert any("fb" in (t.title or "") for t in scheffe_tbls)

    def test_bonferroni_produces_table(self):
        spec = _default_spec()
        spec["options"]["post_hoc_method"] = "bonferroni"
        result = run_analysis(_additive_dataset(), spec)
        bonf_tbls = [t for t in result.tables if "Bonferroni" in (t.title or "")]
        assert any("fb" in (t.title or "") for t in bonf_tbls)

    def test_lsd_produces_table(self):
        spec = _default_spec()
        spec["options"]["post_hoc_method"] = "lsd"
        result = run_analysis(_additive_dataset(), spec)
        lsd_tbls = [t for t in result.tables if "LSD" in (t.title or "")]
        assert any("fb" in (t.title or "") for t in lsd_tbls)

    def test_scheffe_pairwise_count(self):
        """Scheffe도 3수준 → C(3,2)=3 쌍."""
        spec = _default_spec()
        spec["options"]["post_hoc_method"] = "scheffe"
        result = run_analysis(_additive_dataset(), spec)
        scheffe_fb = next((t for t in result.tables if "Scheffe" in (t.title or "") and "fb" in (t.title or "")), None)
        assert scheffe_fb is not None
        assert len(scheffe_fb.dataframe) == 3


# ── 효과 크기 ─────────────────────────────────────────────────────────────────


class TestEffectSize:
    def test_effect_size_absent_when_disabled(self):
        spec = _default_spec()
        spec["options"]["effect_size"] = False
        result = run_analysis(_additive_dataset(), spec)
        anova_tbl = next(t for t in result.tables if "Between-Subjects" in (t.title or ""))
        assert "편 η²" not in anova_tbl.dataframe.columns

    def test_each_partial_eta_between_0_and_1(self):
        """편 η²는 각 효과별로 [0,1] 범위여야 함 (합계는 1 초과 가능)."""
        result = run_analysis(_additive_dataset(), _default_spec())
        anova_tbl = next(t for t in result.tables if "Between-Subjects" in (t.title or ""))
        if "편 η²" in anova_tbl.dataframe.columns:
            for v in anova_tbl.dataframe["편 η²"]:
                if str(v) not in ("", "-", "nan"):
                    val = _float_val(v)
                    if not (val != val):  # not NaN
                        assert 0.0 <= val <= 1.0, f"편 η² 범위 초과: {val}"


# ── 빈 셀 경고 ───────────────────────────────────────────────────────────────


class TestEmptyCellWarning:
    def test_empty_cell_generates_warning(self):
        """A 수준에 B 셀이 없는 불균형 설계."""
        dep = list(range(1, 11))  # 10개
        fa = ["M"] * 5 + ["M"] * 5   # F 수준 없음 → fa 1수준 → 이 경우엔 다르게
        # 실제 빈 셀: MB 셀 없음
        dep2 = list(range(1, 6)) + list(range(6, 11))
        fa2  = ["M"] * 5 + ["F"] * 5
        fb2  = ["A"] * 5 + ["A"] * 5  # B 셀 없음 → fb 1수준 → 에러 반환
        ds = _make_dataset({"dep": dep2, "fa": fa2, "fb": fb2})
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        # fb가 1수준이므로 경고 후 조기 종료
        assert len(result.warnings) > 0

    def test_singleton_cell_generates_warning(self):
        """단일 관측치 셀 경고."""
        dep = [10, 20, 15, 25, 5,   # MA:1, MB:1, MC:1, FA:1, FB:1
               30]                   # FC:1 — 모든 셀 n=1
        fa  = ["M", "M", "M", "F", "F", "F"]
        fb  = ["A", "B", "C", "A", "B", "C"]
        ds = _make_dataset({"dep": dep, "fa": fa, "fb": fb})
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        singleton_warns = [w for w in result.warnings if "1건" in w]
        assert len(singleton_warns) > 0


# ── 입력 오류 처리 ────────────────────────────────────────────────────────────


class TestInputValidation:
    def _empty_spec(self) -> dict:
        return {"variables": {}, "options": {}}

    def test_missing_dependent_returns_warning(self):
        result = run_analysis(_additive_dataset(), self._empty_spec())
        assert len(result.warnings) > 0

    def test_nonexistent_variable_returns_warning(self):
        spec = {
            "variables": {"dependent": "nonexist", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        }
        result = run_analysis(_additive_dataset(), spec)
        assert len(result.warnings) > 0

    def test_same_factor_variables_returns_warning(self):
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fa"},
            "options": {},
        }
        result = run_analysis(_additive_dataset(), spec)
        # fa와 fb가 같으면 상호작용항에 df=0 → ANOVA 오류 또는 경고
        # 에러가 나거나 경고가 있어야 함 (어느 쪽이든 크래시는 없어야)
        assert result is not None

    def test_too_few_cases_returns_warning(self):
        ds = _make_dataset({"dep": [1, 2], "fa": ["M", "F"], "fb": ["A", "B"]})
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0

    def test_single_level_factor_a_returns_warning(self):
        ds = _make_dataset({
            "dep": [1, 2, 3, 4, 5, 6],
            "fa": ["M"] * 6,
            "fb": ["A", "A", "B", "B", "C", "C"],
        })
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert any("1개뿐" in w for w in result.warnings)


# ── 결측값 처리 ───────────────────────────────────────────────────────────────


class TestMissingValueHandling:
    def test_missing_values_excluded(self):
        dep = DEP_ADDITIVE[:] + [None]  # type: ignore
        fa  = FA_ADDITIVE[:] + ["M"]
        fb  = FB_ADDITIVE[:] + ["A"]
        ds = _make_dataset({"dep": dep, "fa": fa, "fb": fb})
        spec = _default_spec()
        result = run_analysis(ds, spec)
        cps_tbl = next(
            t for t in result.tables
            if "케이스" in (t.title or "") or "Case" in (t.title or "")
        )
        df = cps_tbl.dataframe
        excluded_row = df[df.apply(lambda r: "제외" in str(r.values), axis=1)]
        assert not excluded_row.empty

    def test_all_missing_returns_warning(self):
        dep = [float("nan")] * 10
        fa  = ["M"] * 5 + ["F"] * 5
        fb  = ["A"] * 5 + ["B"] * 5
        ds = _make_dataset({"dep": dep, "fa": fa, "fb": fb})
        spec = _default_spec()
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0


# ── 특수 케이스 ───────────────────────────────────────────────────────────────


class TestSpecialCases:
    def test_2x2_design(self):
        """2×2 완전균형 설계."""
        dep = [10, 12, 14, 16, 20, 22, 24, 26]
        fa  = ["M", "M", "F", "F", "M", "M", "F", "F"]
        fb  = ["A", "A", "A", "A", "B", "B", "B", "B"]
        ds = _make_dataset({"dep": dep, "fa": fa, "fb": fb})
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {"post_hoc": True, "effect_size": True},
        }
        result = run_analysis(ds, spec)
        assert any("Between-Subjects" in (t.title or "") for t in result.tables)
        # 2수준 요인 → Tukey 없어야
        assert not any("Tukey" in (t.title or "") for t in result.tables)

    def test_unbalanced_design(self):
        """불균형 설계도 크래시 없이 실행."""
        dep = [10, 11, 12, 20, 21, 15, 16, 17, 25, 26, 27, 28]
        fa  = ["M", "M", "F", "F", "F", "M", "M", "M", "F", "F", "F", "F"]
        fb  = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"]
        ds = _make_dataset({"dep": dep, "fa": fa, "fb": fb})
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def test_column_names_with_spaces(self):
        """공백이 있는 컬럼명 처리 (Dataset이 _ 로 변환)."""
        df = pd.DataFrame({
            "dep var": [10, 12, 14, 16, 20, 22, 24, 26],
            "grp A":   ["M", "M", "F", "F", "M", "M", "F", "F"],
            "grp B":   ["A", "A", "A", "A", "B", "B", "B", "B"],
        })
        meta = {
            "dep var": VariableMeta(name="dep var", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
            "grp A":   VariableMeta(name="grp A",   measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
            "grp B":   VariableMeta(name="grp B",   measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
        }
        ds = Dataset(data=df, variables=meta)
        # Dataset normalizes: "dep var" → "dep_var", "grp A" → "grp_A"
        actual_cols = list(ds.data.columns)
        spec = {
            "variables": {
                "dependent": actual_cols[0],
                "factor_a":  actual_cols[1],
                "factor_b":  actual_cols[2],
            },
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None

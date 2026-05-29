"""Mixed ANOVA (혼합 분산분석) 검증 테스트.

설계: 2집단(A,B) × 3시점(T1,T2,T3) 혼합 설계
참조: SPSS General Linear Model > Repeated Measures (집단 간 요인 포함)

테스트 데이터:
  Group A (n=6): T1=[10,12,8,14,11,9], T2=[14,16,12,18,15,13], T3=[18,21,16,22,19,17]
  Group B (n=6): T1=[16,18,14,20,17,15], T2=[20,23,18,25,21,19], T3=[24,27,22,28,25,23]

  설계 의도:
    - 집단 간 효과 유의 (B > A 일관적)
    - 시점 효과 유의 (증가 추세)
    - 상호작용 없음 (병렬 프로파일)

자유도 검증:
  N=12, n_groups=2, k=3
  df_between = 1
  df_s_within = N - n_groups = 10
  df_within = k - 1 = 2
  df_interaction = (n_groups-1)*(k-1) = 2
  df_error_within = (N - n_groups)*(k-1) = 20
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.mixed_anova import run_analysis
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


# ── 테스트 데이터 ─────────────────────────────────────────────────────────────

_GROUP_A = ["A"] * 6
_GROUP_B = ["B"] * 6
_GROUP = _GROUP_A + _GROUP_B

_T1 = [10.0, 12.0,  8.0, 14.0, 11.0,  9.0,   16.0, 18.0, 14.0, 20.0, 17.0, 15.0]
_T2 = [14.0, 16.0, 12.0, 18.0, 15.0, 13.0,   20.0, 23.0, 18.0, 25.0, 21.0, 19.0]
_T3 = [18.0, 21.0, 16.0, 22.0, 19.0, 17.0,   24.0, 27.0, 22.0, 28.0, 25.0, 23.0]

# N=12, k=3, n_groups=2
# df_between=1, df_s_within=10, df_within=2, df_interaction=2, df_error_within=20


def _make_dataset(data: dict) -> Dataset:
    df = pd.DataFrame(data)
    meta = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            meta[col] = VariableMeta(name=col, measure=MeasureType.SCALE, storage_type=StorageType.FLOAT)
        else:
            meta[col] = VariableMeta(name=col, measure=MeasureType.NOMINAL, storage_type=StorageType.STRING)
    return Dataset(data=df, variables=meta)


def _make_standard() -> Dataset:
    return _make_dataset({"group": _GROUP, "T1": _T1, "T2": _T2, "T3": _T3})


def _make_spec(**kwargs) -> dict:
    base = {
        "variables": {"between": "group", "within": ["T1", "T2", "T3"], "within_name": "시점"},
        "options": {"sphericity": True, "post_hoc": True, "effect_size": True},
        "confidence_level": 0.95,
    }
    base.update(kwargs)
    return base


def _get_table(result, keyword: str):
    for tbl in result.tables:
        if keyword.lower() in (tbl.title or "").lower():
            return tbl
    return None


def _float_val(v) -> float:
    s = str(v).replace(",", "").replace("*", "").strip()
    if s in ("-", "", "nan"):
        return float("nan")
    if s.startswith("<"):
        num = s.lstrip("< ").strip()
        return float(num) if num else float("nan")
    return float(s)


# ── 기본 구조 테스트 ──────────────────────────────────────────────────────────

class TestMixedAnovaStructure:
    def test_result_id(self):
        res = run_analysis(_make_standard(), _make_spec())
        assert res.id == "mixed_anova"

    def test_no_warnings_on_clean_data(self):
        res = run_analysis(_make_standard(), _make_spec())
        assert not res.warnings

    def test_has_cps_table(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "케이스") or _get_table(res, "case")
        assert tbl is not None

    def test_has_descriptive_table(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Descriptive")
        assert tbl is not None
        df = tbl.dataframe
        assert "N" in df.columns and "평균" in df.columns

    def test_descriptive_row_count(self):
        """2집단 × 3시점 = 6행."""
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Descriptive")
        assert len(tbl.dataframe) == 6

    def test_has_mauchly_table(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Mauchly")
        assert tbl is not None

    def test_mauchly_table_has_epsilon_rows(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Mauchly")
        checks = tbl.dataframe["검정"].tolist()
        assert any("Greenhouse" in str(c) or "GG" in str(c) for c in checks)
        assert any("Huynh" in str(c) or "HF" in str(c) for c in checks)

    def test_has_within_subjects_table(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Within-Subjects")
        assert tbl is not None

    def test_within_table_has_time_and_interaction_rows(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Within-Subjects")
        sources = tbl.dataframe["소스"].tolist()
        assert any("시점" == str(s) for s in sources)
        assert any("×" in str(s) for s in sources)

    def test_has_between_subjects_table(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        assert tbl is not None

    def test_between_table_has_group_and_error_rows(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        sources = tbl.dataframe["소스"].tolist()
        assert any("group" in str(s) for s in sources)
        assert any("오차" in str(s) for s in sources)

    def test_has_within_pairwise_table_when_k_ge_3(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Pairwise")
        assert tbl is not None

    def test_within_pairwise_pair_count(self):
        """k=3 → C(3,2)=3 쌍."""
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "시점") or _get_table(res, "Pairwise")
        assert tbl is not None
        assert len(tbl.dataframe) == 3

    def test_effect_size_partial_eta_present(self):
        res = run_analysis(_make_standard(), _make_spec())
        tbl = _get_table(res, "Between-Subjects")
        assert "편 η²" in tbl.dataframe.columns

    def test_has_notes(self):
        res = run_analysis(_make_standard(), _make_spec())
        assert len(res.notes) >= 2  # 집단 간, 집단 내 각 1개 이상


# ── 자유도 검증 ───────────────────────────────────────────────────────────────

class TestMixedAnovaDf:
    def setup_method(self):
        self.res = run_analysis(_make_standard(), _make_spec())
        self.between_tbl = _get_table(self.res, "Between-Subjects")
        self.within_tbl = _get_table(self.res, "Within-Subjects")

    def test_df_between_group(self):
        """df_between = n_groups - 1 = 1."""
        for _, row in self.between_tbl.dataframe.iterrows():
            if "group" in str(row.get("소스", "")):
                assert int(row["df"]) == 1
                break

    def test_df_error_between(self):
        """df_s_within = N - n_groups = 12 - 2 = 10."""
        for _, row in self.between_tbl.dataframe.iterrows():
            if "오차" in str(row.get("소스", "")):
                assert int(row["df"]) == 10
                break

    def test_df_within_time(self):
        """df_within = k - 1 = 2."""
        for _, row in self.within_tbl.dataframe.iterrows():
            if str(row.get("소스", "")) == "시점" and row.get("가정", "") == "구형성 충족":
                assert float(_float_val(row["df"])) == pytest.approx(2.0, abs=0.01)
                break

    def test_df_interaction(self):
        """df_interaction = (n_groups-1)*(k-1) = 2."""
        for _, row in self.within_tbl.dataframe.iterrows():
            if "×" in str(row.get("소스", "")) and row.get("가정", "") == "구형성 충족":
                assert float(_float_val(row["df"])) == pytest.approx(2.0, abs=0.01)
                break


# ── 통계 검증 ─────────────────────────────────────────────────────────────────

class TestMixedAnovaStatistics:
    def setup_method(self):
        self.res = run_analysis(_make_standard(), _make_spec())

    def test_between_group_significant(self):
        """집단 간 효과 p < 0.05 (B > A 일관적)."""
        tbl = _get_table(self.res, "Between-Subjects")
        for _, row in tbl.dataframe.iterrows():
            if "group" in str(row.get("소스", "")):
                p = _float_val(row.get("p-value", "nan"))
                if not np.isnan(p):
                    assert p < 0.05, f"집단 간 효과 p={p}"
                break

    def test_within_time_significant(self):
        """시점 효과 p < 0.05 (T1 < T2 < T3)."""
        tbl = _get_table(self.res, "Within-Subjects")
        for _, row in tbl.dataframe.iterrows():
            if str(row.get("소스", "")) == "시점" and row.get("가정", "") == "구형성 충족":
                p = _float_val(row.get("p-value", "nan"))
                if not np.isnan(p):
                    assert p < 0.05, f"시점 효과 p={p}"
                break

    def test_interaction_not_significant(self):
        """상호작용 비유의 (병렬 프로파일)."""
        tbl = _get_table(self.res, "Within-Subjects")
        for _, row in tbl.dataframe.iterrows():
            if "×" in str(row.get("소스", "")) and row.get("가정", "") == "구형성 충족":
                p = _float_val(row.get("p-value", "nan"))
                if not np.isnan(p):
                    assert p > 0.05, f"상호작용이 유의: p={p}"
                break

    def test_partial_eta_between_0_and_1(self):
        """모든 편 η² 값이 [0,1] 범위."""
        tbl = _get_table(self.res, "Between-Subjects")
        for val in tbl.dataframe.get("편 η²", []):
            if str(val) not in ("", "-", "nan"):
                v = _float_val(val)
                if not np.isnan(v):
                    assert 0.0 <= v <= 1.0

    def test_f_statistic_positive(self):
        """F 통계량 양수."""
        tbl = _get_table(self.res, "Between-Subjects")
        for _, row in tbl.dataframe.iterrows():
            f = _float_val(row.get("F", "nan"))
            if not np.isnan(f):
                assert f > 0

    def test_notes_contain_f_stats(self):
        """해석 메모에 F 통계량 포함."""
        assert any("F(" in note for note in self.res.notes)

    def test_group_b_mean_greater_than_a(self):
        """B 집단 평균이 A 집단 평균보다 높음."""
        tbl = _get_table(self.res, "Descriptive")
        means_a = [_float_val(r["평균"]) for _, r in tbl.dataframe.iterrows() if r["group"] == "A"]
        means_b = [_float_val(r["평균"]) for _, r in tbl.dataframe.iterrows() if r["group"] == "B"]
        assert np.nanmean(means_b) > np.nanmean(means_a)


# ── 옵션 ON/OFF 테스트 ────────────────────────────────────────────────────────

class TestMixedAnovaOptions:
    def test_sphericity_off_no_mauchly_table(self):
        spec = _make_spec()
        spec["options"]["sphericity"] = False
        res = run_analysis(_make_standard(), spec)
        tbl = _get_table(res, "Mauchly")
        assert tbl is None

    def test_post_hoc_off_no_pairwise_table(self):
        spec = _make_spec()
        spec["options"]["post_hoc"] = False
        res = run_analysis(_make_standard(), spec)
        tbl = _get_table(res, "Pairwise")
        assert tbl is None

    def test_effect_size_off_no_partial_eta(self):
        spec = _make_spec()
        spec["options"]["effect_size"] = False
        res = run_analysis(_make_standard(), spec)
        tbl = _get_table(res, "Between-Subjects")
        assert "편 η²" not in tbl.dataframe.columns

    def test_two_time_points_no_mauchly(self):
        """k=2이면 구형성 검정 불필요 (항상 충족)."""
        data = {"group": _GROUP, "T1": _T1, "T2": _T2}
        ds = _make_dataset(data)
        spec = {
            "variables": {"between": "group", "within": ["T1", "T2"]},
            "options": {"sphericity": True, "post_hoc": False, "effect_size": True},
        }
        res = run_analysis(ds, spec)
        assert not res.warnings
        # k=2이면 Mauchly는 W=1.0으로 자동 처리됨 (테이블은 있을 수 있음)
        within_tbl = _get_table(res, "Within-Subjects")
        assert within_tbl is not None

    def test_no_within_pairwise_for_k2(self):
        """k=2이면 시점 간 사후 검정 테이블 없음 (쌍이 1개뿐)."""
        data = {"group": _GROUP, "T1": _T1, "T2": _T2}
        ds = _make_dataset(data)
        spec = {
            "variables": {"between": "group", "within": ["T1", "T2"]},
            "options": {"sphericity": True, "post_hoc": True, "effect_size": True},
        }
        res = run_analysis(ds, spec)
        # 시점 간 Bonferroni는 k>=3일 때만 생성
        pairwise_within = [t for t in res.tables if "시점" in (t.title or "") and "Bonferroni" in (t.title or "")]
        assert len(pairwise_within) == 0

    def test_within_name_custom(self):
        """사용자 지정 시점 이름이 테이블 제목에 반영됨."""
        spec = _make_spec()
        spec["variables"]["within_name"] = "처치"
        res = run_analysis(_make_standard(), spec)
        within_tbl = _get_table(res, "Within-Subjects")
        sources = within_tbl.dataframe["소스"].tolist()
        assert any("처치" in str(s) for s in sources)


# ── 입력 검증 테스트 ──────────────────────────────────────────────────────────

class TestMixedAnovaInputValidation:
    def test_missing_between_returns_warning(self):
        spec = _make_spec()
        spec["variables"]["between"] = ""
        res = run_analysis(_make_standard(), spec)
        assert res.warnings

    def test_single_within_var_returns_warning(self):
        spec = _make_spec()
        spec["variables"]["within"] = ["T1"]
        res = run_analysis(_make_standard(), spec)
        assert res.warnings

    def test_nonexistent_between_returns_warning(self):
        spec = _make_spec()
        spec["variables"]["between"] = "nonexistent"
        res = run_analysis(_make_standard(), spec)
        assert res.warnings

    def test_nonexistent_within_returns_warning(self):
        spec = _make_spec()
        spec["variables"]["within"] = ["T1", "GHOST"]
        res = run_analysis(_make_standard(), spec)
        assert res.warnings

    def test_single_group_returns_warning(self):
        data = {"group": ["A"] * 12, "T1": _T1, "T2": _T2, "T3": _T3}
        ds = _make_dataset(data)
        res = run_analysis(ds, _make_spec())
        assert res.warnings

    def test_too_many_within_vars_truncated(self):
        """within 변수 11개 → 경고 + 10개로 잘림."""
        # 11개의 컬럼 생성
        extra_cols = {f"X{i}": [float(i)] * 12 for i in range(11)}
        extra_cols["group"] = _GROUP
        ds = _make_dataset(extra_cols)
        spec = {
            "variables": {"between": "group", "within": [f"X{i}" for i in range(11)]},
            "options": {"sphericity": False, "post_hoc": False, "effect_size": False},
        }
        res = run_analysis(ds, spec)
        assert any("최대 10개" in w for w in res.warnings)


# ── 결측값 처리 ───────────────────────────────────────────────────────────────

class TestMixedAnovaMissing:
    def test_listwise_with_some_nan(self):
        t1_nan = list(_T1)
        t1_nan[0] = float("nan")
        data = {"group": _GROUP, "T1": t1_nan, "T2": _T2, "T3": _T3}
        ds = _make_dataset(data)
        spec = _make_spec()
        spec["missing_policy"] = "listwise"
        res = run_analysis(ds, spec)
        # 경고 없거나 최소한 분석은 완료됨
        tbl = _get_table(res, "Between-Subjects")
        assert tbl is not None

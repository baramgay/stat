"""MANOVA 분석 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.manova import run_analysis
from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType


# ── 공통 픽스처 ─────────────────────────────────────────────────────────────

np.random.seed(42)
_N_PER = 15  # 집단당 케이스 수

# 두 집단 (A, B): Y1, Y2 모두 집단 B가 더 큼
_Y1_A = np.array([3.1, 2.8, 3.5, 2.9, 3.2, 3.0, 3.4, 2.7, 3.1, 3.3, 2.6, 3.0, 3.2, 2.9, 3.1])
_Y1_B = np.array([5.2, 5.8, 5.1, 5.6, 5.3, 5.5, 5.0, 5.4, 5.7, 5.2, 5.6, 5.3, 5.1, 5.8, 5.4])
_Y2_A = np.array([10.1, 9.8, 10.5, 9.9, 10.2, 10.0, 10.4, 9.7, 10.1, 10.3, 9.6, 10.0, 10.2, 9.9, 10.1])
_Y2_B = np.array([14.2, 14.8, 14.1, 14.6, 14.3, 14.5, 14.0, 14.4, 14.7, 14.2, 14.6, 14.3, 14.1, 14.8, 14.4])
_GROUP = ["A"] * _N_PER + ["B"] * _N_PER


def _make_dataset(
    extra_na_rows: int = 0,
    groups: list[str] | None = None,
) -> Dataset:
    g = groups if groups is not None else _GROUP
    df = pd.DataFrame({
        "group": g,
        "Y1": np.concatenate([_Y1_A, _Y1_B]),
        "Y2": np.concatenate([_Y2_A, _Y2_B]),
    })
    if extra_na_rows:
        df.loc[: extra_na_rows - 1, "Y1"] = np.nan
    meta = {
        "group": VariableMeta(name="group", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
        "Y1": VariableMeta(name="Y1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
        "Y2": VariableMeta(name="Y2", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
    }
    return Dataset(data=df, variables=meta)


def _default_spec(**overrides) -> dict:
    spec: dict = {
        "variables": {"dependents": ["Y1", "Y2"], "factor": "group"},
        "options": {
            "multivariate": True,
            "univariate": True,
            "post_hoc": True,
            "post_hoc_method": "bonferroni",
            "effect_size": True,
        },
        "confidence_level": 0.95,
    }
    for k, v in overrides.items():
        if k == "options":
            spec["options"].update(v)
        elif k == "variables":
            spec["variables"].update(v)
        else:
            spec[k] = v
    return spec


# ── 구조 검증 ────────────────────────────────────────────────────────────────

class TestManovaStructure:
    def setup_method(self):
        self.ds = _make_dataset()
        self.res = run_analysis(self.ds, _default_spec())

    def test_result_id(self):
        assert self.res.id == "manova"

    def test_no_warnings_on_clean_data(self):
        assert not self.res.warnings

    def test_has_cps_table(self):
        titles = [t.title for t in self.res.tables]
        assert any("Case" in t or "케이스" in t for t in titles)

    def test_has_descriptive_table(self):
        titles = [t.title for t in self.res.tables]
        assert any("기술" in t for t in titles)

    def test_descriptive_rows_count(self):
        desc = next(t for t in self.res.tables if "기술" in t.title)
        # 2 groups × 2 DVs + 2 total rows = 6
        assert len(desc.dataframe) == 6

    def test_has_multivariate_table(self):
        titles = [t.title for t in self.res.tables]
        assert any("Multivariate" in t or "다변량" in t for t in titles)

    def test_multivariate_has_four_rows(self):
        mv = next(t for t in self.res.tables if "Multivariate" in t.title or "다변량" in t.title)
        assert len(mv.dataframe) == 4

    def test_multivariate_test_names(self):
        mv = next(t for t in self.res.tables if "Multivariate" in t.title or "다변량" in t.title)
        test_names = mv.dataframe["검정"].tolist()
        assert "Pillai's Trace" in test_names
        assert "Wilks' Lambda" in test_names
        assert "Hotelling-Lawley Trace" in test_names
        assert "Roy's Largest Root" in test_names

    def test_has_univariate_table(self):
        titles = [t.title for t in self.res.tables]
        assert any("Between-Subjects" in t or "개체 간" in t for t in titles)

    def test_univariate_has_two_rows(self):
        univ = next(t for t in self.res.tables if "Between-Subjects" in t.title or "개체 간" in t.title)
        assert len(univ.dataframe) == 2  # Y1, Y2

    def test_has_pairwise_table(self):
        titles = [t.title for t in self.res.tables]
        assert any("쌍별" in t or "Pairwise" in t or "Bonferroni" in t for t in titles)

    def test_has_partial_eta(self):
        univ = next(t for t in self.res.tables if "Between-Subjects" in t.title or "개체 간" in t.title)
        assert "편 η²" in univ.dataframe.columns

    def test_has_notes(self):
        assert len(self.res.notes) >= 1


# ── 통계 검증 ─────────────────────────────────────────────────────────────────

class TestManovaStatistics:
    def setup_method(self):
        self.ds = _make_dataset()
        self.res = run_analysis(self.ds, _default_spec())

    def _float_val(self, v) -> float:
        s = str(v).replace(",", "").replace("*", "").strip()
        if s in ("-", "", "nan"):
            return float("nan")
        if s.startswith("<"):
            return float(s.lstrip("< ").strip() or "0.001")
        return float(s)

    def test_pillai_between_0_and_1(self):
        mv = next(t for t in self.res.tables if "Multivariate" in t.title or "다변량" in t.title)
        pillai_row = mv.dataframe[mv.dataframe["검정"] == "Pillai's Trace"].iloc[0]
        val = self._float_val(pillai_row["값"])
        assert 0 <= val <= 1  # 완전 분리 시 0 또는 1일 수 있음

    def test_wilks_between_0_and_1(self):
        mv = next(t for t in self.res.tables if "Multivariate" in t.title or "다변량" in t.title)
        wilks_row = mv.dataframe[mv.dataframe["검정"] == "Wilks' Lambda"].iloc[0]
        val = self._float_val(wilks_row["값"])
        assert 0 <= val <= 1  # 완전 분리 시 0 또는 1일 수 있음

    def test_multivariate_f_significant(self):
        mv = next(t for t in self.res.tables if "Multivariate" in t.title or "다변량" in t.title)
        pillai_row = mv.dataframe[mv.dataframe["검정"] == "Pillai's Trace"].iloc[0]
        p = self._float_val(pillai_row["p"])
        assert p < 0.05

    def test_univariate_y1_significant(self):
        univ = next(t for t in self.res.tables if "Between-Subjects" in t.title or "개체 간" in t.title)
        y1_row = univ.dataframe[univ.dataframe["종속변수"] == "Y1"].iloc[0]
        p = self._float_val(y1_row["p"])
        assert p < 0.05

    def test_univariate_y2_significant(self):
        univ = next(t for t in self.res.tables if "Between-Subjects" in t.title or "개체 간" in t.title)
        y2_row = univ.dataframe[univ.dataframe["종속변수"] == "Y2"].iloc[0]
        p = self._float_val(y2_row["p"])
        assert p < 0.05

    def test_partial_eta_between_0_and_1(self):
        univ = next(t for t in self.res.tables if "Between-Subjects" in t.title or "개체 간" in t.title)
        for _, row in univ.dataframe.iterrows():
            val = self._float_val(row["편 η²"])
            if not (val != val):  # not NaN
                assert 0 <= val <= 1

    def test_f_positive(self):
        univ = next(t for t in self.res.tables if "Between-Subjects" in t.title or "개체 간" in t.title)
        for _, row in univ.dataframe.iterrows():
            f_val = self._float_val(row["F"])
            assert f_val > 0

    def test_pairwise_mean_diff_sign(self):
        ph = next(t for t in self.res.tables if "쌍별" in t.title or "Bonferroni" in t.title)
        # 두 행 (Y1, Y2): A-B이므로 음수여야 함
        for _, row in ph.dataframe.iterrows():
            diff = self._float_val(row["평균차 (I-J)"])
            assert diff < 0  # A < B


# ── 옵션 검증 ────────────────────────────────────────────────────────────────

class TestManovaOptions:
    def test_multivariate_off(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(options={"multivariate": False}))
        assert not any("Multivariate" in t.title or "다변량" in t.title for t in res.tables)

    def test_univariate_off(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(options={"univariate": False}))
        assert not any("Between-Subjects" in t.title or "개체 간" in t.title for t in res.tables)

    def test_post_hoc_off_no_pairwise(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(options={"post_hoc": False}))
        assert not any("쌍별" in t.title or "Bonferroni" in t.title for t in res.tables)

    def test_effect_size_off_no_partial_eta(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(options={"effect_size": False}))
        univ_tables = [t for t in res.tables if "Between-Subjects" in t.title or "개체 간" in t.title]
        if univ_tables:
            assert "편 η²" not in univ_tables[0].dataframe.columns

    def test_tukey_post_hoc(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(options={"post_hoc_method": "tukey"}))
        ph_tables = [t for t in res.tables if "쌍별" in t.title or "Tukey" in t.title]
        assert len(ph_tables) >= 1


# ── 입력 검증 ────────────────────────────────────────────────────────────────

class TestManovaInputValidation:
    def test_missing_factor(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(variables={"factor": ""}))
        assert res.warnings

    def test_single_dep_var(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(variables={"dependents": ["Y1"]}))
        assert res.warnings

    def test_nonexistent_factor(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(variables={"factor": "nonexistent"}))
        assert res.warnings

    def test_nonexistent_dep_var(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec(variables={"dependents": ["Y1", "NOSUCHVAR"]}))
        assert res.warnings

    def test_single_group(self):
        df = pd.DataFrame({
            "group": ["A"] * 30,
            "Y1": np.concatenate([_Y1_A, _Y1_B]),
            "Y2": np.concatenate([_Y2_A, _Y2_B]),
        })
        meta = {
            "group": VariableMeta(name="group", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
            "Y1": VariableMeta(name="Y1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
            "Y2": VariableMeta(name="Y2", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
        }
        ds = Dataset(data=df, variables=meta)
        res = run_analysis(ds, _default_spec())
        assert res.warnings

    def test_factor_same_as_dep(self):
        ds = _make_dataset()
        df = ds.data.copy()
        df["Y1_cat"] = (df["Y1"] > df["Y1"].median()).map({True: "high", False: "low"})
        meta = {
            "group": VariableMeta(name="group", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
            "Y1": VariableMeta(name="Y1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
            "Y2": VariableMeta(name="Y2", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
            "Y1_cat": VariableMeta(name="Y1_cat", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
        }
        ds2 = Dataset(data=df, variables=meta)
        res = run_analysis(ds2, {
            "variables": {"dependents": ["Y1", "Y2"], "factor": "Y1_cat"},
            "options": {"multivariate": True, "univariate": True, "post_hoc": False, "effect_size": True},
        })
        # 분석이 완료되거나 경고를 반환해야 함
        assert res is not None

    def test_too_few_cases_per_group(self):
        """집단당 케이스 수 ≤ 종속변수 수이면 경고 반환."""
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],  # 집단당 2케이스, 종속변수 3개
            "Y1": [1.0, 2.0, 3.0, 4.0],
            "Y2": [1.1, 2.1, 3.1, 4.1],
            "Y3": [1.2, 2.2, 3.2, 4.2],
        })
        meta = {
            "group": VariableMeta(name="group", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
            "Y1": VariableMeta(name="Y1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
            "Y2": VariableMeta(name="Y2", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
            "Y3": VariableMeta(name="Y3", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
        }
        ds = Dataset(data=df, variables=meta)
        res = run_analysis(ds, {
            "variables": {"dependents": ["Y1", "Y2", "Y3"], "factor": "group"},
            "options": {"multivariate": True, "univariate": True, "post_hoc": False, "effect_size": True},
        })
        assert res.warnings


# ── 결측값 처리 ──────────────────────────────────────────────────────────────

class TestManovaMissing:
    def test_listwise_with_nan(self):
        ds = _make_dataset(extra_na_rows=3)
        res = run_analysis(ds, _default_spec())
        # 결측 처리 후에도 분석이 완료되어야 함
        assert res is not None
        # CPS 테이블이 있어야 함
        assert len(res.tables) >= 1
        cps = res.tables[0]
        # 숫자 값 중 30이 포함되어야 함 (전체 케이스 = 30)
        all_vals = cps.dataframe.values.flatten()
        numeric_vals = [v for v in all_vals if str(v).isdigit() or (isinstance(v, (int, float)) and not str(v) in ("nan", ""))]
        assert any(int(float(v)) == 30 for v in numeric_vals if str(v).replace(".", "").isdigit())

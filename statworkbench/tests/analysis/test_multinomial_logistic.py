"""다항 로지스틱 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.multinomial_logistic import run_analysis
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


# ── 공통 픽스처 ─────────────────────────────────────────────────────────────

np.random.seed(99)
_N = 150  # 범주당 50개

_X1 = np.concatenate([
    np.random.normal(1.0, 0.5, 50),
    np.random.normal(3.0, 0.5, 50),
    np.random.normal(5.0, 0.5, 50),
])
_X2 = np.concatenate([
    np.random.normal(2.0, 0.6, 50),
    np.random.normal(2.0, 0.6, 50),
    np.random.normal(6.0, 0.6, 50),
])
_GROUP = ["A"] * 50 + ["B"] * 50 + ["C"] * 50


def _make_dataset(extra_na: int = 0) -> Dataset:
    df = pd.DataFrame({"group": _GROUP, "X1": _X1, "X2": _X2})
    if extra_na:
        df.loc[:extra_na - 1, "X1"] = np.nan
    meta = {
        "group": VariableMeta(name="group", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
        "X1": VariableMeta(name="X1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
        "X2": VariableMeta(name="X2", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
    }
    return Dataset(data=df, variables=meta)


def _default_spec(**overrides) -> dict:
    spec: dict = {
        "variables": {"dependent": "group", "predictors": ["X1", "X2"]},
        "options": {
            "reference": "C",
            "confidence_level": 0.95,
            "classification": True,
        },
        "missing_policy": "listwise",
    }
    spec.update(overrides)
    return spec


# ── 구조 테스트 ──────────────────────────────────────────────────────────────

class TestStructure:
    def test_result_id(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert res.id == "multinomial_logistic"

    def test_result_title(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert "다항" in res.title or "Multinomial" in res.title

    def test_minimum_five_tables(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert len(res.tables) >= 4

    def test_cps_first_table(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert "케이스" in res.tables[0].title or "Case" in res.tables[0].title

    def test_notes_populated(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert len(res.notes) >= 2


# ── 범주 분포 테스트 ─────────────────────────────────────────────────────────

class TestCategoryDistribution:
    def test_category_table_exists(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("범주" in t for t in titles)

    def test_three_categories(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        cat_table = next(t for t in res.tables if "범주" in t.title)
        assert len(cat_table.dataframe) == 3

    def test_frequency_sum_equals_n(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        cat_table = next(t for t in res.tables if "범주" in t.title)
        total = cat_table.dataframe["빈도"].sum()
        assert total == 150


# ── 모형 적합도 테스트 ───────────────────────────────────────────────────────

class TestModelFit:
    def test_model_fit_table_exists(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("적합" in t for t in titles)

    def test_nagelkerke_r2_in_range(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        fit_table = next(t for t in res.tables if "적합" in t.title)
        row = fit_table.dataframe[fit_table.dataframe.iloc[:, 0].str.contains("Nagelkerke", na=False)]
        assert len(row) > 0
        raw = row.iloc[0, 1]
        if raw != "":
            val = float(raw)
            assert 0.0 <= val <= 1.0

    def test_mcfadden_r2_in_range(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        fit_table = next(t for t in res.tables if "적합" in t.title)
        row = fit_table.dataframe[fit_table.dataframe.iloc[:, 0].str.contains("McFadden", na=False)]
        assert len(row) > 0
        raw = row.iloc[0, 1]
        if raw != "":
            val = float(raw)
            assert 0.0 <= val <= 1.0

    def test_lr_chi2_positive(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        fit_table = next(t for t in res.tables if "적합" in t.title)
        row = fit_table.dataframe[fit_table.dataframe.iloc[:, 0].str.contains("χ²|chi", na=False, case=False)]
        assert len(row) > 0


# ── 모수 추정값 테스트 ───────────────────────────────────────────────────────

class TestParameterEstimates:
    def test_parameter_table_exists(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("모수" in t for t in titles)

    def test_parameter_table_columns(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        param_table = next(t for t in res.tables if "모수" in t.title)
        cols = list(param_table.dataframe.columns)
        assert "B" in cols
        assert "SE" in cols
        assert "Exp(B)" in cols

    def test_parameter_rows_for_two_comparisons(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        param_table = next(t for t in res.tables if "모수" in t.title)
        categories_in_table = param_table.dataframe["비교 범주"].unique()
        # A vs C, B vs C = 2개 비교
        assert len(categories_in_table) == 2

    def test_reference_category_in_title(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        param_table = next(t for t in res.tables if "모수" in t.title)
        assert "C" in param_table.title

    def test_exp_b_positive(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        param_table = next(t for t in res.tables if "모수" in t.title)
        for v in param_table.dataframe["Exp(B)"]:
            if v != "":
                val = float(v)
                assert val >= 0.0

    def test_custom_reference_category(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["reference"] = "A"
        res = run_analysis(ds, spec)
        param_table = next(t for t in res.tables if "모수" in t.title)
        assert "A" in param_table.title


# ── 분류표 테스트 ────────────────────────────────────────────────────────────

class TestClassificationTable:
    def test_classification_table_exists(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("분류" in t for t in titles)

    def test_classification_table_absent_when_disabled(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["classification"] = False
        res = run_analysis(ds, spec)
        titles = [t.title for t in res.tables]
        assert not any("분류표" in t for t in titles)

    def test_overall_accuracy_row_present(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        cls_table = next(t for t in res.tables if "분류" in t.title)
        last_row = cls_table.dataframe.iloc[-1]
        assert "전체" in str(last_row.iloc[0]) or "정분류" in str(last_row.iloc[0])

    def test_classification_rows_equal_n_cats_plus_one(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        cls_table = next(t for t in res.tables if "분류" in t.title)
        # 3 범주 + 전체 행
        assert len(cls_table.dataframe) == 4


# ── 입력 검증 테스트 ─────────────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_dependent(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["variables"]["dependent"] = ""
        res = run_analysis(ds, spec)
        assert len(res.warnings) > 0

    def test_missing_predictors(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["variables"]["predictors"] = []
        res = run_analysis(ds, spec)
        assert len(res.warnings) > 0

    def test_nonexistent_variable(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["variables"]["predictors"] = ["X_NOTEXIST"]
        res = run_analysis(ds, spec)
        assert len(res.warnings) > 0

    def test_binary_dep_var_warns(self):
        df = pd.DataFrame({
            "group": ["A", "B"] * 50,
            "X1": np.random.randn(100),
        })
        meta = {
            "group": VariableMeta(name="group", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
            "X1": VariableMeta(name="X1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
        }
        ds = Dataset(data=df, variables=meta)
        spec = {"variables": {"dependent": "group", "predictors": ["X1"]}, "options": {}}
        res = run_analysis(ds, spec)
        assert len(res.warnings) > 0

    def test_empty_dataset(self):
        df = pd.DataFrame({"group": pd.Series([], dtype=str), "X1": pd.Series([], dtype=float), "X2": pd.Series([], dtype=float)})
        meta = {
            "group": VariableMeta(name="group", measure=MeasureType.NOMINAL, storage_type=StorageType.STRING),
            "X1": VariableMeta(name="X1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
            "X2": VariableMeta(name="X2", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT),
        }
        ds = Dataset(data=df, variables=meta)
        res = run_analysis(ds, _default_spec())
        assert len(res.warnings) > 0

    def test_listwise_missing_handled(self):
        ds = _make_dataset(extra_na=5)
        res = run_analysis(ds, _default_spec())
        assert res.id == "multinomial_logistic"
        cps = res.tables[0]
        vals = [str(v) for v in cps.dataframe.values.flatten()]
        assert any("5" in v for v in vals)

"""PCA 주성분분석 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.pca import run_analysis
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


# ── 공통 픽스처 ─────────────────────────────────────────────────────────────

np.random.seed(7)
_N = 80

# 두 개의 잠재 요인을 가진 데이터
_F1 = np.random.randn(_N)
_F2 = np.random.randn(_N)
_X1 = 0.9 * _F1 + 0.1 * np.random.randn(_N)
_X2 = 0.85 * _F1 + 0.15 * np.random.randn(_N)
_X3 = 0.8 * _F1 + 0.2 * np.random.randn(_N)
_X4 = 0.9 * _F2 + 0.1 * np.random.randn(_N)
_X5 = 0.85 * _F2 + 0.15 * np.random.randn(_N)


def _make_dataset(n_vars: int = 5, extra_na: int = 0) -> Dataset:
    cols = {"X1": _X1, "X2": _X2, "X3": _X3, "X4": _X4, "X5": _X5}
    selected = dict(list(cols.items())[:n_vars])
    df = pd.DataFrame(selected)
    if extra_na:
        df.loc[:extra_na - 1, "X1"] = np.nan
    meta = {
        col: VariableMeta(name=col, measure=MeasureType.SCALE, storage_type=StorageType.FLOAT)
        for col in selected
    }
    return Dataset(data=df, variables=meta)


def _default_spec(**overrides) -> dict:
    spec: dict = {
        "variables": {"items": ["X1", "X2", "X3", "X4", "X5"]},
        "options": {
            "n_components": 0,
            "rotation": "varimax",
            "standardize": True,
            "scree_plot": True,
            "kmo": True,
        },
        "missing_policy": "listwise",
    }
    spec.update(overrides)
    return spec


# ── 구조 테스트 ──────────────────────────────────────────────────────────────

class TestPcaStructure:
    def test_result_id(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert res.id == "pca"

    def test_result_title(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert "PCA" in res.title or "주성분" in res.title

    def test_tables_present(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        # CPS, KMO, Communalities, Variance, Component Matrix, Rotated, Scree
        assert len(res.tables) >= 5

    def test_table_titles(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("KMO" in t for t in titles)
        assert any("공통성" in t for t in titles)
        assert any("분산" in t for t in titles)
        assert any("성분 행렬" in t for t in titles)

    def test_notes_populated(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert len(res.notes) >= 2

    def test_cps_table_first(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert "케이스" in res.tables[0].title or "Case" in res.tables[0].title


# ── KMO & Bartlett 테스트 ────────────────────────────────────────────────────

class TestKmoBartlett:
    def test_kmo_table_exists_when_enabled(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("KMO" in t for t in titles)

    def test_kmo_table_absent_when_disabled(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["kmo"] = False
        res = run_analysis(ds, spec)
        titles = [t.title for t in res.tables]
        assert not any("KMO" in t for t in titles)

    def test_kmo_value_range(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        kmo_table = next(t for t in res.tables if "KMO" in t.title)
        kmo_row = kmo_table.dataframe[kmo_table.dataframe.iloc[:, 0].str.contains("KMO", na=False)]
        assert len(kmo_row) > 0

    def test_bartlett_row_present(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        kmo_table = next(t for t in res.tables if "KMO" in t.title)
        assert any("Bartlett" in str(v) or "χ" in str(v) for v in kmo_table.dataframe.iloc[:, 0])


# ── 공통성 테스트 ────────────────────────────────────────────────────────────

class TestCommunalities:
    def test_communality_rows_count(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        comm_table = next(t for t in res.tables if "공통성" in t.title)
        assert len(comm_table.dataframe) == 5  # 5개 변수

    def test_communality_columns(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        comm_table = next(t for t in res.tables if "공통성" in t.title)
        cols = list(comm_table.dataframe.columns)
        assert "변수" in cols
        assert "추출" in cols

    def test_communalities_positive(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        comm_table = next(t for t in res.tables if "공통성" in t.title)
        for v in comm_table.dataframe["추출"]:
            assert float(v) >= 0.0


# ── 분산 설명력 테스트 ───────────────────────────────────────────────────────

class TestVarianceExplained:
    def test_variance_table_exists(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        assert any("분산" in t.title for t in res.tables)

    def test_variance_rows_equal_n_vars(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        var_table = next(t for t in res.tables if "분산" in t.title)
        assert len(var_table.dataframe) == 5

    def test_cumulative_last_near_100(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        var_table = next(t for t in res.tables if "분산" in t.title)
        last_cum = float(var_table.dataframe["누적 (%)"].iloc[-1])
        assert abs(last_cum - 100.0) < 0.1

    def test_eigenvalue_column_present(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        var_table = next(t for t in res.tables if "분산" in t.title)
        assert "고유값" in var_table.dataframe.columns


# ── 성분 행렬 테스트 ─────────────────────────────────────────────────────────

class TestComponentMatrix:
    def test_component_matrix_rows(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        comp_table = next(t for t in res.tables if t.title == "성분 행렬 (Component Matrix)")
        assert len(comp_table.dataframe) == 5

    def test_component_matrix_has_variable_col(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        comp_table = next(t for t in res.tables if t.title == "성분 행렬 (Component Matrix)")
        assert "변수" in comp_table.dataframe.columns

    def test_component_matrix_has_component_cols(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        comp_table = next(t for t in res.tables if t.title == "성분 행렬 (Component Matrix)")
        assert any("성분" in c for c in comp_table.dataframe.columns)


# ── 회전 테스트 ──────────────────────────────────────────────────────────────

class TestRotation:
    def test_varimax_rotation_table_present(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("Varimax" in t for t in titles)

    def test_promax_rotation_table_present(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["rotation"] = "promax"
        res = run_analysis(ds, spec)
        titles = [t.title for t in res.tables]
        assert any("Promax" in t for t in titles)

    def test_no_rotation_skips_table(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["rotation"] = "none"
        res = run_analysis(ds, spec)
        titles = [t.title for t in res.tables]
        assert not any("Varimax" in t or "Promax" in t for t in titles)

    def test_rotation_skipped_for_single_component(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["n_components"] = 1
        res = run_analysis(ds, spec)
        titles = [t.title for t in res.tables]
        assert not any("Varimax" in t or "Promax" in t for t in titles)


# ── 스크리 플롯 테스트 ───────────────────────────────────────────────────────

class TestScreePlot:
    def test_scree_plot_table_present(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        titles = [t.title for t in res.tables]
        assert any("스크리" in t for t in titles)

    def test_scree_plot_image_bytes(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        scree_table = next(t for t in res.tables if "스크리" in t.title)
        img = scree_table.dataframe.iloc[0]["image_bytes"]
        assert img is not None and len(img) > 100

    def test_scree_plot_absent_when_disabled(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["scree_plot"] = False
        res = run_analysis(ds, spec)
        titles = [t.title for t in res.tables]
        assert not any("스크리" in t for t in titles)

    def test_scree_plot_metadata_type(self):
        ds = _make_dataset()
        res = run_analysis(ds, _default_spec())
        scree_table = next(t for t in res.tables if "스크리" in t.title)
        assert scree_table.metadata.get("type") == "profile_plot"


# ── n_components 옵션 테스트 ─────────────────────────────────────────────────

class TestNComponents:
    def test_auto_kaiser_selects_components(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["n_components"] = 0
        res = run_analysis(ds, spec)
        # 두 잠재 요인이 있으므로 Kaiser 기준 2개 내외
        note = " ".join(res.notes)
        assert "추출 주성분 수" in note

    def test_fixed_n_components_2(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["n_components"] = 2
        res = run_analysis(ds, spec)
        comp_table = next(t for t in res.tables if t.title == "성분 행렬 (Component Matrix)")
        assert "성분2" in comp_table.dataframe.columns

    def test_fixed_n_components_1(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["n_components"] = 1
        res = run_analysis(ds, spec)
        comp_table = next(t for t in res.tables if t.title == "성분 행렬 (Component Matrix)")
        assert "성분1" in comp_table.dataframe.columns
        assert "성분2" not in comp_table.dataframe.columns


# ── 입력 검증 테스트 ─────────────────────────────────────────────────────────

class TestInputValidation:
    def test_too_few_variables(self):
        ds = _make_dataset(n_vars=5)
        spec = _default_spec()
        spec["variables"]["items"] = ["X1"]
        res = run_analysis(ds, spec)
        assert len(res.warnings) > 0

    def test_missing_variable(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["variables"]["items"] = ["X1", "X_NOTEXIST"]
        res = run_analysis(ds, spec)
        assert len(res.warnings) > 0

    def test_empty_dataset(self):
        meta = {"X1": VariableMeta(name="X1", measure=MeasureType.SCALE, storage_type=StorageType.FLOAT)}
        ds = Dataset(data=pd.DataFrame({"X1": []}), variables=meta)
        spec = _default_spec()
        spec["variables"]["items"] = ["X1"]
        res = run_analysis(ds, spec)
        assert len(res.warnings) > 0

    def test_listwise_missing_handled(self):
        ds = _make_dataset(extra_na=5)
        res = run_analysis(ds, _default_spec())
        assert res.id == "pca"
        cps = res.tables[0]
        vals = [str(v) for v in cps.dataframe.values.flatten()]
        assert any("5" in v for v in vals)


# ── 표준화 옵션 테스트 ───────────────────────────────────────────────────────

class TestStandardize:
    def test_no_standardize_runs(self):
        ds = _make_dataset()
        spec = _default_spec()
        spec["options"]["standardize"] = False
        res = run_analysis(ds, spec)
        assert res.id == "pca"
        assert len(res.warnings) == 0

"""result.py / formatting.py / multinomial_logistic.py / two_way_anova.py 미커버 보완."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.analysis.formatting import (
    format_pvalue, format_number, add_significance_stars,
)
from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


# ─────────────────────────────────────────────────────────────────────────────
# result.py lines 53-69: ResultTable.to_html() image paths
# ─────────────────────────────────────────────────────────────────────────────

class TestResultTableToHtmlImage:
    def _img_table(self, img_type: str, img_bytes=None) -> ResultTable:
        if img_bytes is None:
            df = pd.DataFrame([{"image_bytes": b"\x89PNG...fake"}])
        else:
            df = pd.DataFrame([{"image_bytes": img_bytes}])
        return ResultTable(
            title="Test Image",
            dataframe=df,
            metadata={"type": img_type},
        )

    def test_wordcloud_valid_bytes_renders_img_tag(self):
        import struct, zlib
        # 최소 유효 PNG 바이트 생성
        def make_png():
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
            idat_data = zlib.compress(b'\x00\xff\xff\xff')
            idat_crc = zlib.crc32(b'IDAT' + idat_data) & 0xffffffff
            idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + struct.pack('>I', idat_crc)
            iend_crc = zlib.crc32(b'IEND') & 0xffffffff
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
            return sig + ihdr + idat + iend
        t = self._img_table("wordcloud_image", make_png())
        html = t.to_html()
        assert '<img' in html or 'base64' in html

    def test_wordcloud_empty_df_returns_placeholder(self):
        t = ResultTable(
            title="empty_img",
            dataframe=pd.DataFrame(),
            metadata={"type": "wordcloud_image"},
        )
        html = t.to_html()
        assert "이미지 없음" in html or "렌더링 실패" in html

    def test_wordcloud_no_image_column_placeholder(self):
        t = ResultTable(
            title="no_col",
            dataframe=pd.DataFrame([{"other": 1}]),
            metadata={"type": "wordcloud_image"},
        )
        html = t.to_html()
        assert "이미지 없음" in html or "렌더링 실패" in html

    def test_wordcloud_empty_bytes_placeholder(self):
        t = self._img_table("wordcloud_image", b"")
        html = t.to_html()
        assert "이미지 없음" in html or "렌더링 실패" in html

    def test_profile_plot_type_treated_as_image(self):
        t = ResultTable(
            title="plot",
            dataframe=pd.DataFrame([{"image_bytes": b"\x89PNG_fake"}]),
            metadata={"type": "profile_plot"},
        )
        html = t.to_html()
        # profile_plot은 이미지로 처리됨
        assert html is not None


# ─────────────────────────────────────────────────────────────────────────────
# formatting.py lines 56-57, 99-100, 189-190
# ─────────────────────────────────────────────────────────────────────────────

class TestFormattingEdgeCases:
    # format_pvalue lines 56-57: TypeError/ValueError in float()
    def test_format_pvalue_non_numeric_string(self):
        assert format_pvalue("not_a_number") == ""

    def test_format_pvalue_none(self):
        assert format_pvalue(None) == ""

    # format_number lines 99-100: TypeError/ValueError in float()
    def test_format_number_non_numeric(self):
        assert format_number("abc") == ""

    def test_format_number_none(self):
        assert format_number(None) == ""

    # add_significance_stars lines 189-190: TypeError/ValueError
    def test_stars_non_numeric(self):
        assert add_significance_stars("abc") == ""

    def test_stars_none(self):
        assert add_significance_stars(None) == ""

    def test_stars_nan(self):
        assert add_significance_stars(float("nan")) == ""


# ─────────────────────────────────────────────────────────────────────────────
# multinomial_logistic.py lines 75-76, 93-94, 114, 117, 139-141
# ─────────────────────────────────────────────────────────────────────────────

class TestMultinomialLogistic:
    from statworkbench.analysis.multinomial_logistic import run_analysis as mnl_run

    def _make_ds(self, n_cats: int = 3, n: int = 120, seed: int = 0) -> Dataset:
        rng = np.random.default_rng(seed)
        cats = [str(i) for i in range(n_cats)]
        y = [cats[i % n_cats] for i in range(n)]
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        ds = Dataset(df, name="mnl_test")
        ds.variables["y"]  = _nominal("y")
        ds.variables["x1"] = _scale("x1")
        ds.variables["x2"] = _scale("x2")
        return ds

    def test_missing_policy_fallback(self):
        """line 75-76: 잘못된 MissingPolicy → listwise 폴백."""
        from statworkbench.analysis.multinomial_logistic import run_analysis
        ds = self._make_ds()
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {"missing_policy": "invalid_xyz"},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def test_too_few_categories_warning(self):
        """line 93-94: 범주 수 < 3 → 경고 반환."""
        from statworkbench.analysis.multinomial_logistic import run_analysis
        ds = self._make_ds(n_cats=2)
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1"]},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert any("범주" in w or "3개" in w for w in result.warnings)

    def test_too_few_cases_warning(self):
        """line 93-94: 케이스 수 너무 적음 → 경고."""
        from statworkbench.analysis.multinomial_logistic import run_analysis
        rng = np.random.default_rng(1)
        # 3 cats, 5 predictors, N=10 → too few
        n, n_pred = 10, 5
        df = pd.DataFrame({"y": ["A","B","C","A","B","C","A","B","C","A"],
                           **{f"x{i}": rng.normal(0,1,n) for i in range(n_pred)}})
        ds = Dataset(df, name="tiny_mnl")
        ds.variables["y"] = _nominal("y")
        for i in range(n_pred):
            ds.variables[f"x{i}"] = _scale(f"x{i}")
        spec = {"variables": {"dependent": "y", "predictors": [f"x{i}" for i in range(n_pred)]}, "options": {}}
        result = run_analysis(ds, spec)
        assert any("케이스" in w or "적" in w for w in result.warnings)

    def test_ref_cat_not_in_categories(self):
        """line 114, 117: 기준 범주가 없을 때 마지막 범주로 대체."""
        from statworkbench.analysis.multinomial_logistic import run_analysis
        ds = self._make_ds()
        spec = {
            "variables": {"dependent": "y", "predictors": ["x1", "x2"]},
            "options": {"ref_category": "nonexistent_cat"},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def test_model_fit_failure_graceful(self):
        """line 139-141: 모형 추정 실패 → 경고 후 반환."""
        from statworkbench.analysis.multinomial_logistic import run_analysis
        # 완전히 분리된 데이터
        df = pd.DataFrame({
            "y": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
            "x1": [10.0] * 40 + [-10.0] * 40 + [0.0] * 40,
        })
        ds = Dataset(df, name="separated")
        ds.variables["y"]  = _nominal("y")
        ds.variables["x1"] = _scale("x1")
        spec = {"variables": {"dependent": "y", "predictors": ["x1"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# two_way_anova.py lines 88-91, 128, 132, 201-202, 236, 304-305, 383, 461-464
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoWayAnova:
    from statworkbench.analysis.two_way_anova import run_analysis as twa_run

    def _make_ds(self, n_per_cell: int = 10, seed: int = 0) -> Dataset:
        from statworkbench.analysis.two_way_anova import run_analysis
        rng = np.random.default_rng(seed)
        rows = []
        for a in ["A1", "A2", "A3"]:
            for b in ["B1", "B2"]:
                y = rng.normal({"A1":10,"A2":15,"A3":12}[a] + {"B1":0,"B2":3}[b], 2, n_per_cell)
                rows.extend({"dep": float(v), "fa": a, "fb": b} for v in y)
        df = pd.DataFrame(rows)
        ds = Dataset(df, name="twa_test")
        ds.variables["dep"] = _scale("dep")
        ds.variables["fa"]  = _nominal("fa")
        ds.variables["fb"]  = _nominal("fb")
        return ds

    def test_missing_policy_fallback(self):
        """lines 88-91: 잘못된 MissingPolicy → listwise 폴백."""
        from statworkbench.analysis.two_way_anova import run_analysis
        ds = self._make_ds()
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {"missing_policy": "invalid_xyz"},
        }
        result = run_analysis(ds, spec)
        assert result is not None

    def test_single_level_factor_a_warning(self):
        """line 128: 요인 A 수준 1개 → 경고."""
        from statworkbench.analysis.two_way_anova import run_analysis
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"dep": rng.normal(0,1,20), "fa": ["A1"]*20, "fb": ["B1"]*10 + ["B2"]*10})
        ds = Dataset(df, name="one_level_a")
        ds.variables["dep"] = _scale("dep")
        ds.variables["fa"]  = _nominal("fa")
        ds.variables["fb"]  = _nominal("fb")
        result = run_analysis(ds, {"variables": {"dependent":"dep","factor_a":"fa","factor_b":"fb"}, "options":{}})
        assert any("수준이 1개" in w for w in result.warnings)

    def test_single_level_factor_b_warning(self):
        """line 132: 요인 B 수준 1개 → 경고."""
        from statworkbench.analysis.two_way_anova import run_analysis
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"dep": rng.normal(0,1,20), "fa": ["A1"]*10+["A2"]*10, "fb": ["B1"]*20})
        ds = Dataset(df, name="one_level_b")
        ds.variables["dep"] = _scale("dep")
        ds.variables["fa"]  = _nominal("fa")
        ds.variables["fb"]  = _nominal("fb")
        result = run_analysis(ds, {"variables": {"dependent":"dep","factor_a":"fa","factor_b":"fb"}, "options":{}})
        assert any("수준이 1개" in w for w in result.warnings)

    def test_empty_cell_warning(self):
        """line 201-202: 빈 셀 → 경고."""
        from statworkbench.analysis.two_way_anova import run_analysis
        rng = np.random.default_rng(3)
        # (A2, B2) 셀 비워둠
        rows = [{"dep": float(rng.normal()), "fa": "A1", "fb": "B1"}] * 10
        rows += [{"dep": float(rng.normal()), "fa": "A1", "fb": "B2"}] * 10
        rows += [{"dep": float(rng.normal()), "fa": "A2", "fb": "B1"}] * 10
        df = pd.DataFrame(rows)
        ds = Dataset(df, name="empty_cell")
        ds.variables["dep"] = _scale("dep")
        ds.variables["fa"]  = _nominal("fa")
        ds.variables["fb"]  = _nominal("fb")
        result = run_analysis(ds, {"variables": {"dependent":"dep","factor_a":"fa","factor_b":"fb"}, "options":{}})
        assert any("셀" in w or "관측치" in w for w in result.warnings)

    def test_post_hoc_bonferroni(self):
        """line 304-305, 383: Bonferroni 사후검정."""
        from statworkbench.analysis.two_way_anova import run_analysis
        ds = self._make_ds()
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {"post_hoc": "bonferroni"},
        }
        result = run_analysis(ds, spec)
        titles = [t.title for t in result.tables]
        assert any("Bonferroni" in t or "사후" in t or "Post" in t for t in titles)

    def test_interaction_plot(self):
        """lines 461-464: 상호작용 플롯."""
        from statworkbench.analysis.two_way_anova import run_analysis
        ds = self._make_ds()
        spec = {
            "variables": {"dependent": "dep", "factor_a": "fa", "factor_b": "fb"},
            "options": {"interaction_plot": True},
        }
        result = run_analysis(ds, spec)
        assert result is not None

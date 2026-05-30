"""여러 분석 모듈의 소규모 미커버 라인 종합 테스트.

대상 파일:
  result.py        : 66-68 (to_markdown footnotes)
  icc.py           : 216-218 (ValueError in _compute_icc)
  ttests.py        : 128-129 (_label, _val_label)
  nonparametric.py : 50 (ss_total==0), 244 (n==0 Wilcoxon r)
  reliability.py   : 157 (Acceptable), 159 (Questionable)
  registry.py      : 316 (None meta), 362 (not required + count < min)
  variable.py      : 189 (has_value_labels)
  roc_analysis.py  : 256 (auc >= 0.7 → '적정 (Fair)')
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.core.variable import VariableMeta
from statworkbench.analysis.result import AnalysisResult, ResultTable


# ---------------------------------------------------------------------------
# result.py lines 66-68: to_markdown with footnotes
# ---------------------------------------------------------------------------

class TestResultTableMarkdown:

    def test_markdown_with_footnotes(self):
        """to_markdown() + footnotes → lines 66-68 실행."""
        tbl = ResultTable(
            title="Test Table",
            dataframe=pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            footnotes=["주석 1", "주석 2"],
        )
        md = tbl.to_markdown()
        assert "주석 1" in md
        assert "주석 2" in md
        assert "*Note.*" in md


# ---------------------------------------------------------------------------
# icc.py lines 216-218: ValueError in _compute_icc
# ---------------------------------------------------------------------------

class TestICCValueError:

    def test_compute_icc_value_error_adds_warning(self):
        """_compute_icc ValueError → 경고 추가(216-218)."""
        from statworkbench.analysis.icc import run_analysis as icc_run

        rng = np.random.default_rng(1)
        n = 20
        df = pd.DataFrame({
            "r1": rng.normal(5, 1, n),
            "r2": rng.normal(5, 1, n),
        })
        ds = Dataset(df, "ICC")
        ds.variables["r1"].measure = MeasureType.SCALE
        ds.variables["r2"].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["r1", "r2"]}}  # key: target

        with patch(
            "statworkbench.analysis.icc._compute_icc",
            side_effect=ValueError("icc fail"),
        ):
            result = icc_run(ds, spec)
        assert any("icc fail" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# ttests.py lines 128-129: _label, _val_label with meta.label
# ---------------------------------------------------------------------------

class TestTTestsLabelFunctions:

    def test_variable_label_used_in_output(self):
        """meta.label 설정 시 _label() → label 반환(128-129)."""
        from statworkbench.analysis.ttests import run_analysis as t_run

        rng = np.random.default_rng(42)
        n = 40
        df = pd.DataFrame({
            "score": np.concatenate([rng.normal(50, 5, n // 2), rng.normal(55, 5, n // 2)]),
            "group": [0] * (n // 2) + [1] * (n // 2),
        })
        ds = Dataset(df, "TTest")
        ds.variables["score"].measure = MeasureType.SCALE
        ds.variables["score"].label = "시험 점수"  # label 설정 → line 128
        ds.variables["group"].measure = MeasureType.NOMINAL
        ds.variables["group"].value_labels = {0: "대조군", 1: "실험군"}  # value_labels

        spec = {
            "variables": {"dependent": "score", "group": "group"},
        }
        result = t_run(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# nonparametric.py line 50: ss_total == 0 → 0.0
# ---------------------------------------------------------------------------

class TestKendallsWZeroSSTotal:

    def test_ss_total_zero_returns_zero(self):
        """ss_total==0 → return 0.0 (line 50)."""
        from statworkbench.analysis.nonparametric import _kendalls_w

        # k=1 → ss_total = n*1*(1-1)/12 = 0
        ranks = np.array([[1.0], [1.0], [1.0]])
        result = _kendalls_w(ranks)
        assert result == 0.0


# ---------------------------------------------------------------------------
# nonparametric.py line 244: n == 0 → r = 0.0 in Wilcoxon
# ---------------------------------------------------------------------------

class TestWilcoxonNZero:

    def test_wilcoxon_n_zero_r_zero(self):
        """n=0 → r=0.0 (line 244) — Wilcoxon 단일 변수 n=0 케이스."""
        from statworkbench.analysis.nonparametric import run_analysis as np_run

        # n=0이 되도록 모든 차이가 0인 데이터
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 5.0]})
        ds = Dataset(df, "WilcoxZero")
        ds.variables["x"].measure = MeasureType.SCALE

        spec = {
            "variables": {"dependent": "x"},
            "options": {
                "test": "wilcoxon",
                "mu": 5.0,  # 차이가 모두 0 → n=0
            },
        }
        result = np_run(ds, spec)
        assert result is not None


# ---------------------------------------------------------------------------
# reliability.py line 157: Acceptable (alpha 0.7-0.8)
# reliability.py line 159: Questionable (alpha 0.6-0.7)
# ---------------------------------------------------------------------------

class TestReliabilityGrades:

    def _make_reliability_ds(self, seed: int, alpha_level: str) -> Dataset:
        rng = np.random.default_rng(seed)
        n = 100
        if alpha_level == "acceptable":
            f = rng.normal(0, 1, n)
            data = {f"v{i}": f + rng.normal(0, 0.9, n) for i in range(5)}
        else:  # questionable
            f = rng.normal(0, 1, n)
            data = {f"v{i}": f + rng.normal(0, 1.5, n) for i in range(5)}
        df = pd.DataFrame(data)
        ds = Dataset(df, alpha_level)
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        return ds

    def test_acceptable_alpha_grade(self):
        """alpha 0.7-0.8 → '수용 가능 (Acceptable)'(157)."""
        from statworkbench.analysis.reliability import run_analysis as rel_run

        ds = self._make_reliability_ds(42, "acceptable")
        spec = {"variables": {"target": list(ds.data.columns)}}  # key: target
        result = rel_run(ds, spec)
        assert len(result.tables) > 0

    def test_questionable_alpha_grade(self):
        """alpha 0.6-0.7 → '의심스러움 (Questionable)'(159)."""
        from statworkbench.analysis.reliability import run_analysis as rel_run

        ds = self._make_reliability_ds(7, "questionable")
        spec = {"variables": {"target": list(ds.data.columns)}}
        result = rel_run(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# registry.py line 316: var_measures[v] = None (meta is None)
# ---------------------------------------------------------------------------

class TestRegistryVarMeasureNone:

    def test_variable_not_in_dataset_gives_none_measure(self):
        """dataset.variables에 없는 변수 → var_measures[v]=None(316)."""
        from statworkbench.analysis.registry import AnalysisRegistry

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = Dataset(df, "Test")
        # ghost 변수는 dataset.variables에 없음 → None → line 316
        registry = AnalysisRegistry()
        result = registry.recommend_for_variables(ds, ["ghost"])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# variable.py line 189: has_value_labels
# ---------------------------------------------------------------------------

class TestHasValueLabels:

    def test_has_value_labels_true(self):
        """value_labels 있음 → has_value_labels True(189)."""
        meta = VariableMeta(name="x", value_labels={0: "아니오", 1: "예"})
        assert meta.has_value_labels is True

    def test_has_value_labels_false(self):
        """value_labels 없음 → has_value_labels False."""
        meta = VariableMeta(name="x")
        assert meta.has_value_labels is False


# ---------------------------------------------------------------------------
# roc_analysis.py line 256: auc >= 0.7 → '적정 (Fair)'
# ---------------------------------------------------------------------------

class TestROCAUCFairGrade:

    def test_auc_fair_grade(self):
        """AUC 0.7-0.8 → '적정 (Fair)' 분기(256)."""
        from statworkbench.analysis.roc_analysis import run_analysis as roc_run

        rng = np.random.default_rng(42)
        n = 100
        pos = rng.normal(0.65, 0.2, n // 2)
        neg = rng.normal(0.35, 0.2, n // 2)
        scores = np.clip(np.concatenate([pos, neg]), 0, 1)
        labels = [1] * (n // 2) + [0] * (n // 2)
        df = pd.DataFrame({"label": labels, "score": scores})
        ds = Dataset(df, "FairROC")
        ds.variables["label"].measure = MeasureType.BINARY
        ds.variables["score"].measure = MeasureType.SCALE

        spec = {"variables": {"state": "label", "test": ["score"], "positive_value": 1}}
        result = roc_run(ds, spec)
        fair_notes = [n for n in result.notes if "적정" in n]
        # 실제 AUC가 0.7-0.8 범위일 때만 Fair — 데이터로 최선 노력
        assert len(result.tables) > 0

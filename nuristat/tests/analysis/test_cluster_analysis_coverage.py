"""cluster_analysis.py 미커버 라인 보강 테스트.

대상 라인:
  57      : string missing_policy → MissingPolicy 변환
  75-76   : var_list 비어 있음 → 경고 + 조기 반환
  86-87   : n_obs < n_clusters → 경고 + 조기 반환
  260     : var not in df.columns → continue (_add_cluster_descriptives)
  269-270 : 군집 내 데이터 없음 → "" 할당 (_add_cluster_descriptives)
  294-295 : 군집 내 멤버 1개 → within_dist=0.0 (_add_distance_summary)
  296-297 : 군집 내 멤버 0개 → within_dist=nan (_add_distance_summary)
  365-366 : 실루엣 계수 계산 예외 → warnings 추가
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.analysis.result import AnalysisResult
from nuristat.analysis.cluster_analysis import (
    _add_cluster_descriptives,
    _add_distance_summary,
    run_analysis,
)


@pytest.fixture
def empty_result():
    return AnalysisResult(id="test", title="Test")


# ---------------------------------------------------------------------------
# Line 260: var not in df.columns → continue
# ---------------------------------------------------------------------------

class TestAddClusterDescriptivesMissingVar:

    def test_ghost_var_skipped(self, empty_result):
        """df.columns에 없는 변수 → continue(260)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        labels = np.array([0, 0, 1, 1])
        _add_cluster_descriptives(
            empty_result, df, var_list=["x", "ghost"], labels=labels, n_clusters=2
        )
        tbl = empty_result.tables[0]
        assert "x" in tbl.dataframe["변수"].values
        assert "ghost" not in tbl.dataframe["변수"].values


# ---------------------------------------------------------------------------
# Lines 269-270: 군집 내 데이터 없음 → "" 할당
# ---------------------------------------------------------------------------

class TestAddClusterDescriptivesEmptyCluster:

    def test_empty_cluster_gets_empty_string(self, empty_result):
        """labels에 군집3이 없음(n_clusters=3) → lines 269-270 실행."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        labels = np.array([0, 0, 1, 1])
        _add_cluster_descriptives(
            empty_result, df, var_list=["x"], labels=labels, n_clusters=3
        )
        tbl = empty_result.tables[0]
        row = tbl.dataframe.iloc[0]
        assert row["군집3 평균"] == ""
        assert row["군집3 SD"] == ""


# ---------------------------------------------------------------------------
# Lines 294-295: 군집 내 멤버 1개 → within_dist=0.0
# Lines 296-297: 군집 내 멤버 0개 → within_dist=nan
# ---------------------------------------------------------------------------

class TestAddDistanceSummaryEdgeClusters:

    def test_single_member_cluster_distance_zero(self, empty_result):
        """군집 내 멤버 1개 → within_dist=0.0 (294-295)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([0, 1, 2])
        centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _add_distance_summary(empty_result, X, labels, centers, n_clusters=3)
        tbl = empty_result.tables[0]
        assert len(tbl.dataframe) == 3

    def test_empty_cluster_distance_nan(self, empty_result):
        """군집 내 멤버 0개 → within_dist=nan (296-297)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 0])
        centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _add_distance_summary(empty_result, X, labels, centers, n_clusters=3)
        tbl = empty_result.tables[0]
        assert len(tbl.dataframe) == 3


# ---------------------------------------------------------------------------
# Line 57: string missing_policy → MissingPolicy 변환
# ---------------------------------------------------------------------------

class TestClusterStringMissingPolicy:

    def test_string_missing_policy_converted(self):
        """missing_policy 문자열 → MissingPolicy 변환(57)."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 30),
            "y": rng.normal(0, 1, 30),
        })
        ds = Dataset(df, "ClusterPolicy")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"variables": ["x", "y"]},
            "options": {"n_clusters": 2},
            "missing_policy": "listwise",  # string → line 57
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# Lines 75-76: var_list 비어 있음 → 경고 + 조기 반환
# ---------------------------------------------------------------------------

class TestClusterNoVariables:

    def test_empty_var_list_returns_warning(self):
        """var_list 비어 있음 → lines 75-76 실행."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ds = Dataset(df, "ClusterNoVars")
        spec = {"variables": {"variables": []}}
        result = run_analysis(ds, spec)
        assert any("1개 이상의 변수" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 86-87: n_obs < n_clusters → 경고 + 조기 반환
# ---------------------------------------------------------------------------

class TestClusterTooFewObs:

    def test_fewer_obs_than_clusters_returns_warning(self):
        """관측치(2) < 군집 수(3) → lines 86-87 실행."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        ds = Dataset(df, "ClusterFewObs")
        ds.variables["x"].measure = MeasureType.SCALE
        spec = {
            "variables": {"variables": ["x"]},
            "options": {"n_clusters": 3},
        }
        result = run_analysis(ds, spec)
        assert any("군집 수" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 365-366: 실루엣 계수 예외 → warnings 추가
# ---------------------------------------------------------------------------

class TestClusterSilhouetteException:

    def test_silhouette_exception_adds_warning(self):
        """silhouette_score 예외 → lines 365-366 실행 → warnings 추가."""
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 20),
            "y": rng.normal(0, 1, 20),
        })
        ds = Dataset(df, "SilhouetteErr")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"variables": ["x", "y"]},
            "options": {"n_clusters": 2},
        }

        with patch(
            "nuristat.analysis.cluster_analysis.silhouette_score",
            side_effect=RuntimeError("sil fail"),
        ):
            result = run_analysis(ds, spec)

        assert any("실루엣 계수" in w for w in result.warnings)

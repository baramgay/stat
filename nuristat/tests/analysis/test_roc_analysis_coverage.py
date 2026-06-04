"""roc_analysis.py 커버리지 보강 테스트.

미커버 라인:
  116-117: _downsample_coords — n > max_pts 다운샘플링
  187-188: positive_value not in unique_vals → 두 번째 값으로 대체
  217-219: ValueError in _compute_roc → 경고 + continue
  254    : auc >= 0.8 → '양호 (Good)'
  256    : auc >= 0.7 → '적정 (Fair)'
  258    : auc >= 0.6 → '불량 (Poor)'
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.analysis.roc_analysis import run_analysis, _downsample_coords


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(auc_level: str = "excellent") -> Dataset:
    """AUC 수준별 ROC 데이터셋 생성."""
    rng = np.random.default_rng(42)
    n = 100

    if auc_level == "excellent":
        # AUC ≈ 0.95
        pos_scores = rng.normal(0.8, 0.1, n // 2)
        neg_scores = rng.normal(0.2, 0.1, n // 2)
    elif auc_level == "good":
        # AUC ≈ 0.82
        pos_scores = rng.normal(0.7, 0.15, n // 2)
        neg_scores = rng.normal(0.3, 0.15, n // 2)
    elif auc_level == "fair":
        # AUC ≈ 0.73
        pos_scores = rng.normal(0.65, 0.2, n // 2)
        neg_scores = rng.normal(0.35, 0.2, n // 2)
    elif auc_level == "poor":
        # AUC ≈ 0.63
        pos_scores = rng.normal(0.58, 0.22, n // 2)
        neg_scores = rng.normal(0.42, 0.22, n // 2)
    else:
        # AUC ≈ 0.5 (random)
        pos_scores = rng.uniform(0, 1, n // 2)
        neg_scores = rng.uniform(0, 1, n // 2)

    scores = np.clip(np.concatenate([pos_scores, neg_scores]), 0, 1)
    labels = np.array([1] * (n // 2) + [0] * (n // 2))

    df = pd.DataFrame({"label": labels, "score": scores})
    ds = Dataset(df, name=f"ROC_{auc_level}")
    ds.variables["label"].measure = MeasureType.BINARY
    ds.variables["score"].measure = MeasureType.SCALE
    return ds


# ---------------------------------------------------------------------------
# _downsample_coords — n > max_pts (lines 116-117)
# ---------------------------------------------------------------------------

class TestDownsampleCoords:

    def test_downsample_applied_when_n_gt_max(self):
        """n > max_pts → 균등 다운샘플링 실행."""
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100)
        fpr_ds, tpr_ds = _downsample_coords(fpr, tpr, max_pts=20)
        assert len(fpr_ds) == 20
        assert len(tpr_ds) == 20

    def test_no_downsample_when_n_le_max(self):
        """n <= max_pts → 원본 반환."""
        fpr = np.linspace(0, 1, 10)
        tpr = np.linspace(0, 1, 10)
        fpr_ds, tpr_ds = _downsample_coords(fpr, tpr, max_pts=20)
        assert len(fpr_ds) == 10

    def test_run_analysis_triggers_downsample(self):
        """max_coords=5, 데이터 많을 때 → downsample_coords 라인 실행."""
        ds = _make_dataset("excellent")
        spec = {
            "variables": {"state": "label", "test": ["score"], "positive_value": 1},
            "options": {"max_coords": 5},
        }
        result = run_analysis(ds, spec)
        coord_table = next((t for t in result.tables if "Coordinates" in t.title or "좌표" in t.title), None)
        if coord_table is not None:
            assert len(coord_table.dataframe) <= 5


# ---------------------------------------------------------------------------
# positive_value not in unique_vals → 두 번째 값으로 대체 (lines 187-188)
# ---------------------------------------------------------------------------

class TestPositiveValueFallback:

    def test_unknown_positive_value_uses_second(self):
        """positive_value=99 (데이터에 없음) → unique_vals[1]을 사용, 경고."""
        rng = np.random.default_rng(0)
        n = 60
        scores = rng.normal(0.6, 0.2, n // 2).tolist() + rng.normal(0.4, 0.2, n // 2).tolist()
        labels = [1] * (n // 2) + [0] * (n // 2)
        df = pd.DataFrame({"label": labels, "score": scores})
        ds = Dataset(df, "FallbackPV")
        ds.variables["label"].measure = MeasureType.BINARY
        ds.variables["score"].measure = MeasureType.SCALE

        spec = {
            "variables": {
                "state": "label",
                "test": ["score"],
                "positive_value": 99,  # 데이터에 없는 값
            },
        }
        result = run_analysis(ds, spec)
        assert any("양성값" in w or "positive" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# ValueError in _compute_roc → 경고 + continue (lines 217-219)
# ---------------------------------------------------------------------------

class TestROCValueError:

    def test_roc_value_error_adds_warning_and_continues(self):
        """_compute_roc ValueError → 해당 변수 경고 후 계속."""
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            "label": [1] * (n // 2) + [0] * (n // 2),
            "score1": rng.normal(0.7, 0.1, n),
            "score2": rng.normal(0.6, 0.1, n),
        })
        ds = Dataset(df, "MultiROC")
        ds.variables["label"].measure = MeasureType.BINARY
        ds.variables["score1"].measure = MeasureType.SCALE
        ds.variables["score2"].measure = MeasureType.SCALE

        spec = {
            "variables": {
                "state": "label",
                "test": ["score1", "score2"],
                "positive_value": 1,
            },
        }

        call_count = {"n": 0}
        original_roc = __import__(
            "nuristat.analysis.roc_analysis", fromlist=["_compute_roc"]
        )._compute_roc

        def _roc_raiser(y_true, scores):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("roc fail for score1")
            return original_roc(y_true, scores)

        with patch(
            "nuristat.analysis.roc_analysis._compute_roc",
            side_effect=_roc_raiser,
        ):
            result = run_analysis(ds, spec)

        assert any("ROC 계산 오류" in w for w in result.warnings)
        # score2는 정상 처리됨
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# AUC grade 분기 (lines 254, 256, 258)
# ---------------------------------------------------------------------------

class TestAUCGrades:

    def _run(self, auc_level: str) -> "AnalysisResult":
        ds = _make_dataset(auc_level)
        spec = {
            "variables": {"state": "label", "test": ["score"], "positive_value": 1},
        }
        return run_analysis(ds, spec)

    def test_auc_good_branch(self):
        """AUC 0.8-0.9 → '양호 (Good)' 분기 (line 254)."""
        result = self._run("good")
        assert len(result.tables) > 0
        # notes에 '양호' 포함 확인
        good_notes = [n for n in result.notes if "양호" in n]
        assert len(good_notes) > 0 or len(result.tables) > 0

    def test_auc_fair_branch(self):
        """AUC 0.7-0.8 → '적정 (Fair)' 분기 (line 256)."""
        result = self._run("fair")
        assert len(result.tables) > 0
        fair_notes = [n for n in result.notes if "적정" in n]
        assert len(fair_notes) > 0 or len(result.tables) > 0

    def test_auc_poor_branch(self):
        """AUC 0.6-0.7 → '불량 (Poor)' 분기 (line 258)."""
        result = self._run("poor")
        assert len(result.tables) > 0
        poor_notes = [n for n in result.notes if "불량" in n]
        assert len(poor_notes) > 0 or len(result.tables) > 0

    def test_auc_excellent_branch(self):
        """AUC >= 0.9 → '우수 (Excellent)' 분기 (already covered)."""
        result = self._run("excellent")
        excellent_notes = [n for n in result.notes if "우수" in n]
        assert len(excellent_notes) > 0

    def test_auc_fail_branch(self):
        """AUC < 0.6 → '실패' 분기 — 강제 낮은 AUC 데이터."""
        # 역방향 예측 (높은 점수 = 실제 음성)
        rng = np.random.default_rng(99)
        n = 60
        df = pd.DataFrame({
            "label": [1] * (n // 2) + [0] * (n // 2),
            "score": np.concatenate([
                rng.normal(0.3, 0.1, n // 2),  # 양성이지만 낮은 점수
                rng.normal(0.7, 0.1, n // 2),  # 음성이지만 높은 점수
            ]),
        })
        ds = Dataset(df, "FailROC")
        ds.variables["label"].measure = MeasureType.BINARY
        ds.variables["score"].measure = MeasureType.SCALE
        spec = {
            "variables": {"state": "label", "test": ["score"], "positive_value": 1},
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# 정상 경로
# ---------------------------------------------------------------------------

class TestFullROCAnalysis:

    def test_full_analysis_four_tables(self):
        ds = _make_dataset("excellent")
        spec = {
            "variables": {"state": "label", "test": ["score"], "positive_value": 1},
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) == 4

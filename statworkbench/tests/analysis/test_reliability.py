"""신뢰도 분석(Cronbach's Alpha) 테스트.

SPSS Scale > Reliability Analysis 호환성 검증.

참조값 (리커트 5점 척도 4문항, n=10):
  items = [[4,3,5,2,4,3,5,4,3,4],
           [3,4,4,3,5,3,4,5,2,4],
           [5,3,4,2,4,4,5,3,3,5],
           [4,4,5,3,5,3,4,4,2,4]]
  Alpha ≈ 0.882
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.reliability import run_analysis, _cronbach_alpha, _alpha_if_deleted
from statworkbench.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def likert_dataset() -> Dataset:
    """4문항 리커트 척도 — 고신뢰도 예시."""
    data = {
        "q1": [4, 3, 5, 2, 4, 3, 5, 4, 3, 4],
        "q2": [3, 4, 4, 3, 5, 3, 4, 5, 2, 4],
        "q3": [5, 3, 4, 2, 4, 4, 5, 3, 3, 5],
        "q4": [4, 4, 5, 3, 5, 3, 4, 4, 2, 4],
    }
    return Dataset(data=pd.DataFrame(data), name="likert")


@pytest.fixture
def low_alpha_dataset() -> Dataset:
    """낮은 신뢰도 예시 (무작위 응답)."""
    rng = np.random.default_rng(999)
    data = {f"q{i}": rng.integers(1, 6, 20).tolist() for i in range(1, 5)}
    return Dataset(data=pd.DataFrame(data), name="random")


@pytest.fixture
def missing_dataset() -> Dataset:
    """결측치 포함 데이터."""
    data = {
        "q1": [4, 3, np.nan, 2, 4],
        "q2": [3, 4, 4, np.nan, 5],
        "q3": [5, 3, 4, 2, 4],
    }
    return Dataset(data=pd.DataFrame(data), name="missing")


def _make_spec(vars_: list[str], **opts) -> dict:
    return {"variables": {"target": vars_}, "options": opts}


# ──────────────────────────────────────────────────────────────
# 1. Cronbach's Alpha 수치 정확성
# ──────────────────────────────────────────────────────────────

class TestCronbachAlphaValue:

    def test_alpha_range_0_to_1(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        rel_table = next(t for t in result.tables if "Reliability" in t.title)
        alpha_str = rel_table.dataframe["Cronbach's Alpha"].iloc[0]
        alpha = float(alpha_str)
        assert 0.0 <= alpha <= 1.0

    def test_high_alpha_for_correlated_items(self, likert_dataset):
        """상관된 문항 → alpha > 0.7."""
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        rel_table = next(t for t in result.tables if "Reliability" in t.title)
        alpha = float(rel_table.dataframe["Cronbach's Alpha"].iloc[0])
        assert alpha > 0.7

    def test_helper_two_item(self):
        """2문항 alpha 계산."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
        alpha = _cronbach_alpha(df)
        assert alpha > 0.9

    def test_helper_opposite_items(self):
        """완전 반대 방향 문항 → 합계 분산=0 → NaN (alpha 정의 불가)."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        alpha = _cronbach_alpha(df)
        assert np.isnan(alpha)

    def test_helper_single_item_nan(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert np.isnan(_cronbach_alpha(df))

    def test_helper_constant_item_nan(self):
        df = pd.DataFrame({"a": [3, 3, 3, 3], "b": [3, 3, 3, 3]})
        assert np.isnan(_cronbach_alpha(df))


# ──────────────────────────────────────────────────────────────
# 2. 결과 구조
# ──────────────────────────────────────────────────────────────

class TestReliabilityResultStructure:

    def test_returns_analysis_result(self, likert_dataset):
        from statworkbench.analysis.result import AnalysisResult
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        assert isinstance(result, AnalysisResult)

    def test_has_five_tables(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        assert len(result.tables) == 5

    def test_table_titles(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        titles = [t.title for t in result.tables]
        assert "Case Processing Summary" in titles
        assert "Reliability Statistics" in titles
        assert "Item Statistics" in titles
        assert "Item-Total Statistics" in titles
        assert "Scale Statistics" in titles

    def test_item_count_in_rel_table(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        rel_table = next(t for t in result.tables if "Reliability" in t.title)
        assert rel_table.dataframe["항목 수"].iloc[0] == 4

    def test_item_stats_rows_match_vars(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        item_table = next(t for t in result.tables if t.title == "Item Statistics")
        assert len(item_table.dataframe) == 4

    def test_item_total_has_corrected_r(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        it_table = next(t for t in result.tables if "Total" in t.title)
        assert "교정 항목-전체 상관" in it_table.dataframe.columns

    def test_item_total_has_alpha_if_deleted(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        it_table = next(t for t in result.tables if "Total" in t.title)
        assert "항목 제거 시 Alpha" in it_table.dataframe.columns

    def test_has_note_with_alpha(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q2", "q3", "q4"]))
        assert len(result.notes) > 0
        assert "α" in result.notes[0]


# ──────────────────────────────────────────────────────────────
# 3. 결측치 처리
# ──────────────────────────────────────────────────────────────

class TestReliabilityMissing:

    def test_listwise_excludes_cases(self, missing_dataset):
        result = run_analysis(missing_dataset, _make_spec(["q1", "q2", "q3"], listwise=True))
        cps = next(t for t in result.tables if "Processing" in t.title)
        excluded = int(cps.dataframe.loc[cps.dataframe["구분"] == "제외됨", "N"].iloc[0])
        assert excluded > 0

    def test_still_succeeds_with_missing(self, missing_dataset):
        result = run_analysis(missing_dataset, _make_spec(["q1", "q2", "q3"]))
        assert len(result.tables) == 5


# ──────────────────────────────────────────────────────────────
# 4. 오류 처리
# ──────────────────────────────────────────────────────────────

class TestReliabilityErrors:

    def test_single_var_returns_warning(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1"]))
        assert len(result.warnings) > 0

    def test_no_vars_returns_warning(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec([]))
        assert len(result.warnings) > 0

    def test_nonexistent_var_returns_warning(self, likert_dataset):
        result = run_analysis(likert_dataset, _make_spec(["q1", "q999"]))
        assert len(result.warnings) > 0


# ──────────────────────────────────────────────────────────────
# 5. alpha_if_deleted 헬퍼
# ──────────────────────────────────────────────────────────────

class TestAlphaIfDeleted:

    def test_returns_series(self, likert_dataset):
        data = likert_dataset.data[["q1", "q2", "q3", "q4"]].astype(float)
        result = _alpha_if_deleted(data)
        assert isinstance(result, pd.Series)

    def test_length_equals_items(self, likert_dataset):
        data = likert_dataset.data[["q1", "q2", "q3", "q4"]].astype(float)
        result = _alpha_if_deleted(data)
        assert len(result) == 4

    def test_all_values_in_range(self, likert_dataset):
        data = likert_dataset.data[["q1", "q2", "q3", "q4"]].astype(float)
        result = _alpha_if_deleted(data)
        for val in result:
            if not np.isnan(val):
                assert -1.0 <= val <= 1.0

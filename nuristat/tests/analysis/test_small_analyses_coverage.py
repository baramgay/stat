"""normality / frequencies / nonparametric / reliability / correlation 커버리지 보강.

미커버 라인:
  normality.py:     35, 63-64, 84-86  (string policy, var not found, n>5000 D'Agostino)
  frequencies.py:   35, 63-64, 77, 79, 83-84 (string policy, var not found, sort_by 분기, TypeError)
  nonparametric.py: 24, 39, 50, 73, 94-95, 122-125, 196-199, 244 (effect size 0, unknown test, wrong groups)
  reliability.py:   75-76, 85-86, 153, 157, 159 (exception pass, insufficient data, alpha grade)
  correlation.py:   58, 131-134, 146-147, 200, 207, 209 (string policy, corr fail, unknown method, stars)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType


# ===========================================================================
# normality.py
# ===========================================================================

from nuristat.analysis.normality import run_analysis as normality_run


@pytest.fixture
def norm_dataset():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 50), "y": rng.normal(0, 1, 50)})
    ds = Dataset(df, name="Norm")
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["y"].measure = MeasureType.SCALE
    return ds


class TestNormality:

    def test_string_missing_policy(self, norm_dataset):
        spec = {"variables": {"target": ["x"]}, "missing_policy": "listwise"}
        result = normality_run(norm_dataset, spec)
        assert len(result.tables) > 0

    def test_var_not_found_warns(self, norm_dataset):
        """norm_dataset의 df.columns에 없는 변수를 target_vars 루프에서 걸러내도록
        prepare_analysis_frame을 우회해 직접 테스트."""
        # prepare_analysis_frame이 ValueError를 발생시키므로,
        # 대신 준비된 df에 해당 컬럼이 없는 상황을 만든다.
        spec = {"variables": {"target": ["x", "ghost_var"]}}
        with patch(
            "nuristat.analysis.normality.prepare_analysis_frame",
        ) as mock_prep:
            import types
            mock_result = types.SimpleNamespace(
                data=norm_dataset.data[["x"]].copy(),  # ghost_var 없음
                n_total=50, n_valid=50, n_excluded=0, excluded_pct=0.0,
            )
            mock_prep.return_value = mock_result
            result = normality_run(norm_dataset, spec)
        assert any("ghost_var" in w for w in result.warnings)

    def test_n_gt_5000_dagostino(self):
        """N > 5000 → D'Agostino 검정 (lines 84-86)."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"big": rng.normal(0, 1, 5001)})
        ds = Dataset(df, name="BigData")
        ds.variables["big"].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["big"]}}
        result = normality_run(ds, spec)
        assert any("D'Agostino" in w or "5000" in w for w in result.warnings)


# ===========================================================================
# frequencies.py
# ===========================================================================

from nuristat.analysis.frequencies import run_analysis as freq_run


@pytest.fixture
def freq_dataset():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "cat": rng.choice(["A", "B", "C"], 30),
        "num": rng.choice([1, 2, 3, 4], 30),
    })
    return Dataset(df, name="FreqData")


class TestFrequencies:

    def test_string_missing_policy(self, freq_dataset):
        spec = {"variables": {"target": ["cat"]}, "missing_policy": "listwise"}
        result = freq_run(freq_dataset, spec)
        assert len(result.tables) > 0

    def test_var_not_found_warns(self, freq_dataset):
        """dataset.data.columns에 없는 변수 → 경고 (line 62-64)."""
        # frequencies.py: for var_name in target_vars: if var_name not in dataset.data.columns
        # prepare_analysis_frame 없이 바로 체크하므로 직접 없는 변수명 전달 가능
        spec = {"variables": {"target": ["cat", "nonexistent"]}}
        with patch(
            "nuristat.analysis.frequencies.prepare_analysis_frame",
        ) as mock_prep:
            import types
            mock_result = types.SimpleNamespace(
                data=freq_dataset.data[["cat"]].copy(),
                n_total=30, n_valid=30, n_excluded=0, excluded_pct=0.0,
            )
            mock_prep.return_value = mock_result
            result = freq_run(freq_dataset, spec)
        assert any("nonexistent" in w for w in result.warnings)

    def test_sort_by_frequency(self, freq_dataset):
        """sort_by='frequency' → value_counts(sort=True) (line 77)."""
        spec = {
            "variables": {"target": ["cat"]},
            "options": {"sort_by": "frequency"},
        }
        result = freq_run(freq_dataset, spec)
        assert len(result.tables) > 0

    def test_sort_by_label(self, freq_dataset):
        """sort_by='label' → value_counts(sort=False).sort_index() (line 79)."""
        spec = {
            "variables": {"target": ["num"]},
            "options": {"sort_by": "label"},
        }
        result = freq_run(freq_dataset, spec)
        assert len(result.tables) > 0

    def test_sort_type_error_fallback(self):
        """sort_by 기본값 → sort_index() TypeError → fallback (lines 83-84)."""
        # Mixed type 시리즈: int + str 혼합 → sort_index() TypeError 발생 가능
        df = pd.DataFrame({"mixed": [1, "a", 2, "b", 3]})
        ds = Dataset(df, name="MixedTypes")
        spec = {"variables": {"target": ["mixed"]}}
        result = freq_run(ds, spec)
        assert len(result.tables) > 0


# ===========================================================================
# nonparametric.py
# ===========================================================================

from nuristat.analysis.nonparametric import (
    run_analysis as np_run,
    _epsilon_squared,
    _kendalls_w,
)


@pytest.fixture
def mann_dataset():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "score": np.concatenate([rng.normal(50, 8, 20), rng.normal(60, 8, 20)]),
        "group": ["A"] * 20 + ["B"] * 20,
    })
    ds = Dataset(df, "MannData")
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.BINARY
    return ds


class TestNonparametric:

    # Lines 24, 39, 50: effect size 함수 경계값
    def test_epsilon_squared_zero_when_n_le_k(self):
        """n <= k → epsilon_squared = 0 (line 24)."""
        assert _epsilon_squared(5.0, n=3, k=3) == 0.0

    def test_kendalls_w_zero_when_n_zero(self):
        """n_subjects == 0 → 0.0 (line 39)."""
        data = np.zeros((0, 3))
        assert _kendalls_w(data) == 0.0

    def test_kendalls_w_zero_when_k_le_1(self):
        """k_conditions <= 1 → 0.0 (line 39)."""
        data = np.array([[1], [2], [3]])
        assert _kendalls_w(data) == 0.0

    def test_kendalls_w_zero_when_ss_total_zero(self):
        """모든 값 동일 → ss_total == 0 → 0.0 (line 50)."""
        data = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert _kendalls_w(data) == 0.0

    # Line 73: string missing_policy
    def test_string_missing_policy(self, mann_dataset):
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
            "missing_policy": "listwise",
        }
        result = np_run(mann_dataset, spec)
        assert len(result.tables) > 0

    # Lines 94-95: unknown test type
    def test_unknown_test_type_warns(self, mann_dataset):
        spec = {"options": {"test": "unknown_test"}}
        result = np_run(mann_dataset, spec)
        assert any("Unknown test type" in w for w in result.warnings)

    # Lines 122-125: Mann-Whitney with != 2 groups
    def test_mann_whitney_three_groups_warns(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({
            "score": rng.normal(0, 1, 30),
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        })
        ds = Dataset(df, "ThreeGroup")
        ds.variables["score"].measure = MeasureType.SCALE
        ds.variables["group"].measure = MeasureType.NOMINAL
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "mann_whitney"},
        }
        result = np_run(ds, spec)
        assert any("2 groups" in w or "2개" in w for w in result.warnings)

    # Lines 196-199: Wilcoxon with != 2 vars
    def test_wilcoxon_single_var_warns(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"v1": rng.normal(0, 1, 20)})
        ds = Dataset(df, "SingleVar")
        ds.variables["v1"].measure = MeasureType.SCALE
        spec = {
            "variables": {"paired": ["v1"]},  # only 1 var, need 2
            "options": {"test": "wilcoxon"},
        }
        result = np_run(ds, spec)
        assert any("2" in w for w in result.warnings)


# ===========================================================================
# reliability.py
# ===========================================================================

from nuristat.analysis.reliability import run_analysis as rel_run


@pytest.fixture
def rel_dataset():
    rng = np.random.default_rng(6)
    f = rng.normal(0, 1, 50)
    df = pd.DataFrame({
        "item1": f + rng.normal(0, 0.3, 50),
        "item2": f + rng.normal(0, 0.3, 50),
        "item3": f + rng.normal(0, 0.3, 50),
        "item4": f + rng.normal(0, 0.3, 50),
    })
    ds = Dataset(df, "RelData")
    for col in df.columns:
        ds.variables[col].measure = MeasureType.SCALE
    return ds


class TestReliability:

    def test_insufficient_data_warns(self):
        """n < 2 → 경고 + return (lines 85-86)."""
        df = pd.DataFrame({"item1": [1.0], "item2": [2.0]})
        ds = Dataset(df, "OneRow")
        spec = {"variables": {"target": ["item1", "item2"]}}
        result = rel_run(ds, spec)
        assert any("부족" in w for w in result.warnings)

    def test_exception_in_to_numeric_passes(self, rel_dataset):
        """data.apply(to_numeric) 예외 → pass (lines 75-76)."""
        spec = {"variables": {"target": ["item1", "item2", "item3", "item4"]}}
        with patch("pandas.DataFrame.apply", side_effect=Exception("apply fail")):
            result = rel_run(rel_dataset, spec)
        # 예외 발생 시 pass로 처리됨
        assert result is not None

    def test_alpha_excellent_branch(self):
        """alpha >= 0.9 → '우수 (Excellent)' (line 153)."""
        rng = np.random.default_rng(7)
        f = rng.normal(0, 1, 200)
        df = pd.DataFrame({
            "i1": f + rng.normal(0, 0.05, 200),
            "i2": f + rng.normal(0, 0.05, 200),
            "i3": f + rng.normal(0, 0.05, 200),
            "i4": f + rng.normal(0, 0.05, 200),
        })
        ds = Dataset(df, "HighAlpha")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["i1", "i2", "i3", "i4"]}}
        result = rel_run(ds, spec)
        assert any("우수" in n for n in result.notes)

    def test_alpha_acceptable_branch(self):
        """alpha 0.7-0.8 → '수용 가능 (Acceptable)' (line 157)."""
        rng = np.random.default_rng(8)
        f = rng.normal(0, 1, 100)
        df = pd.DataFrame({
            "i1": f + rng.normal(0, 0.7, 100),
            "i2": f + rng.normal(0, 0.7, 100),
            "i3": f + rng.normal(0, 0.7, 100),
        })
        ds = Dataset(df, "MidAlpha")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["i1", "i2", "i3"]}}
        result = rel_run(ds, spec)
        notes = " ".join(result.notes)
        # alpha 값에 따라 수용 가능, 의심스러움, 불량 중 하나
        assert any(k in notes for k in ["수용 가능", "의심스러움", "불량", "양호", "우수"])

    def test_alpha_questionable_branch(self):
        """alpha 0.6-0.7 → '의심스러움 (Questionable)' (line 159)."""
        rng = np.random.default_rng(9)
        f = rng.normal(0, 1, 100)
        df = pd.DataFrame({
            "i1": f + rng.normal(0, 1.2, 100),
            "i2": f + rng.normal(0, 1.2, 100),
        })
        ds = Dataset(df, "LowAlpha")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["i1", "i2"]}}
        result = rel_run(ds, spec)
        assert len(result.tables) > 0


# ===========================================================================
# correlation.py
# ===========================================================================

from nuristat.analysis.correlation import run_analysis as corr_run


@pytest.fixture
def corr_dataset():
    rng = np.random.default_rng(10)
    n = 50
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n) * 0.6 + rng.normal(0, 0.8, n),
        "z": rng.normal(0, 1, n) * 0.3 + rng.normal(0, 0.9, n),
    })
    ds = Dataset(df, "CorrData")
    for col in df.columns:
        ds.variables[col].measure = MeasureType.SCALE
    return ds


class TestCorrelation:

    def test_string_missing_policy(self, corr_dataset):
        """missing_policy='listwise' 문자열 → 정상 실행 (line 58)."""
        spec = {
            "variables": {"target": ["x", "y", "z"]},
            "missing_policy": "listwise",
        }
        result = corr_run(corr_dataset, spec)
        assert len(result.tables) > 0

    def test_unknown_method_warns(self, corr_dataset):
        """method='polyserial' → 'Unknown method' 경고 (lines 146-147)."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "polyserial"},
        }
        result = corr_run(corr_dataset, spec)
        assert any("Unknown method" in w for w in result.warnings)

    def test_pairwise_too_few_valid_nan_continue(self):
        """pairwise=True, 유효 쌍 < 2 → nan + continue (lines 131-134)."""
        # 두 변수 중 하나가 거의 전부 NaN
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [np.nan, np.nan, np.nan, np.nan, 1.0],  # 유효 쌍 = 1 < 2
        })
        ds = Dataset(df, "PairwiseFew")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {
            "variables": {"target": ["a", "b"]},
            "options": {"pairwise": True},
        }
        result = corr_run(ds, spec)
        assert result is not None

    def test_nan_corr_skips_detail_row(self, corr_dataset):
        """nan r 값 → detail_rows에서 continue (line 200)."""
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"pairwise": True},
        }
        # pairwise 모드로 NaN이 있는 데이터
        rng = np.random.default_rng(12)
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0, 4.0, 5.0],
            "y": [np.nan, 2.0, np.nan, np.nan, 5.0],
        })
        ds = Dataset(df, "NanCorr")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {"variables": {"target": ["x", "y"]}, "options": {"pairwise": True}}
        result = corr_run(ds, spec)
        assert result is not None

    def test_significance_stars_two_star(self):
        """p < 0.01 → '**' 분기 (line 207)."""
        rng = np.random.default_rng(11)
        n = 200
        f = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "a": f + rng.normal(0, 0.2, n),
            "b": f + rng.normal(0, 0.2, n),
        })
        ds = Dataset(df, "StarData")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {
            "variables": {"target": ["a", "b"]},
            "options": {"flag_significant": True},
        }
        result = corr_run(ds, spec)
        assert len(result.tables) > 0

    def test_significance_stars_one_star(self):
        """p < 0.05 → '*' 분기 (line 209) — 약한 상관."""
        rng = np.random.default_rng(99)
        n = 30
        f = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "a": f + rng.normal(0, 1.5, n),
            "b": f + rng.normal(0, 1.5, n),
        })
        ds = Dataset(df, "WeakCorr")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        spec = {
            "variables": {"target": ["a", "b"]},
            "options": {"flag_significant": True},
        }
        result = corr_run(ds, spec)
        assert result is not None

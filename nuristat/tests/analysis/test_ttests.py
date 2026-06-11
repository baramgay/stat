"""Tests for t-test analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy
from nuristat.analysis.ttests import run_analysis


@pytest.fixture
def independent_dataset():
    """Create dataset for independent t-test."""
    np.random.seed(42)
    df = pd.DataFrame({
        "score": np.concatenate([
            np.random.normal(75, 10, 20),
            np.random.normal(85, 10, 20),
        ]),
        "group": ["A"] * 20 + ["B"] * 20,
    })
    ds = Dataset(df, name="Independent")
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def paired_dataset_fixture():
    """Create dataset for paired t-test."""
    np.random.seed(42)
    df = pd.DataFrame({
        "pre": np.random.normal(70, 8, 15),
        "post": np.random.normal(78, 8, 15),
    })
    ds = Dataset(df, name="Paired")
    ds.variables["pre"].measure = MeasureType.SCALE
    ds.variables["post"].measure = MeasureType.SCALE
    return ds


class TestIndependentTTest:
    """Test independent samples t-test."""

    def test_basic_independent_ttest(self, independent_dataset):
        """Test basic independent t-test."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        assert result.id == "t_test"
        assert len(result.tables) >= 3  # CPS + group stats + Levene + t-test

    def test_levene_test(self, independent_dataset):
        """Test that Levene's test is included."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        levene_table = [t for t in result.tables if "Equality of Variances" in t.title][0]
        assert len(levene_table.dataframe) == 1

    def test_both_variants(self, independent_dataset):
        """Test that both equal and unequal variance results are shown."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        ttest_table = [t for t in result.tables if "Independent" in t.title][0]
        df = ttest_table.dataframe
        assert len(df) == 2  # equal var + unequal var

    def test_group_statistics(self, independent_dataset):
        """Test that group statistics are correct."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        group_table = [t for t in result.tables if "Group Statistics" in t.title][0]
        df = group_table.dataframe
        assert len(df) == 2
        assert df.iloc[0]["N"] == 20
        assert df.iloc[1]["N"] == 20

    def test_vs_scipy(self, independent_dataset):
        """Test that results match scipy."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "yes"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(independent_dataset, spec)

        # Manual scipy calculation
        g1 = independent_dataset.data[independent_dataset.data["group"] == "A"]["score"].values
        g2 = independent_dataset.data[independent_dataset.data["group"] == "B"]["score"].values
        t_scipy, p_scipy = stats.ttest_ind(g1, g2)

        ttest_table = [t for t in result.tables if "Independent" in t.title][0]
        df = ttest_table.dataframe
        eq_row = df[df["Variant"] == "Equal variances assumed"].iloc[0]
        assert abs(float(eq_row["t"]) - t_scipy) < 0.01


class TestPairedTTest:
    """Test paired samples t-test."""

    def test_basic_paired_ttest(self, paired_dataset_fixture):
        """Test basic paired t-test."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        assert result.id == "t_test"
        assert len(result.tables) >= 3  # CPS + paired stats + test

    def test_paired_statistics(self, paired_dataset_fixture):
        """Test paired statistics table."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        paired_table = [t for t in result.tables if "Paired Samples Statistics" in t.title][0]
        df = paired_table.dataframe
        assert len(df) == 2

    def test_vs_scipy(self, paired_dataset_fixture):
        """Test that results match scipy."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        pre = paired_dataset_fixture.data["pre"].values
        post = paired_dataset_fixture.data["post"].values
        t_scipy, p_scipy = stats.ttest_rel(pre, post)

        test_table = [t for t in result.tables if "Paired Samples t-Test" in t.title][0]
        df = test_table.dataframe
        t_row = df[df["Statistic"] == "t"].iloc[0]
        assert abs(float(t_row["Value"]) - t_scipy) < 0.01

    def test_cohens_dz(self, paired_dataset_fixture):
        """Test that Cohen's dz is computed."""
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(paired_dataset_fixture, spec)

        test_table = [t for t in result.tables if "Paired Samples t-Test" in t.title][0]
        df = test_table.dataframe
        dz_row = df[df["Statistic"] == "Cohen's dz"]
        assert len(dz_row) == 1
        assert abs(float(dz_row.iloc[0]["Value"])) > 0

    def test_with_missing(self):
        """Test paired t-test with missing data."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, np.nan, 5.0],
            "y": [2.0, 3.0, np.nan, 5.0, 6.0],
        })
        ds = Dataset(df, name="MissingPaired")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"paired": ["x", "y"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        test_table = [t for t in result.tables if "Paired Samples t-Test" in t.title][0]
        df = test_table.dataframe
        # N is in Paired Samples Statistics table, not t-Test table
        stats_table = [t for t in result.tables if "Paired Samples Statistics" in t.title][0]
        stats_df = stats_table.dataframe
        n_row = stats_df[stats_df["Variable"] == "x"].iloc[0]
        assert int(n_row["N"]) == 3  # rows 0, 1, 4


class TestIndependentTTestGroupValues:
    """group_values 옵션 및 3개 이상 그룹 처리 테스트."""

    @pytest.fixture
    def three_group_dataset(self):
        """3개 그룹을 가진 데이터셋."""
        np.random.seed(7)
        df = pd.DataFrame({
            "score": np.concatenate([
                np.random.normal(70, 8, 15),
                np.random.normal(80, 8, 15),
                np.random.normal(90, 8, 15),
            ]),
            "group": [1] * 15 + [2] * 15 + [3] * 15,
        })
        ds = Dataset(df, name="ThreeGroup")
        ds.variables["score"].measure = MeasureType.SCALE
        ds.variables["group"].measure = MeasureType.NOMINAL
        return ds

    def test_three_groups_without_group_values_warns(self, three_group_dataset):
        """group_values 미지정 시 3그룹 → 경고 + ANOVA 안내."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        }
        result = run_analysis(three_group_dataset, spec)
        assert any("3개" in w or "ANOVA" in w or "3 groups" in w.lower() for w in result.warnings)

    def test_three_groups_without_group_values_lists_groups(self, three_group_dataset):
        """경고 메시지에 사용 가능한 그룹 값이 포함되어야 한다."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
        }
        result = run_analysis(three_group_dataset, spec)
        warning_text = " ".join(result.warnings)
        assert "1" in warning_text and "2" in warning_text and "3" in warning_text

    def test_group_values_selects_two_groups(self, three_group_dataset):
        """group_values=[1,2] 지정 시 그룹 1과 2만 비교."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto", "group_values": [1, 2]},
            "confidence_level": 0.95,
        }
        result = run_analysis(three_group_dataset, spec)
        assert not result.warnings, f"경고가 없어야 함: {result.warnings}"
        ttest_table = [t for t in result.tables if "Independent" in t.title]
        assert ttest_table, "t-test 결과 테이블이 있어야 함"

    def test_group_values_group_stats_n(self, three_group_dataset):
        """group_values=[1,3] 지정 시 그룹 1·3 각 15건."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto", "group_values": [1, 3]},
            "confidence_level": 0.95,
        }
        result = run_analysis(three_group_dataset, spec)
        group_table = [t for t in result.tables if "Group Statistics" in t.title][0]
        ns = group_table.dataframe["N"].tolist()
        assert ns == [15, 15]

    def test_group_values_respects_order(self, three_group_dataset):
        """group_values=[3,1] → Mean Difference의 부호가 [1,3]과 반대."""
        spec_12 = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "yes", "group_values": [1, 3]},
            "confidence_level": 0.95,
        }
        spec_21 = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "yes", "group_values": [3, 1]},
            "confidence_level": 0.95,
        }
        r12 = run_analysis(three_group_dataset, spec_12)
        r21 = run_analysis(three_group_dataset, spec_21)
        t12 = [t for t in r12.tables if "Independent" in t.title][0].dataframe
        t21 = [t for t in r21.tables if "Independent" in t.title][0].dataframe
        md12 = float(t12.iloc[0]["Mean Difference"])
        md21 = float(t21.iloc[0]["Mean Difference"])
        assert abs(md12 + md21) < 0.01, "그룹 순서 반전 시 Mean Difference 부호도 반전"

    def test_invalid_group_values_warns(self, three_group_dataset):
        """존재하지 않는 그룹 값 지정 → 경고."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto", "group_values": [1, 99]},
            "confidence_level": 0.95,
        }
        result = run_analysis(three_group_dataset, spec)
        assert any("없습니다" in w or "not in" in w.lower() for w in result.warnings)

    def test_two_group_with_explicit_group_values(self, independent_dataset):
        """2그룹 변수에도 group_values 명시 지정 가능."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto", "group_values": ["A", "B"]},
            "confidence_level": 0.95,
        }
        result = run_analysis(independent_dataset, spec)
        assert not result.warnings
        ttest_table = [t for t in result.tables if "Independent" in t.title]
        assert ttest_table

    def test_same_group_values_warns(self, three_group_dataset):
        """그룹 1·2가 동일한 값이면 경고 반환."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto", "group_values": [1, 1]},
            "confidence_level": 0.95,
        }
        result = run_analysis(three_group_dataset, spec)
        assert any("동일" in w or "same" in w.lower() for w in result.warnings)

    def test_same_group_values_no_tables(self, three_group_dataset):
        """그룹 1·2가 동일하면 분석 결과 테이블 없음."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "auto", "group_values": [2, 2]},
            "confidence_level": 0.95,
        }
        result = run_analysis(three_group_dataset, spec)
        ttest_tables = [t for t in result.tables if "Independent" in t.title]
        assert not ttest_tables

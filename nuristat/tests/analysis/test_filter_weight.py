"""Tests for filter_$ and weight_var support in prepare_analysis_frame / frequencies."""

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.assumptions import prepare_analysis_frame
from nuristat.analysis.frequencies import run_analysis as freq_run
from nuristat.core.dataset import Dataset


def _make_dataset(data: dict) -> Dataset:
    return Dataset(pd.DataFrame(data))


def _make_filtered_dataset(data_cols: dict, filter_values: list) -> Dataset:
    """Create dataset then write filter_$ after init — mirrors select_cases_dialog."""
    ds = Dataset(pd.DataFrame(data_cols))
    ds.data["filter_$"] = filter_values
    return ds


# ---------------------------------------------------------------------------
# prepare_analysis_frame — filter
# ---------------------------------------------------------------------------

class TestFilter:
    def test_no_filter_column_passthrough(self):
        ds = _make_dataset({"x": [1, 2, 3, 4, 5]})
        pf = prepare_analysis_frame(ds, ["x"])
        assert pf.n_total == 5
        assert pf.n_valid == 5
        assert pf.n_filtered == 5

    def test_filter_dollar_subsets_rows(self):
        ds = _make_filtered_dataset({"x": [10, 20, 30, 40]}, [1, 0, 1, 0])
        pf = prepare_analysis_frame(ds, ["x"])
        assert pf.n_total == 4
        assert pf.n_filtered == 2
        assert pf.n_valid == 2
        assert list(pf.data["x"]) == [10, 30]

    def test_filter_with_missing_values(self):
        ds = _make_filtered_dataset({"x": [10, np.nan, 30, 40]}, [1, 1, 1, 0])
        pf = prepare_analysis_frame(ds, ["x"])
        # After filter: rows 0,1,2 → after listwise: 0,2
        assert pf.n_filtered == 3
        assert pf.n_valid == 2

    def test_all_filtered_out(self):
        ds = _make_filtered_dataset({"x": [1, 2, 3]}, [0, 0, 0])
        pf = prepare_analysis_frame(ds, ["x"])
        assert pf.n_filtered == 0
        assert pf.n_valid == 0
        assert len(pf.data) == 0


# ---------------------------------------------------------------------------
# prepare_analysis_frame — weight
# ---------------------------------------------------------------------------

class TestWeight:
    def test_weight_var_attached(self):
        ds = _make_dataset({"x": [1, 2, 3], "wt": [2.0, 1.0, 3.0]})
        pf = prepare_analysis_frame(ds, ["x"], weight_var="wt")
        assert pf.weight_var == "wt"
        assert "wt" in pf.data.columns

    def test_no_weight(self):
        ds = _make_dataset({"x": [1, 2, 3]})
        pf = prepare_analysis_frame(ds, ["x"])
        assert pf.weight_var is None

    def test_invalid_weight_var_ignored(self):
        ds = _make_dataset({"x": [1, 2, 3]})
        pf = prepare_analysis_frame(ds, ["x"], weight_var="nonexistent")
        assert pf.weight_var is None


# ---------------------------------------------------------------------------
# frequencies.run_analysis — weighted counts
# ---------------------------------------------------------------------------

class TestWeightedFrequencies:
    def test_weighted_frequency_matches_manual(self):
        """가중 빈도 = Σ(weight_var where value==v)."""
        ds = _make_dataset({
            "group": ["A", "A", "B", "B", "B"],
            "wt":    [3.0, 2.0, 1.0, 4.0, 2.0],
        })
        spec = {
            "variables": {"target": ["group"]},
            "options": {},
            "weight_var": "wt",
        }
        result = freq_run(ds, spec)
        assert result.tables  # should have at least CPS + freq table
        freq_table = result.tables[-1]  # last table is the freq table
        df = freq_table.dataframe
        row_a = df[df["Value"] == "A"].iloc[0]
        row_b = df[df["Value"] == "B"].iloc[0]
        assert abs(row_a["Frequency"] - 5.0) < 0.01  # 3+2
        assert abs(row_b["Frequency"] - 7.0) < 0.01  # 1+4+2

    def test_unweighted_frequency_unchanged(self):
        ds = _make_dataset({"group": ["A", "A", "B"]})
        spec = {"variables": {"target": ["group"]}, "options": {}}
        result = freq_run(ds, spec)
        freq_table = result.tables[-1]
        df = freq_table.dataframe
        assert int(df[df["Value"] == "A"].iloc[0]["Frequency"]) == 2
        assert int(df[df["Value"] == "B"].iloc[0]["Frequency"]) == 1

    def test_filtered_frequency(self):
        """filter_$ = 1 행만 집계."""
        ds = _make_filtered_dataset(
            {"group": ["A", "A", "B", "B"]},
            [1, 1, 0, 1],
        )
        spec = {"variables": {"target": ["group"]}, "options": {}}
        result = freq_run(ds, spec)
        # Only A,A,B (row 3) should be included (filter_$==1: rows 0,1,3)
        freq_table = result.tables[-1]
        df = freq_table.dataframe
        assert int(df[df["Value"] == "A"].iloc[0]["Frequency"]) == 2
        assert int(df[df["Value"] == "B"].iloc[0]["Frequency"]) == 1
        # Warning about filter should be emitted
        assert any("필터" in w for w in result.warnings)

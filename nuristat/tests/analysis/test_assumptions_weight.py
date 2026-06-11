"""Tests for prepare_analysis_frame reading Dataset.active_weight_var as fallback."""

import pandas as pd
import pytest

from nuristat.analysis.assumptions import prepare_analysis_frame
from nuristat.core.dataset import Dataset


@pytest.fixture
def ds():
    return Dataset(pd.DataFrame({
        "score": [1.0, 2.0, 3.0, 4.0],
        "group": ["A", "B", "A", "B"],
        "wt": [1, 2, 1, 3],
    }))


class TestPrepareAnalysisFrameWeight:
    def test_explicit_weight_var_used(self, ds):
        result = prepare_analysis_frame(ds, ["score"], weight_var="wt")
        assert result.weight_var == "wt"

    def test_dataset_active_weight_var_as_fallback(self, ds):
        ds.active_weight_var = "wt"
        result = prepare_analysis_frame(ds, ["score"])
        assert result.weight_var == "wt"

    def test_no_weight_when_not_set(self, ds):
        result = prepare_analysis_frame(ds, ["score"])
        assert result.weight_var is None

    def test_explicit_arg_overrides_dataset_active(self, ds):
        ds.active_weight_var = "wt"
        result = prepare_analysis_frame(ds, ["score", "wt"], weight_var="wt")
        assert result.weight_var == "wt"

    def test_weight_column_present_in_output(self, ds):
        ds.active_weight_var = "wt"
        result = prepare_analysis_frame(ds, ["score"])
        # weight column must be in the returned DataFrame so analysis can use it
        assert "wt" in result.data.columns

    def test_weight_var_none_when_column_missing(self):
        """If active_weight_var names a column not in the data, resolved to None."""
        ds = Dataset(pd.DataFrame({"score": [1.0, 2.0]}))
        ds.active_weight_var = "nonexistent_weight"
        result = prepare_analysis_frame(ds, ["score"])
        assert result.weight_var is None

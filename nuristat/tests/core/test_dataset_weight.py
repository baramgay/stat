"""Tests for Dataset.active_weight_var field."""

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset


@pytest.fixture
def ds():
    return Dataset(pd.DataFrame({"score": [1.0, 2.0, 3.0], "wt": [1, 2, 1]}))


class TestActiveWeightVar:
    def test_default_is_none(self, ds):
        assert ds.active_weight_var is None

    def test_can_set_weight_var(self, ds):
        ds.active_weight_var = "wt"
        assert ds.active_weight_var == "wt"

    def test_can_clear_weight_var(self, ds):
        ds.active_weight_var = "wt"
        ds.active_weight_var = None
        assert ds.active_weight_var is None

    def test_weight_var_not_lost_after_data_replace(self, ds):
        """Replacing .data should not reset active_weight_var."""
        ds.active_weight_var = "wt"
        new_df = pd.DataFrame({"score": [10.0], "wt": [2]})
        ds.data = new_df
        assert ds.active_weight_var == "wt"

    def test_weight_var_survives_copy(self, ds):
        """Dataset.copy() should NOT carry over weight state (clean slate)."""
        ds.active_weight_var = "wt"
        copied = ds.copy()
        # copy() is a fresh dataset — weight state is not part of serialized form
        assert copied.active_weight_var is None

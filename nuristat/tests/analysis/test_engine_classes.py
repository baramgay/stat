"""Engine wrapper class 커버리지 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset


def _make_dataset() -> Dataset:
    df = pd.DataFrame({
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        "group": ["A", "B", "A", "B", "A"],
        "label": ["x", "y", "x", "y", "x"],
    })
    return Dataset(data=df)


class TestTtestEngine:

    def test_validate_returns_empty_list(self):
        from nuristat.analysis.ttests import TtestEngine
        engine = TtestEngine()
        result = engine.validate(_make_dataset(), {})
        assert result == []

    def test_run_delegates_to_run_analysis(self):
        from nuristat.analysis.ttests import TtestEngine
        engine = TtestEngine()
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"test": "independent"},
        }
        result = engine.run(_make_dataset(), spec)
        assert result is not None


class TestDescriptiveEngine:

    def test_validate_returns_empty_list(self):
        from nuristat.analysis.descriptive import DescriptiveEngine
        engine = DescriptiveEngine()
        assert engine.validate(_make_dataset(), {}) == []

    def test_run_returns_result(self):
        from nuristat.analysis.descriptive import DescriptiveEngine
        engine = DescriptiveEngine()
        spec = {"variables": {"scale": ["score"]}}
        result = engine.run(_make_dataset(), spec)
        assert result is not None


class TestAnovaEngine:

    def test_validate_returns_empty_list(self):
        from nuristat.analysis.anova import AnovaEngine
        engine = AnovaEngine()
        assert engine.validate(_make_dataset(), {}) == []

    def test_run_returns_result(self):
        from nuristat.analysis.anova import AnovaEngine
        engine = AnovaEngine()
        spec = {
            "variables": {"dependent": "score", "factor": "group"},
            "options": {},
        }
        result = engine.run(_make_dataset(), spec)
        assert result is not None


class TestCorrelationEngine:

    def test_validate_returns_empty_list(self):
        from nuristat.analysis.correlation import CorrelationEngine
        engine = CorrelationEngine()
        assert engine.validate(_make_dataset(), {}) == []

    def test_run_returns_result(self):
        from nuristat.analysis.correlation import CorrelationEngine

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
        ds = Dataset(data=df)
        engine = CorrelationEngine()
        spec = {"variables": {"target": ["x", "y"]}, "options": {"method": "pearson"}}
        result = engine.run(ds, spec)
        assert result is not None


class TestRegressionEngine:

    def test_validate_returns_empty_list(self):
        from nuristat.analysis.regression import RegressionEngine
        engine = RegressionEngine()
        assert engine.validate(_make_dataset(), {}) == []

    def test_run_returns_result(self):
        from nuristat.analysis.regression import RegressionEngine

        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [2.0, 4.0, 5.0, 4.0, 5.0],
        })
        ds = Dataset(data=df)
        engine = RegressionEngine()
        spec = {
            "variables": {"dependent": "y", "predictors": ["x"]},
            "options": {},
        }
        result = engine.run(ds, spec)
        assert result is not None


class TestFrequenciesEngine:

    def test_validate_returns_empty_list(self):
        from nuristat.analysis.frequencies import FrequenciesEngine
        engine = FrequenciesEngine()
        assert engine.validate(_make_dataset(), {}) == []

    def test_run_returns_result(self):
        from nuristat.analysis.frequencies import FrequenciesEngine
        engine = FrequenciesEngine()
        spec = {"variables": {"target": ["group"]}}
        result = engine.run(_make_dataset(), spec)
        assert result is not None

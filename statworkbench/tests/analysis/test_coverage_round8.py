"""커버리지 8라운드 — ImportError 분기 테스트.

대상:
  ml_engine.py 37-38   : kmeans_clustering — sklearn 없음 → ImportError
  ml_engine.py 98-99   : decision_tree_classifier — sklearn 없음 → ImportError
  ml_engine.py 166-167 : linear_regression_ml — sklearn 없음 → ImportError
  survival_analysis.py 35-36 : lifelines 없음 → _LIFELINES_AVAILABLE = False (pass)
  factor_analysis.py 22-23   : sklearn 없음 → _SKLEARN_AVAILABLE = False
  logistic_regression.py 24-25: sklearn 없음 → _SKLEARN_AVAILABLE = False
  cluster_analysis.py 25-26  : sklearn 없음 → _SKLEARN_AVAILABLE = False
  discriminant_analysis.py 24-25: sklearn 없음 → _SKLEARN_AVAILABLE = False
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# ml_engine.py 37-38: kmeans_clustering → sklearn ImportError
# ---------------------------------------------------------------------------

class TestMLEngineKmeansImportError:

    def test_kmeans_raises_when_sklearn_missing(self):
        """sklearn.cluster 없음 → ImportError 재발생 (lines 37-38)."""
        from statworkbench.analysis.ml_engine import kmeans_clustering

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

        with patch.dict(sys.modules, {
            "sklearn": None,
            "sklearn.cluster": None,
            "sklearn.preprocessing": None,
        }):
            with pytest.raises(ImportError, match="scikit-learn"):
                kmeans_clustering(df, ["x", "y"], n_clusters=2)


# ---------------------------------------------------------------------------
# ml_engine.py 98-99: decision_tree_classifier → sklearn ImportError
# ---------------------------------------------------------------------------

class TestMLEngineDecisionTreeImportError:

    def test_decision_tree_raises_when_sklearn_missing(self):
        """sklearn.tree 없음 → ImportError 재발생 (lines 98-99)."""
        from statworkbench.analysis.ml_engine import decision_tree_classifier

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0, 0, 1, 1],
        })

        with patch.dict(sys.modules, {
            "sklearn": None,
            "sklearn.tree": None,
            "sklearn.model_selection": None,
            "sklearn.metrics": None,
            "sklearn.preprocessing": None,
        }):
            with pytest.raises(ImportError, match="scikit-learn"):
                decision_tree_classifier(df, ["x"], "y")


# ---------------------------------------------------------------------------
# ml_engine.py 166-167: linear_regression_ml → sklearn ImportError
# ---------------------------------------------------------------------------

class TestMLEngineLinearRegressionImportError:

    def test_linear_regression_raises_when_sklearn_missing(self):
        """sklearn.linear_model 없음 → ImportError 재발생 (lines 166-167)."""
        from statworkbench.analysis.ml_engine import linear_regression_ml

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
        })

        with patch.dict(sys.modules, {
            "sklearn": None,
            "sklearn.linear_model": None,
            "sklearn.model_selection": None,
            "sklearn.metrics": None,
        }):
            with pytest.raises(ImportError, match="scikit-learn"):
                linear_regression_ml(df, ["x"], "y")


# ---------------------------------------------------------------------------
# survival_analysis.py 35-36: lifelines 없음 → _LIFELINES_AVAILABLE = False
# ---------------------------------------------------------------------------

class TestSurvivalLifelinesImportError:

    def test_lifelines_unavailable_sets_flag_false(self):
        """lifelines 없음 → except ImportError: pass → _LIFELINES_AVAILABLE=False (lines 35-36)."""
        import statworkbench.analysis.survival_analysis as mod

        with patch.dict(sys.modules, {
            "lifelines": None,
            "lifelines.statistics": None,
        }):
            importlib.reload(mod)
            assert mod._LIFELINES_AVAILABLE is False

        # 복원
        importlib.reload(mod)
        assert mod._LIFELINES_AVAILABLE is True


# ---------------------------------------------------------------------------
# factor_analysis.py 22-23: sklearn 없음 → _SKLEARN_AVAILABLE = False
# ---------------------------------------------------------------------------

class TestFactorAnalysisSklearnImportError:

    def test_sklearn_unavailable_sets_flag_false(self):
        """sklearn 없음 → except ImportError: _SKLEARN_AVAILABLE=False (lines 22-23)."""
        import statworkbench.analysis.factor_analysis as mod

        with patch.dict(sys.modules, {
            "sklearn": None,
            "sklearn.decomposition": None,
        }):
            importlib.reload(mod)
            assert mod._SKLEARN_AVAILABLE is False

        # 복원
        importlib.reload(mod)
        assert mod._SKLEARN_AVAILABLE is True


# ---------------------------------------------------------------------------
# logistic_regression.py 24-25: sklearn 없음 → _SKLEARN_AVAILABLE = False
# ---------------------------------------------------------------------------

class TestLogisticRegressionSklearnImportError:

    def test_sklearn_unavailable_sets_flag_false(self):
        """sklearn 없음 → except ImportError: _SKLEARN_AVAILABLE=False (lines 24-25)."""
        import statworkbench.analysis.logistic_regression as mod

        with patch.dict(sys.modules, {
            "sklearn": None,
            "sklearn.metrics": None,
        }):
            importlib.reload(mod)
            assert mod._SKLEARN_AVAILABLE is False

        # 복원
        importlib.reload(mod)
        assert mod._SKLEARN_AVAILABLE is True


# ---------------------------------------------------------------------------
# cluster_analysis.py 25-26: sklearn 없음 → _SKLEARN_AVAILABLE = False
# ---------------------------------------------------------------------------

class TestClusterAnalysisSklearnImportError:

    def test_sklearn_unavailable_sets_flag_false(self):
        """sklearn 없음 → except ImportError: _SKLEARN_AVAILABLE=False (lines 25-26)."""
        import statworkbench.analysis.cluster_analysis as mod

        with patch.dict(sys.modules, {
            "sklearn": None,
            "sklearn.metrics": None,
            "sklearn.preprocessing": None,
        }):
            importlib.reload(mod)
            assert mod._SKLEARN_AVAILABLE is False

        # 복원
        importlib.reload(mod)
        assert mod._SKLEARN_AVAILABLE is True


# ---------------------------------------------------------------------------
# discriminant_analysis.py 24-25: sklearn 없음 → _SKLEARN_AVAILABLE = False
# ---------------------------------------------------------------------------

class TestDiscriminantAnalysisSklearnImportError:

    def test_sklearn_unavailable_sets_flag_false(self):
        """sklearn 없음 → except ImportError: _SKLEARN_AVAILABLE=False (lines 24-25)."""
        import statworkbench.analysis.discriminant_analysis as mod

        with patch.dict(sys.modules, {
            "sklearn": None,
            "sklearn.metrics": None,
            "sklearn.preprocessing": None,
        }):
            importlib.reload(mod)
            assert mod._SKLEARN_AVAILABLE is False

        # 복원
        importlib.reload(mod)
        assert mod._SKLEARN_AVAILABLE is True

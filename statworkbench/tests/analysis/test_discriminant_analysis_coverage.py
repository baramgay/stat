"""discriminant_analysis.py 미커버 라인 보강 테스트.

private 함수 직접 호출로 numpy 전역 패치 범위를 최소화.

대상 라인:
  57      : string missing_policy → MissingPolicy 변환
  79-80   : dependent/predictors 누락 → 경고 + 조기 반환
  91-92   : 유효 관측치 < 2 → 경고 + 조기 반환
  97-98   : n_classes < 2 → 경고 + 조기 반환
  122     : equal prior → np.ones(k)/k
  213     : wilks = np.nan (det_T <= 0)
  215     : wilks = np.nan (LinAlgError)
  223-225 : chi2=NaN, df_chi, p_val=NaN (wilks NaN 또는 <=0)
  245-246 : outer except (Wilks) → warnings 추가
  322-323 : except LinAlgError → pinv 사용
  345-346 : except Exception → warnings 추가
  410-411 : except Exception in _add_structure_matrix → warnings 추가
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.result import AnalysisResult
from statworkbench.analysis.discriminant_analysis import (
    _add_wilks_lambda,
    _add_classification_functions,
    _add_structure_matrix,
    run_analysis,
)


def _make_arrays(n: int = 40, seed: int = 42):
    """2집단 판별분석용 X, y 배열 생성."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack([
        rng.normal([0, 0], 1, (half, 2)),
        rng.normal([5, 5], 1, (half, 2)),
    ])
    y = np.array([0] * half + [1] * half)
    return X, y


@pytest.fixture
def result():
    return AnalysisResult(id="test", title="Test")


# ---------------------------------------------------------------------------
# Lines 213, 223-225: det_T <= 0 → wilks=NaN → chi2/p_val=NaN
# np.linalg.det를 0.0 반환으로 패치하되 _add_wilks_lambda 호출 범위만
# ---------------------------------------------------------------------------

class TestWilksLambdaDetTZero:

    def test_wilks_nan_when_det_t_zero(self, result):
        """det → 0.0 → det_T<=0 → wilks=NaN → chi2=NaN (213, 223-225)."""
        X, y = _make_arrays()
        predictors = ["x1", "x2"]
        class_labels = np.array(["A", "B"])
        k, p, n = 2, 2, len(X)

        with patch("numpy.linalg.det", return_value=0.0):
            _add_wilks_lambda(result, X, y, k, p, n, predictors, class_labels)

        wilks_tbl = next(t for t in result.tables if "Wilks" in t.title)
        # NaN 값은 포맷 후 "" 또는 "nan" 으로 표시됨
        assert wilks_tbl is not None


# ---------------------------------------------------------------------------
# Lines 215, 223-225: LinAlgError → wilks=NaN 경로
# ---------------------------------------------------------------------------

class TestWilksLambdaLinAlgError:

    def test_wilks_nan_when_linalg_error(self, result):
        """det → LinAlgError → wilks=NaN (215, 223-225)."""
        X, y = _make_arrays()
        predictors = ["x1", "x2"]
        class_labels = np.array(["A", "B"])
        k, p, n = 2, 2, len(X)

        with patch(
            "numpy.linalg.det",
            side_effect=np.linalg.LinAlgError("singular"),
        ):
            _add_wilks_lambda(result, X, y, k, p, n, predictors, class_labels)

        wilks_tbl = next(t for t in result.tables if "Wilks" in t.title)
        assert wilks_tbl is not None


# ---------------------------------------------------------------------------
# Lines 322-323: LinAlgError in inv → pinv fallback
# ---------------------------------------------------------------------------

class TestClassificationFunctionInvFallback:

    def test_singular_sw_uses_pinv(self, result):
        """inv → LinAlgError → pinv 사용 (322-323)."""
        X, y = _make_arrays()
        predictors = ["x1", "x2"]
        class_labels = np.array(["A", "B"])
        k = 2

        with patch(
            "numpy.linalg.inv",
            side_effect=np.linalg.LinAlgError("singular"),
        ):
            _add_classification_functions(result, X, y, predictors, class_labels, k)

        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# Lines 345-346: except Exception → warnings 추가
# ---------------------------------------------------------------------------

class TestClassificationFunctionException:

    def test_classification_coef_exception_adds_warning(self, result):
        """inv + pinv 모두 예외 → lines 345-346 실행 → warnings 추가."""
        X, y = _make_arrays()
        predictors = ["x1", "x2"]
        class_labels = np.array(["A", "B"])
        k = 2

        def raise_error(*args, **kwargs):
            raise RuntimeError("matrix error")

        with patch("numpy.linalg.inv", side_effect=raise_error), \
             patch("numpy.linalg.pinv", side_effect=raise_error):
            _add_classification_functions(result, X, y, predictors, class_labels, k)

        assert any("분류 함수 계수" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 245-246: outer except in _add_wilks_lambda → warnings 추가
# det가 LinAlgError 가 아닌 TypeError 를 raise → 내부 except 통과 → 외부 except 유도
# ---------------------------------------------------------------------------

class TestWilksLambdaOuterException:

    def test_outer_wilks_exception_adds_warning(self, result):
        """det → TypeError → outer except (245-246) → warnings 추가."""
        X, y = _make_arrays()
        predictors = ["x1", "x2"]
        class_labels = np.array(["A", "B"])
        k, p, n = 2, 2, len(X)

        with patch("numpy.linalg.det", side_effect=TypeError("type fail")):
            _add_wilks_lambda(result, X, y, k, p, n, predictors, class_labels)

        assert any("Wilks" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lines 410-411: except Exception in _add_structure_matrix
# ---------------------------------------------------------------------------

class TestStructureMatrixException:

    def test_structure_matrix_exception_adds_warning(self, result):
        """lda.transform → RuntimeError → lines 410-411 실행."""
        X, y = _make_arrays()
        predictors = ["x1", "x2"]
        n_components = 1

        lda_mock = MagicMock()
        lda_mock.transform.side_effect = RuntimeError("transform fail")

        _add_structure_matrix(result, X, y, lda_mock, predictors, n_components)

        assert any("구조 행렬" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Line 57: string missing_policy → MissingPolicy 변환
# Lines 79-80: dependent/predictors 누락 → 경고 + 조기 반환
# Lines 91-92: 유효 관측치 < 2 → 경고 + 조기 반환
# Lines 97-98: n_classes < 2 → 경고 + 조기 반환
# Line 122: equal prior
# ---------------------------------------------------------------------------

def _make_da_dataset(n: int = 30, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    half = n // 2
    df = pd.DataFrame({
        "grp": ["A"] * half + ["B"] * half,
        "x1": np.concatenate([rng.normal(0, 1, half), rng.normal(5, 1, half)]),
        "x2": np.concatenate([rng.normal(0, 1, half), rng.normal(5, 1, half)]),
    })
    ds = Dataset(df, "DA")
    ds.variables["grp"].measure = MeasureType.NOMINAL
    ds.variables["x1"].measure = MeasureType.SCALE
    ds.variables["x2"].measure = MeasureType.SCALE
    return ds


class TestDiscriminantRunAnalysisBranches:

    def test_string_missing_policy(self):
        """string missing_policy → MissingPolicy 변환 (57)."""
        ds = _make_da_dataset()
        spec = {
            "variables": {"dependent": "grp", "predictors": ["x1", "x2"]},
            "missing_policy": "listwise",
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0

    def test_no_predictors_warns(self):
        """predictors 누락 → lines 79-80 실행."""
        ds = _make_da_dataset()
        spec = {"variables": {"dependent": "grp", "predictors": []}}
        result = run_analysis(ds, spec)
        assert any("예측변수" in w for w in result.warnings)

    def test_too_few_obs_warns(self):
        """유효 관측치 1개 → lines 91-92 실행."""
        df = pd.DataFrame({"grp": ["A"], "x1": [1.0], "x2": [2.0]})
        ds = Dataset(df, "TinyDA")
        ds.variables["grp"].measure = MeasureType.NOMINAL
        ds.variables["x1"].measure = MeasureType.SCALE
        ds.variables["x2"].measure = MeasureType.SCALE
        spec = {"variables": {"dependent": "grp", "predictors": ["x1", "x2"]}}
        result = run_analysis(ds, spec)
        assert any("2개 미만" in w for w in result.warnings)

    def test_single_class_warns(self):
        """집단 변수에 범주 1개 → lines 97-98 실행."""
        df = pd.DataFrame({
            "grp": ["A", "A", "A"],
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
        })
        ds = Dataset(df, "OneClass")
        ds.variables["grp"].measure = MeasureType.NOMINAL
        ds.variables["x1"].measure = MeasureType.SCALE
        ds.variables["x2"].measure = MeasureType.SCALE
        spec = {"variables": {"dependent": "grp", "predictors": ["x1", "x2"]}}
        result = run_analysis(ds, spec)
        assert any("2개 이상" in w for w in result.warnings)

    def test_equal_prior(self):
        """prior='equal' → np.ones(k)/k 사전 확률 (122)."""
        ds = _make_da_dataset()
        spec = {
            "variables": {"dependent": "grp", "predictors": ["x1", "x2"]},
            "options": {"prior": "equal"},
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) > 0

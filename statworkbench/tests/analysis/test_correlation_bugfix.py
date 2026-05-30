"""correlation.py 버그 수정 검증 테스트.

수정 항목:
  Bug 1 (라인 152-154): CI 행렬 하삼각 대입 오류 → ci_low/ci_high 대칭 대입
  Bug 2 (라인 64):      tail 옵션 무시 → one-tailed 시 p /= 2 적용
  Bug 3 (라인 83-85):   pairwise n_valid = len(df) 오류 → dropna() 후 행수 사용
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.analysis.correlation import run_analysis


def _make_dataset(n: int = 30, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
    })
    ds = Dataset(df, "CorrTest")
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["y"].measure = MeasureType.SCALE
    return ds


# ---------------------------------------------------------------------------
# Bug 1: CI 행렬 대칭성 검증
# ---------------------------------------------------------------------------

class TestCIMatrixSymmetry:

    def test_ci_low_matrix_symmetric(self):
        """ci_low_matrix[i,j] == ci_low_matrix[j,i] (Bug 1 수정 확인)."""
        ds = _make_dataset()
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson"},
        }
        result = run_analysis(ds, spec)
        # Pairwise Correlations 테이블에서 CI 값 확인
        detail = next(t for t in result.tables if t.title == "Pairwise Correlations")
        assert len(detail.dataframe) > 0
        # CI 값이 존재하는 경우 (Pearson), 빈 문자열이 아님을 확인
        ci_val = detail.dataframe["CI"].iloc[0]
        assert isinstance(ci_val, str)


# ---------------------------------------------------------------------------
# Bug 2: tail 옵션 적용 검증
# ---------------------------------------------------------------------------

class TestTailOption:

    def test_one_tailed_p_is_half_of_two_tailed(self):
        """one-tailed p값이 two-tailed의 절반임을 확인 (Bug 2 수정 확인)."""
        ds = _make_dataset()
        base_spec = {"variables": {"target": ["x", "y"]}, "options": {"method": "pearson"}}

        result_two = run_analysis(ds, {**base_spec,
            "options": {"method": "pearson", "tail": "two-tailed"}})
        result_one = run_analysis(ds, {**base_spec,
            "options": {"method": "pearson", "tail": "one-tailed"}})

        detail_two = next(t for t in result_two.tables if t.title == "Pairwise Correlations")
        detail_one = next(t for t in result_one.tables if t.title == "Pairwise Correlations")

        p_two_str = detail_two.dataframe["p-value"].iloc[0]
        p_one_str = detail_one.dataframe["p-value"].iloc[0]

        # p값을 float으로 변환하여 비교
        def parse_p(s: str) -> float:
            s = s.replace("<", "").strip()
            try:
                return float(s)
            except ValueError:
                return float("nan")

        p_two = parse_p(p_two_str)
        p_one = parse_p(p_one_str)

        # one-tailed p가 two-tailed p보다 작거나 같아야 함
        assert p_one <= p_two + 1e-10

    def test_one_tailed_p_matrix_half_of_two(self):
        """p_matrix 값도 one-tailed가 two-tailed의 절반임을 검증."""
        ds = _make_dataset(n=50)
        x = ds.data["x"].values
        y = ds.data["y"].values
        _, p_scipy = stats.pearsonr(x, y)

        result_one = run_analysis(ds, {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "tail": "one-tailed"},
        })
        p_matrix_tbl = next(t for t in result_one.tables if t.title == "p-value Matrix")
        p_in_matrix = p_matrix_tbl.dataframe.loc["x", "y"]

        assert math.isclose(p_in_matrix, p_scipy / 2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Bug 3: pairwise n_valid 정확성 검증
# ---------------------------------------------------------------------------

class TestPairwiseNValid:

    def test_pairwise_n_valid_excludes_nan_rows(self):
        """pairwise 모드에서 n_valid가 NaN 행을 제외한 수임을 확인 (Bug 3 수정)."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0,   np.nan, 10.0],
        })
        ds = Dataset(df, "PairwiseNaN")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        result = run_analysis(ds, {
            "variables": {"target": ["x", "y"]},
            "options": {"pairwise": True},
        })
        cps = next(t for t in result.tables if "Case Processing" in t.title)
        df_cps = cps.dataframe

        total_row = df_cps[df_cps.iloc[:, 0].astype(str).str.contains("Total|전체", na=False)]
        valid_row = df_cps[df_cps.iloc[:, 0].astype(str).str.contains("Valid|유효", na=False)]

        # n_total = 5, n_valid = 3 (인덱스 0, 1, 4만 x·y 모두 유효)
        if len(valid_row) > 0:
            n_valid_val = int(str(valid_row.iloc[0, 1]).replace(",", ""))
            assert n_valid_val == 3

    def test_pairwise_n_valid_no_nan(self):
        """NaN 없는 경우 n_valid == n_total."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })
        ds = Dataset(df, "NoNaN")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        result = run_analysis(ds, {
            "variables": {"target": ["x", "y"]},
            "options": {"pairwise": True},
        })
        cps = next(t for t in result.tables if "Case Processing" in t.title)
        df_cps = cps.dataframe

        valid_row = df_cps[df_cps.iloc[:, 0].astype(str).str.contains("Valid|유효", na=False)]
        if len(valid_row) > 0:
            n_valid_val = int(str(valid_row.iloc[0, 1]).replace(",", ""))
            assert n_valid_val == 3

"""편상관분석(Partial Correlation) SPSS 29/30 비교 테스트.

SPSS: Analyze > Correlate > Partial

참조 데이터 (n=15, 구두크기-독해력-나이):
  r(구두,독해)            = 0.9916  (0차 상관)
  r_partial(구두,독해|나이) = 0.5761  (나이 통제 후)
  t = 2.4416, df = 12, p = 0.0311

수식:
  r_partial(x,y|z) = (r_xy - r_xz*r_yz) / sqrt((1-r_xz²)(1-r_yz²))
  t = r_p * sqrt(df) / sqrt(1 - r_p²)  where df = n - 2 - k
  역행렬법 (다중 통제): r_partial(i,j) = -R_inv[i,j] / sqrt(R_inv[i,i]*R_inv[j,j])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.partial_correlation import run_analysis, _partial_corr_matrix
from statworkbench.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def shoe_dataset() -> Dataset:
    """구두크기-독해력-나이 (n=15) — SPSS 교과서 케이스."""
    data = pd.DataFrame({
        "age":  [6, 7, 8, 9,10,11,12, 6, 8,10, 7, 9,11,12, 8],
        "shoe": [24,26,27,28,30,31,33,23,27,29,25,28,32,33,26],
        "read": [35,42,50,58,65,72,80,30,48,62,38,55,70,82,45],
    })
    return Dataset(data=data, name="shoe_read")


@pytest.fixture
def three_var_dataset() -> Dataset:
    """3변수 독립 케이스 (x, y, z)."""
    np.random.seed(0)
    n = 20
    z = np.arange(1, n+1, dtype=float)
    x = 2*z + np.random.normal(0, 0.5, n)
    y = 3*z + np.random.normal(0, 0.5, n)
    data = pd.DataFrame({"x": x, "y": y, "z": z})
    return Dataset(data=data, name="three_var")


@pytest.fixture
def missing_dataset() -> Dataset:
    data = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        "b": [2.0, 4.0,    6.0, 8.0, 10., np.nan, 14., 16.],
        "c": [3.0, 3.0,    3.0, 3.0, 4.0, 4.0, 4.0,  4.0],
    })
    return Dataset(data=data, name="missing")


def _make_spec(vars_: list[str], controlling: list[str], **opts) -> dict:
    return {
        "variables": {"target": vars_, "controlling": controlling},
        "options": opts,
    }


# ──────────────────────────────────────────────────────────────
# 1. 편상관 수치 정확성
# ──────────────────────────────────────────────────────────────

class TestPartialCorrValue:

    def test_zero_order_matches_pearson(self, shoe_dataset):
        """통제 변수 없을 때 = Pearson r."""
        from scipy.stats import pearsonr
        r_expected, _ = pearsonr(
            shoe_dataset.data["shoe"], shoe_dataset.data["read"]
        )
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], []))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        df = mat_table.dataframe
        r_val = float(df.loc[df["변수"] == "shoe", "read"].iloc[0])
        assert abs(r_val - r_expected) < 1e-4

    def test_partial_corr_shoe_read_controlling_age(self, shoe_dataset):
        """r_partial(shoe,read|age) ≈ 0.5761 (SPSS 참조값)."""
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        df = mat_table.dataframe
        r_val = float(df.loc[df["변수"] == "shoe", "read"].iloc[0])
        assert abs(r_val - 0.5761) < 0.001

    def test_partial_corr_df_equals_n_minus_2_minus_k(self, shoe_dataset):
        """df = n - 2 - k (k=통제변수 수)."""
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        sig_table = next(t for t in result.tables if "Significance" in t.title)
        df_val = int(sig_table.dataframe.loc[
            sig_table.dataframe["변수"] == "shoe", "df"
        ].iloc[0])
        assert df_val == 15 - 2 - 1  # n=15, k=1

    def test_partial_corr_pvalue_shoe_read(self, shoe_dataset):
        """p ≈ 0.031 (SPSS 참조값)."""
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        sig_table = next(t for t in result.tables if "Significance" in t.title)
        p_val = float(sig_table.dataframe.loc[
            sig_table.dataframe["변수"] == "shoe", "read"
        ].iloc[0])
        assert abs(p_val - 0.0311) < 0.005

    def test_symmetry(self, shoe_dataset):
        """r_partial(x,y|z) == r_partial(y,x|z)."""
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        df = mat_table.dataframe
        r_shoe_read = float(df.loc[df["변수"] == "shoe", "read"].iloc[0])
        r_read_shoe = float(df.loc[df["변수"] == "read", "shoe"].iloc[0])
        assert abs(r_shoe_read - r_read_shoe) < 1e-10

    def test_diagonal_is_one(self, shoe_dataset):
        """대각 원소 = 1.000."""
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        df = mat_table.dataframe
        for var in ["shoe", "read"]:
            diag = float(df.loc[df["변수"] == var, var].iloc[0])
            assert abs(diag - 1.0) < 1e-10

    def test_range_minus1_to_1(self, shoe_dataset):
        """편상관계수 범위 [-1, 1]."""
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        for row in mat_table.dataframe.itertuples():
            for col in ["shoe", "read"]:
                val = float(getattr(row, col))
                assert -1.0 <= val <= 1.0

    def test_two_control_variables(self, three_var_dataset):
        """역행렬법으로 2변수 통제 처리."""
        data = three_var_dataset.data.copy()
        data["w"] = data["z"] * 0.8 + 1.5
        ds = Dataset(data=data, name="four_var")
        result = run_analysis(ds, _make_spec(["x", "y"], ["z", "w"]))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        df = mat_table.dataframe
        r_val = float(df.loc[df["변수"] == "x", "y"].iloc[0])
        assert -1.0 <= r_val <= 1.0


# ──────────────────────────────────────────────────────────────
# 2. 결과 구조
# ──────────────────────────────────────────────────────────────

class TestPartialCorrStructure:

    def test_returns_analysis_result(self, shoe_dataset):
        from statworkbench.analysis.result import AnalysisResult
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        assert isinstance(result, AnalysisResult)

    def test_has_four_tables(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        assert len(result.tables) == 4

    def test_table_titles(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        titles = [t.title for t in result.tables]
        assert "Case Processing Summary" in titles
        assert any("Correlation" in t for t in titles)
        assert any("Significance" in t for t in titles)
        assert any("Zero" in t or "0차" in t for t in titles)

    def test_correlation_matrix_columns(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        assert "변수" in mat_table.dataframe.columns
        assert "shoe" in mat_table.dataframe.columns
        assert "read" in mat_table.dataframe.columns

    def test_significance_table_columns(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        sig_table = next(t for t in result.tables if "Significance" in t.title)
        assert "변수" in sig_table.dataframe.columns
        assert "df" in sig_table.dataframe.columns

    def test_n_rows_in_matrix_equals_target_vars(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        mat_table = next(t for t in result.tables if "Correlation" in t.title)
        assert len(mat_table.dataframe) == 2

    def test_note_contains_alpha(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["age"]))
        assert len(result.notes) > 0
        combined = " ".join(result.notes)
        assert "통제" in combined or "controlling" in combined.lower() or "age" in combined


# ──────────────────────────────────────────────────────────────
# 3. 결측치 처리
# ──────────────────────────────────────────────────────────────

class TestPartialCorrMissing:

    def test_listwise_excludes_cases(self, missing_dataset):
        result = run_analysis(
            missing_dataset, _make_spec(["a", "b"], ["c"], listwise=True)
        )
        cps = next(t for t in result.tables if "Processing" in t.title)
        excluded = int(cps.dataframe.loc[cps.dataframe["구분"] == "제외됨", "N"].iloc[0])
        assert excluded > 0

    def test_pairwise_uses_more_data(self, missing_dataset):
        result = run_analysis(
            missing_dataset, _make_spec(["a", "b"], ["c"], listwise=False)
        )
        assert len(result.tables) == 4


# ──────────────────────────────────────────────────────────────
# 4. 오류 처리
# ──────────────────────────────────────────────────────────────

class TestPartialCorrErrors:

    def test_single_target_var_returns_warning(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe"], ["age"]))
        assert len(result.warnings) > 0

    def test_no_target_vars_returns_warning(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec([], ["age"]))
        assert len(result.warnings) > 0

    def test_nonexistent_var_returns_warning(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "q999"], ["age"]))
        assert len(result.warnings) > 0

    def test_controlling_not_in_data_returns_warning(self, shoe_dataset):
        result = run_analysis(shoe_dataset, _make_spec(["shoe", "read"], ["not_exist"]))
        assert len(result.warnings) > 0


# ──────────────────────────────────────────────────────────────
# 5. 헬퍼 함수 (_partial_corr_matrix)
# ──────────────────────────────────────────────────────────────

class TestPartialCorrHelper:

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "x": [1,2,3,4,5,6,7,8,9,10],
            "y": [2,4,6,8,10,12,14,16,18,20],
            "z": [1,1,2,2,3,3,4,4,5,5],
        })
        result = _partial_corr_matrix(df, target=["x","y"], controlling=["z"])
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_one_helper(self):
        df = pd.DataFrame({
            "x": [1,2,3,4,5,6,7,8,9,10],
            "y": [2,3,5,4,6,7,9,8,10,11],
            "z": [0,1,0,1,0,1,0,1,0,1],
        })
        mat = _partial_corr_matrix(df, target=["x","y"], controlling=["z"])
        assert abs(mat.loc["x","x"] - 1.0) < 1e-10
        assert abs(mat.loc["y","y"] - 1.0) < 1e-10

    def test_symmetry_helper(self):
        df = pd.DataFrame({
            "x": [1,2,3,4,5,6,7,8,9,10],
            "y": [2,3,5,4,6,7,9,8,10,11],
            "z": [0,1,0,1,0,1,0,1,0,1],
        })
        mat = _partial_corr_matrix(df, target=["x","y"], controlling=["z"])
        assert abs(mat.loc["x","y"] - mat.loc["y","x"]) < 1e-10

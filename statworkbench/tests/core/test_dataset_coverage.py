"""core/dataset.py 커버리지 보강 테스트.

미커버 라인:
  57      : del self._variables[col] — data setter에서 컬럼 삭제 시
  67      : variables setter
  71      : var_names property
  75      : is_empty property
  98      : get_column()
  137-138 : copy()
  184     : add_variable() with meta 인자
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType
from statworkbench.core.variable import VariableMeta


@pytest.fixture
def simple_ds():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(df, name="TestDS")


# ---------------------------------------------------------------------------
# Line 57: data setter — 컬럼 줄면 변수도 삭제
# ---------------------------------------------------------------------------

class TestDataSetterDeletesVariable:

    def test_removing_col_removes_variable(self, simple_ds):
        """data setter에 x 없는 DataFrame → _variables에서 x 삭제(57)."""
        new_df = pd.DataFrame({"y": [10.0, 11.0, 12.0]})
        simple_ds.data = new_df
        assert "x" not in simple_ds.variables
        assert "y" in simple_ds.variables


# ---------------------------------------------------------------------------
# Line 67: variables setter
# ---------------------------------------------------------------------------

class TestVariablesSetter:

    def test_variables_setter(self, simple_ds):
        """variables setter 직접 호출(67)."""
        meta = VariableMeta(name="x")
        simple_ds.variables = {"x": meta}
        assert "x" in simple_ds.variables
        assert simple_ds.variables["x"] is meta


# ---------------------------------------------------------------------------
# Line 71: var_names property
# ---------------------------------------------------------------------------

class TestVarNames:

    def test_var_names_returns_list(self, simple_ds):
        """var_names → list of column names(71)."""
        names = simple_ds.var_names
        assert isinstance(names, list)
        assert "x" in names and "y" in names


# ---------------------------------------------------------------------------
# Line 75: is_empty property
# ---------------------------------------------------------------------------

class TestIsEmpty:

    def test_is_empty_false_for_nonempty(self, simple_ds):
        """is_empty → False for non-empty Dataset(75)."""
        assert simple_ds.is_empty is False

    def test_is_empty_true_for_empty(self):
        """is_empty → True for empty DataFrame."""
        ds = Dataset(pd.DataFrame(), name="Empty")
        assert ds.is_empty is True


# ---------------------------------------------------------------------------
# Line 98: get_column()
# ---------------------------------------------------------------------------

class TestGetColumn:

    def test_get_column_returns_series(self, simple_ds):
        """get_column('x') → pd.Series(98)."""
        col = simple_ds.get_column("x")
        assert isinstance(col, pd.Series)
        assert list(col) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Lines 137-138: copy()
# ---------------------------------------------------------------------------

class TestCopy:

    def test_copy_returns_independent_dataset(self, simple_ds):
        """copy() → 독립 Dataset 반환(137-138)."""
        copied = simple_ds.copy()
        assert isinstance(copied, Dataset)
        # 독립성 확인
        copied.data.loc[0, "x"] = 999.0
        assert simple_ds.data.loc[0, "x"] != 999.0


# ---------------------------------------------------------------------------
# Line 184: add_variable() with meta
# ---------------------------------------------------------------------------

class TestAddVariableWithMeta:

    def test_add_variable_with_meta(self, simple_ds):
        """add_variable(meta=...) → meta 그대로 저장(184)."""
        custom_meta = VariableMeta(name="z", label="Z 변수")
        simple_ds.add_variable("z", data=pd.Series([7.0, 8.0, 9.0]), meta=custom_meta)
        assert "z" in simple_ds.variables
        assert simple_ds.variables["z"].label == "Z 변수"

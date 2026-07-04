"""SPSSGridModel get_dataframe() dtype 변환 캐싱 테스트."""

import pandas as pd
import pytest

from nuristat.core.variable import VariableMeta
from nuristat.core.typing import StorageType, MeasureType
from nuristat.ui.models.spss_grid_model import SPSSGridModel


def _make_model(n_rows: int = 10, n_cols: int = 3) -> SPSSGridModel:
    df = pd.DataFrame({
        f"v{i}": [float(j) for j in range(n_rows)] for i in range(n_cols)
    })
    variables = {
        f"v{i}": VariableMeta(
            name=f"v{i}", label=f"변수{i}",
            storage_type=StorageType.FLOAT, measure=MeasureType.SCALE,
        )
        for i in range(n_cols)
    }
    return SPSSGridModel(dataframe=df, variables=variables)


def test_cache_starts_dirty(qapp):
    """초기 상태는 _cache_dirty=True여야 한다."""
    model = _make_model()
    assert model._cache_dirty is True
    assert model._typed_df_cache is None


def test_get_dataframe_sets_cache(qapp):
    """첫 호출 후 캐시가 유효 상태여야 한다."""
    model = _make_model()
    _ = model.get_dataframe()
    assert model._cache_dirty is False
    assert model._typed_df_cache is not None


def test_get_dataframe_cached_no_recompute(qapp):
    """캐시 유효 시 재변환 없이 즉시 반환해야 한다."""
    model = _make_model()
    df1 = model.get_dataframe()
    # 캐시 상태에서 다시 호출
    assert model._cache_dirty is False
    df2 = model.get_dataframe()
    pd.testing.assert_frame_equal(df1, df2)


def test_setdata_invalidates_cache(qapp):
    """setData 후 캐시가 무효화돼야 한다."""
    from PySide6.QtCore import Qt
    model = _make_model()
    _ = model.get_dataframe()
    assert model._cache_dirty is False
    idx = model.index(0, 0)
    model.setData(idx, "99.0", Qt.ItemDataRole.EditRole)
    assert model._cache_dirty is True


def test_sort_invalidates_cache(qapp):
    """sort_by_column 후 캐시가 무효화돼야 한다."""
    model = _make_model()
    _ = model.get_dataframe()
    model.sort_by_column(0)
    assert model._cache_dirty is True


def test_add_row_invalidates_cache(qapp):
    """add_row 후 캐시가 무효화돼야 한다."""
    model = _make_model()
    _ = model.get_dataframe()
    model.add_row()
    assert model._cache_dirty is True


def test_undo_invalidates_cache(qapp):
    """undo 후 캐시가 무효화돼야 한다."""
    from PySide6.QtCore import Qt
    model = _make_model()
    idx = model.index(0, 0)
    model.setData(idx, "1.0", Qt.ItemDataRole.EditRole)
    _ = model.get_dataframe()
    model.undo()
    assert model._cache_dirty is True


def test_get_dataframe_cache_hit_returns_same_object(qapp):
    """캐시 유효 시 copy() 없이 캐시 객체 자체를 반환해야 한다(P1-3, 읽기전용 계약).

    호출부(data_view.py:779)가 read-only 사용만 하므로 copy 비용을 없앤다.
    """
    model = _make_model()
    df1 = model.get_dataframe()
    assert model._cache_dirty is False
    df2 = model.get_dataframe()
    assert df1 is df2

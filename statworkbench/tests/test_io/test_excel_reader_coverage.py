"""excel_reader.py 커버리지 보강 테스트.

미커버 라인:
  62   : FileNotFoundError → FileReadError
  64   : PermissionError → FileReadError
  67-68: OSError → FileReadError
  73   : dict인 df가 비어 있을 때 → FileReadError
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from statworkbench.core.exceptions import FileReadError
from statworkbench.io.excel_reader import read_excel


@pytest.fixture
def excel_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "data.xlsx"
    df.to_excel(str(path), index=False)
    return path


class TestExcelReaderExceptionPaths:

    def test_file_not_found_raises(self, excel_file):
        """pd.read_excel FileNotFoundError → FileReadError (line 62)."""
        with patch("pandas.read_excel", side_effect=FileNotFoundError("not found")):
            with pytest.raises(FileReadError) as exc_info:
                read_excel(str(excel_file))
        assert "찾을 수 없습니다" in str(exc_info.value)

    def test_permission_error_raises(self, excel_file):
        """pd.read_excel PermissionError → FileReadError (line 64)."""
        with patch("pandas.read_excel", side_effect=PermissionError("denied")):
            with pytest.raises(FileReadError) as exc_info:
                read_excel(str(excel_file))
        assert "권한" in str(exc_info.value)

    def test_value_error_raises(self, excel_file):
        """pd.read_excel ValueError → FileReadError (line 66-67)."""
        with patch("pandas.read_excel", side_effect=ValueError("bad sheet")):
            with pytest.raises(FileReadError) as exc_info:
                read_excel(str(excel_file))
        assert "파싱 오류" in str(exc_info.value)

    def test_oserror_raises(self, excel_file):
        """pd.read_excel OSError → FileReadError (line 67-68)."""
        with patch("pandas.read_excel", side_effect=OSError("disk error")):
            with pytest.raises(FileReadError) as exc_info:
                read_excel(str(excel_file))
        assert "disk error" in str(exc_info.value)

    def test_empty_dict_result_raises(self, excel_file):
        """read_excel 반환값이 빈 dict → FileReadError (line 73)."""
        with patch("pandas.read_excel", return_value={}):
            with pytest.raises(FileReadError) as exc_info:
                read_excel(str(excel_file))
        assert "비어 있습니다" in str(exc_info.value)

    def test_dict_result_uses_first_sheet(self, excel_file):
        """read_excel 반환값이 비지 않은 dict → 첫 번째 시트 DataFrame 사용 (line 75)."""
        df_a = pd.DataFrame({"x": [1, 2]})
        df_b = pd.DataFrame({"y": [3, 4]})
        with patch("pandas.read_excel", return_value={"Sheet1": df_a, "Sheet2": df_b}):
            ds = read_excel(str(excel_file))
        assert list(ds.data.columns) == ["x"]
        assert len(ds.data) == 2

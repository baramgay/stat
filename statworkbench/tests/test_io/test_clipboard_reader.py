"""Tests for clipboard_reader module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import IORError
from statworkbench.io.clipboard_reader import read_clipboard, read_clipboard_from_qt


# ===========================================================================
# read_clipboard_from_qt 테스트 (실제 클립보드 불필요)
# ===========================================================================


class TestReadClipboardFromQt:
    """read_clipboard_from_qt 테스트 — mock 불필요."""

    TAB_TWO_COL = "a\tb\n1\t2\n3\t4"
    COMMA_TWO_COL = "x,y\n10,20\n30,40"
    SEMI_TWO_COL = "p;q\n5;6\n7;8"

    # ------------------------------------------------------------------
    # 1. 탭 구분 기본 파싱
    # ------------------------------------------------------------------
    def test_tab_delimiter_columns(self):
        """탭 구분 문자열에서 컬럼 [a, b]가 생성된다."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert list(ds.data.columns) == ["a", "b"]

    def test_tab_delimiter_values(self):
        """탭 구분 문자열의 값이 정확히 파싱된다."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert ds.data.iloc[0, 0] == 1
        assert ds.data.iloc[1, 1] == 4

    def test_tab_delimiter_shape(self):
        """탭 구분 2행 2열 → shape (2, 2)."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert ds.data.shape == (2, 2)

    # ------------------------------------------------------------------
    # 2. 쉼표 구분
    # ------------------------------------------------------------------
    def test_comma_delimiter(self):
        """delimiter=',' 옵션으로 쉼표 구분 파싱."""
        ds = read_clipboard_from_qt(self.COMMA_TWO_COL, delimiter=",")
        assert list(ds.data.columns) == ["x", "y"]
        assert ds.data.iloc[0, 0] == 10

    # ------------------------------------------------------------------
    # 3. 세미콜론 구분
    # ------------------------------------------------------------------
    def test_semicolon_delimiter(self):
        """delimiter=';' 옵션으로 세미콜론 구분 파싱."""
        ds = read_clipboard_from_qt(self.SEMI_TWO_COL, delimiter=";")
        assert list(ds.data.columns) == ["p", "q"]
        assert ds.data.iloc[1, 1] == 8

    # ------------------------------------------------------------------
    # 4. 여러 행 정확성 (값 및 shape)
    # ------------------------------------------------------------------
    def test_multi_row_shape(self):
        """3행 3열 탭 구분 문자열 → shape (3, 3)."""
        text = "a\tb\tc\n1\t2\t3\n4\t5\t6\n7\t8\t9"
        ds = read_clipboard_from_qt(text)
        assert ds.data.shape == (3, 3)

    def test_multi_row_values(self):
        """3행 3열 마지막 값 확인."""
        text = "a\tb\tc\n1\t2\t3\n4\t5\t6\n7\t8\t9"
        ds = read_clipboard_from_qt(text)
        assert ds.data.iloc[2, 2] == 9

    # ------------------------------------------------------------------
    # 5. 반환 타입
    # ------------------------------------------------------------------
    def test_returns_dataset(self):
        """반환값이 Dataset 인스턴스여야 한다."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert isinstance(ds, Dataset)

    # ------------------------------------------------------------------
    # 6. source_info["format"]
    # ------------------------------------------------------------------
    def test_source_info_format(self):
        """source_info['format'] == 'clipboard_qt'."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert ds.source_info["format"] == "clipboard_qt"

    # ------------------------------------------------------------------
    # 7. source_info n_rows / n_columns
    # ------------------------------------------------------------------
    def test_source_info_n_rows(self):
        """source_info['n_rows'] == 실제 행 수."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert ds.source_info["n_rows"] == 2

    def test_source_info_n_columns(self):
        """source_info['n_columns'] == 실제 열 수."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert ds.source_info["n_columns"] == 2

    # ------------------------------------------------------------------
    # 8. 빈 문자열 → IORError
    # ------------------------------------------------------------------
    def test_empty_string_raises_iorerror(self):
        """빈 문자열 입력 시 IORError가 발생한다."""
        with pytest.raises(IORError):
            read_clipboard_from_qt("")

    # ------------------------------------------------------------------
    # 9. 헤더만 있고 데이터 없음 → IORError
    # ------------------------------------------------------------------
    def test_header_only_raises_iorerror(self):
        """헤더만 있고 데이터 행이 없으면 IORError가 발생한다."""
        with pytest.raises(IORError):
            read_clipboard_from_qt("a\tb\n")

    # ------------------------------------------------------------------
    # 10. 숫자형 컬럼 dtype 확인
    # ------------------------------------------------------------------
    def test_numeric_dtype(self):
        """정수 데이터 컬럼은 숫자형(int 또는 float) dtype이어야 한다."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert pd.api.types.is_numeric_dtype(ds.data["a"])
        assert pd.api.types.is_numeric_dtype(ds.data["b"])

    # ------------------------------------------------------------------
    # 11. name == "ClipboardData"
    # ------------------------------------------------------------------
    def test_dataset_name(self):
        """Dataset.name 은 'ClipboardData' 여야 한다."""
        ds = read_clipboard_from_qt(self.TAB_TWO_COL)
        assert ds.name == "ClipboardData"

    # ------------------------------------------------------------------
    # 12. 공백 포함 값 처리
    # ------------------------------------------------------------------
    def test_values_with_spaces(self):
        """값에 공백이 포함된 문자열도 정상 파싱된다."""
        text = "name\tscore\nAlice Smith\t95\nBob Jones\t88"
        ds = read_clipboard_from_qt(text)
        assert ds.data.iloc[0, 0] == "Alice Smith"
        assert ds.data.iloc[1, 1] == 88

    # ------------------------------------------------------------------
    # 13. 단일 컬럼 파싱
    # ------------------------------------------------------------------
    def test_single_column(self):
        """단일 컬럼 탭 구분 문자열도 정상 파싱된다."""
        text = "value\n10\n20\n30"
        ds = read_clipboard_from_qt(text)
        assert ds.data.shape == (3, 1)
        assert list(ds.data.columns) == ["value"]


# ===========================================================================
# read_clipboard 테스트 (pd.read_clipboard mock 필요)
# ===========================================================================


class TestReadClipboard:
    """read_clipboard 테스트 — pd.read_clipboard를 mock으로 대체."""

    MOCK_DF = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    PATCH_TARGET = "statworkbench.io.clipboard_reader.pd.read_clipboard"

    # ------------------------------------------------------------------
    # 14. 정상 DataFrame → Dataset 반환
    # ------------------------------------------------------------------
    def test_returns_dataset(self):
        """정상 DataFrame이 반환될 때 Dataset 인스턴스를 반환한다."""
        with patch(self.PATCH_TARGET, return_value=self.MOCK_DF.copy()):
            ds = read_clipboard()
        assert isinstance(ds, Dataset)

    # ------------------------------------------------------------------
    # 15. source_info["format"] == "clipboard"
    # ------------------------------------------------------------------
    def test_source_info_format(self):
        """source_info['format'] == 'clipboard'."""
        with patch(self.PATCH_TARGET, return_value=self.MOCK_DF.copy()):
            ds = read_clipboard()
        assert ds.source_info["format"] == "clipboard"

    # ------------------------------------------------------------------
    # 16. EmptyDataError → IORError
    # ------------------------------------------------------------------
    def test_empty_data_error_raises_iorerror(self):
        """pd.errors.EmptyDataError 발생 시 IORError로 변환된다."""
        with patch(self.PATCH_TARGET, side_effect=pd.errors.EmptyDataError):
            with pytest.raises(IORError, match="클립보드에 데이터가 없습니다"):
                read_clipboard()

    # ------------------------------------------------------------------
    # 17. ParserError → IORError
    # ------------------------------------------------------------------
    def test_parser_error_raises_iorerror(self):
        """pd.errors.ParserError 발생 시 IORError로 변환된다."""
        with patch(self.PATCH_TARGET, side_effect=pd.errors.ParserError("bad data")):
            with pytest.raises(IORError, match="파싱 오류"):
                read_clipboard()

    # ------------------------------------------------------------------
    # 18. OSError → IORError
    # ------------------------------------------------------------------
    def test_os_error_raises_iorerror(self):
        """OSError 발생 시 IORError로 변환된다."""
        with patch(self.PATCH_TARGET, side_effect=OSError("no display")):
            with pytest.raises(IORError, match="클립보드 접근 오류"):
                read_clipboard()

    # ------------------------------------------------------------------
    # 19. RuntimeError → IORError
    # ------------------------------------------------------------------
    def test_runtime_error_raises_iorerror(self):
        """RuntimeError 발생 시 IORError로 변환된다."""
        with patch(self.PATCH_TARGET, side_effect=RuntimeError("runtime fail")):
            with pytest.raises(IORError, match="클립보드를 읽을 수 없습니다"):
                read_clipboard()

    # ------------------------------------------------------------------
    # 20. 빈 DataFrame 반환 → IORError
    # ------------------------------------------------------------------
    def test_empty_dataframe_raises_iorerror(self):
        """pd.read_clipboard가 빈 DataFrame을 반환하면 IORError가 발생한다."""
        with patch(self.PATCH_TARGET, return_value=pd.DataFrame()):
            with pytest.raises(IORError, match="유효한 테이블 데이터가 없습니다"):
                read_clipboard()

    # ------------------------------------------------------------------
    # 21. n_rows, n_columns 정확성
    # ------------------------------------------------------------------
    def test_source_info_n_rows(self):
        """source_info['n_rows'] == 실제 행 수."""
        with patch(self.PATCH_TARGET, return_value=self.MOCK_DF.copy()):
            ds = read_clipboard()
        assert ds.source_info["n_rows"] == 3

    def test_source_info_n_columns(self):
        """source_info['n_columns'] == 실제 열 수."""
        with patch(self.PATCH_TARGET, return_value=self.MOCK_DF.copy()):
            ds = read_clipboard()
        assert ds.source_info["n_columns"] == 2

    # ------------------------------------------------------------------
    # 22. kwargs 전달 확인 (sep 등)
    # ------------------------------------------------------------------
    def test_kwargs_forwarded(self):
        """kwargs(sep 등)가 pd.read_clipboard에 그대로 전달된다."""
        with patch(self.PATCH_TARGET, return_value=self.MOCK_DF.copy()) as mock_rc:
            read_clipboard(sep=r"\s+", header=0)
        mock_rc.assert_called_once_with(sep=r"\s+", header=0)

    # ------------------------------------------------------------------
    # 추가: Dataset.name 확인
    # ------------------------------------------------------------------
    def test_dataset_name(self):
        """Dataset.name 은 'ClipboardData' 여야 한다."""
        with patch(self.PATCH_TARGET, return_value=self.MOCK_DF.copy()):
            ds = read_clipboard()
        assert ds.name == "ClipboardData"

    # ------------------------------------------------------------------
    # 추가: 정상 반환 값 확인
    # ------------------------------------------------------------------
    def test_data_values_preserved(self):
        """pd.read_clipboard 반환값이 Dataset.data에 그대로 담긴다."""
        with patch(self.PATCH_TARGET, return_value=self.MOCK_DF.copy()):
            ds = read_clipboard()
        pd.testing.assert_frame_equal(ds.data, self.MOCK_DF)

    # ------------------------------------------------------------------
    # 추가: 단일 행 DataFrame 정상 처리
    # ------------------------------------------------------------------
    def test_single_row_dataframe(self):
        """단일 행 DataFrame도 정상적으로 Dataset으로 변환된다."""
        single_row = pd.DataFrame({"a": [42], "b": ["hello"]})
        with patch(self.PATCH_TARGET, return_value=single_row):
            ds = read_clipboard()
        assert ds.data.shape == (1, 2)
        assert ds.source_info["n_rows"] == 1

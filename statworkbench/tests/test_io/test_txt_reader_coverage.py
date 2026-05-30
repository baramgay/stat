"""txt_reader.py 커버리지 보강 테스트.

미커버 라인(76-85):
  - UnicodeDecodeError → FileReadError
  - pd.errors.EmptyDataError → FileReadError
  - pd.errors.ParserError → FileReadError
  - OSError → FileReadError

전략: 자동 감지를 건너뛰기 위해 encoding/delimiter를 명시 지정 후
      unittest.mock.patch 으로 pd.read_csv 예외를 주입한다.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from statworkbench.core.exceptions import FileReadError
from statworkbench.io.txt_reader import read_txt


@pytest.fixture
def plain_file(tmp_path):
    """탐지를 통과할 최소 유효 TXT 파일."""
    f = tmp_path / "data.txt"
    f.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    return f


class TestReadTxtExceptionPaths:

    def test_unicode_decode_error_raises_file_read_error(self, plain_file):
        """pd.read_csv UnicodeDecodeError → FileReadError."""
        with patch("pandas.read_csv", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "reason")):
            with pytest.raises(FileReadError) as exc_info:
                read_txt(str(plain_file), encoding="utf-8", delimiter=",")
        assert "인코딩 오류" in str(exc_info.value)

    def test_empty_data_error_raises_file_read_error(self, plain_file):
        """pd.read_csv EmptyDataError → FileReadError."""
        with patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError("empty")):
            with pytest.raises(FileReadError) as exc_info:
                read_txt(str(plain_file), encoding="utf-8", delimiter=",")
        assert "데이터가 없습니다" in str(exc_info.value)

    def test_parser_error_raises_file_read_error(self, plain_file):
        """pd.read_csv ParserError → FileReadError."""
        with patch("pandas.read_csv", side_effect=pd.errors.ParserError("bad token")):
            with pytest.raises(FileReadError) as exc_info:
                read_txt(str(plain_file), encoding="utf-8", delimiter=",")
        assert "파싱 오류" in str(exc_info.value)

    def test_oserror_raises_file_read_error(self, plain_file):
        """pd.read_csv OSError → FileReadError."""
        with patch("pandas.read_csv", side_effect=OSError("disk failure")):
            with pytest.raises(FileReadError) as exc_info:
                read_txt(str(plain_file), encoding="utf-8", delimiter=",")
        assert "disk failure" in str(exc_info.value)

    def test_unicode_error_message_includes_encoding(self, plain_file):
        """에러 메시지에 encoding 이름이 포함된다."""
        with patch("pandas.read_csv", side_effect=UnicodeDecodeError("euc-kr", b"", 0, 1, "reason")):
            with pytest.raises(FileReadError) as exc_info:
                read_txt(str(plain_file), encoding="euc-kr", delimiter=",")
        assert "euc-kr" in str(exc_info.value)

"""csv_reader.py 커버리지 보강 테스트.

미커버 라인:
  53-57  : open(path, "rb") OSError → FileReadError
  79     : encoding alias "iso-8859-1" → "latin-1"
  81     : encoding alias "euc-kr" → "cp949"
  235-236: pd.read_csv OSError → FileReadError
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from nuristat.core.exceptions import FileReadError
from nuristat.io.csv_reader import _detect_encoding, read_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "data.csv"
    path.write_text(df.to_csv(index=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Lines 53-57: open(path, "rb") OSError → FileReadError
# ---------------------------------------------------------------------------

class TestReadBytesOSError:

    def test_read_bytes_oserror_raises_file_read_error(self, csv_file):
        """open(path, "rb") OSError → FileReadError."""
        with patch("builtins.open", side_effect=OSError("disk error")):
            with pytest.raises(FileReadError):
                _detect_encoding(csv_file)


# ---------------------------------------------------------------------------
# Lines 79: encoding alias "iso-8859-1" → "latin-1"
# ---------------------------------------------------------------------------

class TestEncodingAliasLatin1:

    def test_iso8859_alias_returns_latin1(self, csv_file):
        """chardet이 'iso-8859-1' 반환 → _detect_encoding 반환값 'latin-1'."""
        with patch("chardet.detect", return_value={"encoding": "iso-8859-1", "confidence": 0.9}):
            result = _detect_encoding(csv_file)
        assert result == "latin-1"

    def test_latin1_alias_returns_latin1(self, csv_file):
        """chardet이 'latin1' 반환 → 'latin-1'."""
        with patch("chardet.detect", return_value={"encoding": "latin1", "confidence": 0.9}):
            result = _detect_encoding(csv_file)
        assert result == "latin-1"

    def test_iso8859_hyphen_alias(self, csv_file):
        """chardet이 'iso8859-1' 반환 → 'latin-1'."""
        with patch("chardet.detect", return_value={"encoding": "iso8859-1", "confidence": 0.85}):
            result = _detect_encoding(csv_file)
        assert result == "latin-1"


# ---------------------------------------------------------------------------
# Line 81: encoding alias "euc-kr" → "cp949"
# ---------------------------------------------------------------------------

class TestEncodingAliasCP949:

    def test_euckr_alias_returns_cp949(self, csv_file):
        """chardet이 'euc-kr' 반환 → 'cp949'."""
        with patch("chardet.detect", return_value={"encoding": "euc-kr", "confidence": 0.9}):
            result = _detect_encoding(csv_file)
        assert result == "cp949"

    def test_euckr_no_hyphen_alias(self, csv_file):
        """chardet이 'euckr' 반환 → 'cp949'."""
        with patch("chardet.detect", return_value={"encoding": "euckr", "confidence": 0.88}):
            result = _detect_encoding(csv_file)
        assert result == "cp949"


# ---------------------------------------------------------------------------
# Lines 235-236: pd.read_csv OSError → FileReadError
# ---------------------------------------------------------------------------

class TestReadCSVOSError:

    def test_read_csv_oserror_raises_file_read_error(self, csv_file):
        """pd.read_csv OSError → FileReadError."""
        with patch("pandas.read_csv", side_effect=OSError("disk full")):
            with pytest.raises(FileReadError) as exc_info:
                read_csv(csv_file, encoding="utf-8", delimiter=",")
        assert "disk full" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 정상 경로 확인
# ---------------------------------------------------------------------------

class TestNormalPaths:

    def test_utf8_sig_alias(self, csv_file):
        """chardet이 'utf-8-sig' 반환 → 'utf-8-sig' 그대로."""
        with patch("chardet.detect", return_value={"encoding": "utf-8-sig", "confidence": 0.99}):
            result = _detect_encoding(csv_file)
        assert result == "utf-8-sig"

    def test_ascii_returns_utf8(self, csv_file):
        """chardet이 'ascii' 반환 → 'utf-8'."""
        with patch("chardet.detect", return_value={"encoding": "ascii", "confidence": 0.99}):
            result = _detect_encoding(csv_file)
        assert result == "utf-8"

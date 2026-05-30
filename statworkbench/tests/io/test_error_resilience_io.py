"""Error-resilience and edge-case tests for IO modules.

Covers:
  - statworkbench.io.csv_reader.read_csv

All tests in this file must not produce unhandled exceptions.
Expected typed exceptions are asserted via pytest.raises.
Successful reads return a Dataset.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure src/ is on path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import (
    DelimiterDetectionError,
    FileReadError,
    IORError,
    StatWorkbenchError,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_temp_csv(content: str, encoding: str = "utf-8", suffix: str = ".csv") -> str:
    """Write content to a temporary file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, encoding=encoding, delete=False
    )
    f.write(content)
    f.close()
    return f.name


def cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ===========================================================================
# CSV Reader Edge Cases
# ===========================================================================

class TestCSVReaderEdgeCases:

    # -----------------------------------------------------------------------
    # Empty file
    # -----------------------------------------------------------------------

    def test_empty_file_raises(self):
        """Completely empty file — should raise FileReadError or IORError."""
        from statworkbench.io.csv_reader import read_csv

        path = make_temp_csv("")
        try:
            with pytest.raises((FileReadError, IORError, StatWorkbenchError)):
                read_csv(path)
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Header only, no data rows
    # -----------------------------------------------------------------------

    def test_header_only_no_data_rows(self):
        """File with only a header line and no data rows."""
        from statworkbench.io.csv_reader import read_csv

        path = make_temp_csv("col1,col2,col3\n")
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert len(result.data) == 0
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # All-missing column
    # -----------------------------------------------------------------------

    def test_all_missing_column(self):
        """File where one column is entirely empty (all NaN)."""
        from statworkbench.io.csv_reader import read_csv

        content = "a,b,c\n1,,3\n4,,6\n7,,9\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert result.data["b"].isna().all()
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Korean characters (cp949)
    # -----------------------------------------------------------------------

    def test_korean_characters_cp949(self):
        """File with Korean text encoded in cp949."""
        from statworkbench.io.csv_reader import read_csv

        content = "이름,나이,점수\n홍길동,25,90\n김철수,30,85\n"
        path = make_temp_csv(content, encoding="cp949")
        try:
            result = read_csv(path, encoding="cp949")
            assert isinstance(result, Dataset)
            assert len(result.data) == 2
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Mixed delimiters — auto-detection
    # -----------------------------------------------------------------------

    def test_semicolon_delimiter_auto_detection(self):
        """Semicolon-delimited file with auto delimiter detection."""
        from statworkbench.io.csv_reader import read_csv

        content = "a;b;c\n1;2;3\n4;5;6\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path, delimiter="auto")
            assert isinstance(result, Dataset)
            assert len(result.data.columns) == 3
        finally:
            cleanup(path)

    def test_tab_delimiter_auto_detection(self):
        """Tab-delimited file with auto delimiter detection."""
        from statworkbench.io.csv_reader import read_csv

        content = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path, delimiter="auto")
            assert isinstance(result, Dataset)
            assert len(result.data.columns) == 3
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Non-existent file path
    # -----------------------------------------------------------------------

    def test_nonexistent_file_raises_file_read_error(self):
        """Non-existent path must raise FileReadError."""
        from statworkbench.io.csv_reader import read_csv

        with pytest.raises(FileReadError):
            read_csv("C:/definitely/does/not/exist/file.csv")

    def test_nonexistent_file_raises_file_read_error_relative(self):
        """Relative non-existent path must raise FileReadError."""
        from statworkbench.io.csv_reader import read_csv

        with pytest.raises(FileReadError):
            read_csv("no_such_file_xyz_12345.csv")

    # -----------------------------------------------------------------------
    # Single column (no delimiter)
    # -----------------------------------------------------------------------

    def test_single_column_no_delimiter(self):
        """File with 1 column only — auto-detection falls back to comma."""
        from statworkbench.io.csv_reader import read_csv

        content = "value\n1\n2\n3\n4\n5\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert len(result.data) == 5
        finally:
            cleanup(path)

    def test_single_column_auto_delimiter(self):
        """Single column with auto delimiter — may raise DelimiterDetectionError or succeed."""
        from statworkbench.io.csv_reader import read_csv

        content = "value\n1\n2\n3\n"
        path = make_temp_csv(content)
        try:
            try:
                result = read_csv(path, delimiter="auto")
                # If it succeeds, it must return a Dataset
                assert isinstance(result, Dataset)
            except (DelimiterDetectionError, FileReadError, IORError, StatWorkbenchError):
                pass  # Typed exception — acceptable
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Very large header (100+ columns)
    # -----------------------------------------------------------------------

    def test_very_large_header_100_columns(self):
        """File with 100 columns in header."""
        from statworkbench.io.csv_reader import read_csv

        cols = [f"col{i}" for i in range(100)]
        header = ",".join(cols)
        row1 = ",".join([str(i) for i in range(100)])
        row2 = ",".join([str(i * 2) for i in range(100)])
        content = f"{header}\n{row1}\n{row2}\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert len(result.data.columns) == 100
            assert len(result.data) == 2
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # UTF-8 BOM
    # -----------------------------------------------------------------------

    def test_utf8_bom_file(self):
        """File with UTF-8 BOM (utf-8-sig)."""
        from statworkbench.io.csv_reader import read_csv

        content = "name,age,score\nAlice,25,90\nBob,30,85\n"
        path = make_temp_csv(content, encoding="utf-8-sig")
        try:
            result = read_csv(path, encoding="auto")
            assert isinstance(result, Dataset)
            # BOM should not appear in column names
            assert "name" in result.data.columns or "﻿name" not in result.data.columns
        finally:
            cleanup(path)

    def test_utf8_bom_explicit_encoding(self):
        """UTF-8-BOM file read with explicit encoding."""
        from statworkbench.io.csv_reader import read_csv

        content = "x,y\n1,2\n3,4\n"
        path = make_temp_csv(content, encoding="utf-8-sig")
        try:
            result = read_csv(path, encoding="utf-8-sig")
            assert isinstance(result, Dataset)
            assert len(result.data) == 2
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Mixed types in a column
    # -----------------------------------------------------------------------

    def test_mixed_types_column(self):
        """Column with mixed numeric and string values."""
        from statworkbench.io.csv_reader import read_csv

        content = "id,value\n1,3.14\n2,hello\n3,42\n4,\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert len(result.data) == 4
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Quoted fields with commas inside
    # -----------------------------------------------------------------------

    def test_quoted_fields_with_commas(self):
        """Fields containing commas wrapped in quotes."""
        from statworkbench.io.csv_reader import read_csv

        content = 'name,address\nAlice,"Seoul, Korea"\nBob,"Busan, Korea"\n'
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert len(result.data) == 2
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # File with Windows line endings (CRLF)
    # -----------------------------------------------------------------------

    def test_windows_crlf_line_endings(self):
        """File with \\r\\n line endings."""
        from statworkbench.io.csv_reader import read_csv

        content = "a,b,c\r\n1,2,3\r\n4,5,6\r\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert len(result.data) == 2
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Only two data rows (very small dataset)
    # -----------------------------------------------------------------------

    def test_two_data_rows(self):
        """Minimum practical dataset: header + 2 rows."""
        from statworkbench.io.csv_reader import read_csv

        content = "x,y\n1,2\n3,4\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert len(result.data) == 2
        finally:
            cleanup(path)

    # -----------------------------------------------------------------------
    # Dataset metadata
    # -----------------------------------------------------------------------

    def test_dataset_has_source_info(self):
        """Dataset returned by read_csv should carry source_info."""
        from statworkbench.io.csv_reader import read_csv

        content = "a,b\n1,2\n3,4\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            assert result.source_info.get("format") == "csv"
        finally:
            cleanup(path)

    def test_dataset_name_matches_stem(self):
        """Dataset name should match the file stem."""
        from statworkbench.io.csv_reader import read_csv

        content = "x,y\n1,2\n"
        path = make_temp_csv(content)
        try:
            result = read_csv(path)
            assert isinstance(result, Dataset)
            stem = Path(path).stem
            assert result.name == stem
        finally:
            cleanup(path)

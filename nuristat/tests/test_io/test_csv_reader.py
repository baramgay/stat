"""Tests for csv_reader and txt_reader modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import (
    DelimiterDetectionError,
    EncodingDetectionError,
    FileReadError,
)
from nuristat.io.csv_reader import _detect_delimiter, _detect_encoding, read_csv
from nuristat.io.txt_reader import read_txt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def utf8_csv(tmp_path: Path) -> str:
    """Create a UTF-8 encoded CSV file."""
    path = tmp_path / "utf8_data.csv"
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [85.5, 90.2, 78.0],
    })
    df.to_csv(path, index=False, encoding="utf-8")
    return str(path)


@pytest.fixture
def cp949_csv(tmp_path: Path) -> str:
    """Create a CP949 (Korean) encoded CSV file."""
    path = tmp_path / "cp949_data.csv"
    df = pd.DataFrame({
        "이름": ["홍길동", "김철수", "이영희"],
        "나이": [25, 30, 35],
        "점수": [85.5, 90.2, 78.0],
    })
    df.to_csv(path, index=False, encoding="cp949")
    return str(path)


@pytest.fixture
def tsv_file(tmp_path: Path) -> str:
    """Create a tab-separated (TSV) file."""
    path = tmp_path / "data.tsv"
    df = pd.DataFrame({
        "gene": ["BRCA1", "TP53", "EGFR"],
        "expression": [12.5, 8.3, 15.7],
        "p_value": [0.001, 0.05, 0.0001],
    })
    df.to_csv(path, index=False, sep="\t", encoding="utf-8")
    return str(path)


@pytest.fixture
def semicolon_csv(tmp_path: Path) -> str:
    """Create a semicolon-delimited CSV file."""
    path = tmp_path / "semicolon_data.csv"
    content = "a;b;c\n1;2;3\n4;5;6\n"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def pipe_delimited(tmp_path: Path) -> str:
    """Create a pipe-delimited text file."""
    path = tmp_path / "pipe_data.txt"
    content = "col_a|col_b|col_c\n10|20|30\n40|50|60\n"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def no_header_csv(tmp_path: Path) -> str:
    """Create a CSV without header row."""
    path = tmp_path / "no_header.csv"
    content = "1,apple,red\n2,banana,yellow\n3,cherry,red\n"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def empty_file(tmp_path: Path) -> str:
    """Create an empty file."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Encoding detection tests
# ---------------------------------------------------------------------------


class TestDetectEncoding:
    def test_utf8(self, utf8_csv: str) -> None:
        enc = _detect_encoding(utf8_csv)
        assert enc in ("utf-8", "ascii")

    def test_cp949(self, cp949_csv: str) -> None:
        enc = _detect_encoding(cp949_csv)
        assert enc in ("cp949", "euc-kr")

    def test_empty_file(self, empty_file: str) -> None:
        enc = _detect_encoding(empty_file)
        assert enc == "utf-8"

    def test_nonexistent_file(self) -> None:
        with pytest.raises(FileReadError):
            _detect_encoding("/nonexistent/path/file.csv")


# ---------------------------------------------------------------------------
# Delimiter detection tests
# ---------------------------------------------------------------------------


class TestDetectDelimiter:
    def test_comma(self, utf8_csv: str) -> None:
        delim = _detect_delimiter(utf8_csv, "utf-8")
        assert delim == ","

    def test_tab(self, tsv_file: str) -> None:
        delim = _detect_delimiter(tsv_file, "utf-8")
        assert delim == "\t"

    def test_semicolon(self, semicolon_csv: str) -> None:
        delim = _detect_delimiter(semicolon_csv, "utf-8")
        assert delim == ";"

    def test_pipe(self, pipe_delimited: str) -> None:
        delim = _detect_delimiter(pipe_delimited, "utf-8")
        assert delim == "|"

    def test_empty_file_raises(self, empty_file: str) -> None:
        with pytest.raises(DelimiterDetectionError):
            _detect_delimiter(empty_file, "utf-8")


# ---------------------------------------------------------------------------
# read_csv tests
# ---------------------------------------------------------------------------


class TestReadCSV:
    def test_basic_utf8(self, utf8_csv: str) -> None:
        ds = read_csv(utf8_csv)
        assert isinstance(ds, Dataset)
        assert ds.n_rows == 3
        assert ds.n_vars == 3
        assert list(ds.data.columns) == ["name", "age", "score"]
        assert ds.data["age"].tolist() == [25, 30, 35]

    def test_cp949_encoding(self, cp949_csv: str) -> None:
        ds = read_csv(cp949_csv, encoding="cp949")
        assert isinstance(ds, Dataset)
        assert ds.n_rows == 3
        assert list(ds.data.columns) == ["이름", "나이", "점수"]

    def test_auto_encoding_cp949(self, cp949_csv: str) -> None:
        ds = read_csv(cp949_csv, encoding="auto")
        assert isinstance(ds, Dataset)
        assert ds.n_rows == 3
        # Columns may be read correctly depending on chardet accuracy

    def test_tsv_with_auto_delimiter(self, tsv_file: str) -> None:
        ds = read_csv(tsv_file, delimiter="auto")
        assert isinstance(ds, Dataset)
        assert ds.n_vars == 3
        assert "gene" in ds.data.columns

    def test_tsv_explicit_tab(self, tsv_file: str) -> None:
        ds = read_csv(tsv_file, delimiter="\t")
        assert ds.n_vars == 3
        assert ds.data["expression"].tolist() == [12.5, 8.3, 15.7]

    def test_semicolon_delimiter(self, semicolon_csv: str) -> None:
        ds = read_csv(semicolon_csv, delimiter=";")
        assert ds.n_vars == 3
        assert list(ds.data.columns) == ["a", "b", "c"]

    def test_no_header(self, no_header_csv: str) -> None:
        ds = read_csv(no_header_csv, header=None)
        assert ds.n_vars == 3
        # pandas assigns integer column names when header=None, then sanitized to var_ prefix
        assert list(ds.data.columns) == ["var_0", "var_1", "var_2"]

    def test_dataset_has_variables(self, utf8_csv: str) -> None:
        ds = read_csv(utf8_csv)
        assert len(ds.variables) == 3
        for col in ds.data.columns:
            assert col in ds.variables
            assert ds.variables[col].name == col

    def test_source_info_populated(self, utf8_csv: str) -> None:
        ds = read_csv(utf8_csv)
        assert "encoding" in ds.source_info
        assert "delimiter" in ds.source_info
        assert "format" in ds.source_info
        assert ds.source_info["format"] == "csv"

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileReadError) as exc_info:
            read_csv("/nonexistent/file.csv")
        assert "존재하지 않습니다" in str(exc_info.value)

    def test_name_from_filename(self, utf8_csv: str) -> None:
        ds = read_csv(utf8_csv)
        assert ds.name == "utf8_data"


# ---------------------------------------------------------------------------
# read_txt tests
# ---------------------------------------------------------------------------


class TestReadTXT:
    def test_pipe_delimited_txt(self, pipe_delimited: str) -> None:
        ds = read_txt(pipe_delimited)
        assert isinstance(ds, Dataset)
        assert ds.n_vars == 3
        assert list(ds.data.columns) == ["col_a", "col_b", "col_c"]
        assert ds.source_info["format"] == "txt"

    def test_txt_with_auto_params(self, pipe_delimited: str) -> None:
        ds = read_txt(pipe_delimited, encoding="auto", delimiter="auto")
        assert ds.n_vars == 3

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileReadError):
            read_txt("/nonexistent/file.txt")


# ---------------------------------------------------------------------------
# Integration with fixture files
# ---------------------------------------------------------------------------


class TestFixtureFiles:
    def test_sample_clinical(self) -> None:
        path = Path(__file__).parent.parent / "fixtures" / "sample_clinical.csv"
        ds = read_csv(str(path))
        assert isinstance(ds, Dataset)
        assert ds.n_rows == 30
        assert ds.n_vars == 12
        assert "patient_id" in ds.data.columns
        assert "bmi" in ds.data.columns

    def test_sample_survey(self) -> None:
        path = Path(__file__).parent.parent / "fixtures" / "sample_survey.csv"
        ds = read_csv(str(path))
        assert isinstance(ds, Dataset)
        assert ds.n_rows == 25
        assert ds.n_vars == 11
        assert "respondent_id" in ds.data.columns

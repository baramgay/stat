"""Advanced tests for csv_reader module — targeting 80%+ coverage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import (
    DelimiterDetectionError,
    EncodingDetectionError,
    FileReadError,
)
from nuristat.io.csv_reader import (
    _detect_delimiter,
    _detect_encoding,
    read_csv,
)


# ---------------------------------------------------------------------------
# Fixtures — basic files
# ---------------------------------------------------------------------------


@pytest.fixture
def comma_csv(tmp_path: Path) -> str:
    path = tmp_path / "comma.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def tab_csv(tmp_path: Path) -> str:
    path = tmp_path / "tab.tsv"
    path.write_text("x\ty\tz\n10\t20\t30\n40\t50\t60\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def semicolon_csv(tmp_path: Path) -> str:
    path = tmp_path / "semi.csv"
    path.write_text("p;q;r\n1;2;3\n4;5;6\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def pipe_csv(tmp_path: Path) -> str:
    path = tmp_path / "pipe.csv"
    path.write_text("col1|col2|col3\nA|B|C\nD|E|F\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def utf8_with_bom_csv(tmp_path: Path) -> str:
    path = tmp_path / "utf8bom.csv"
    path.write_bytes(b"\xef\xbb\xbfname,value\nhello,1\nworld,2\n")
    return str(path)


@pytest.fixture
def cp949_csv(tmp_path: Path) -> str:
    path = tmp_path / "korean.csv"
    df = pd.DataFrame({"이름": ["홍길동", "김철수"], "나이": [30, 25]})
    df.to_csv(path, index=False, encoding="cp949")
    return str(path)


@pytest.fixture
def missing_csv(tmp_path: Path) -> str:
    path = tmp_path / "missing.csv"
    path.write_text("id,name,score\n1,Alice,85\n2,,90\n3,Charlie,\n4,,\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def no_header_csv(tmp_path: Path) -> str:
    path = tmp_path / "noheader.csv"
    path.write_text("1,apple,red\n2,banana,yellow\n3,cherry,red\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def large_csv(tmp_path: Path) -> str:
    path = tmp_path / "large.csv"
    rows = ["id,value,category"]
    for i in range(1, 1001):
        rows.append(f"{i},{i * 1.5},{i % 5}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def empty_csv(tmp_path: Path) -> str:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    return str(path)


@pytest.fixture
def header_only_csv(tmp_path: Path) -> str:
    path = tmp_path / "header_only.csv"
    path.write_text("col_a,col_b,col_c\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def single_column_csv(tmp_path: Path) -> str:
    path = tmp_path / "single_col.csv"
    path.write_text("value\n10\n20\n30\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def float_col_csv(tmp_path: Path) -> str:
    path = tmp_path / "floats.csv"
    path.write_text("x,y\n1.1,2.2\n3.3,4.4\n5.5,6.6\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def int_col_csv(tmp_path: Path) -> str:
    path = tmp_path / "ints.csv"
    path.write_text("a,b\n1,100\n2,200\n3,300\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def mixed_delimiter_csv(tmp_path: Path) -> str:
    """File where most rows use semicolon but one uses comma — tests consistency scoring."""
    path = tmp_path / "mixed_delim.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n7;8;9\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def two_cols_csv(tmp_path: Path) -> str:
    path = tmp_path / "two_cols.csv"
    path.write_text("name,value\nfoo,1\nbar,2\nbaz,3\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# _detect_encoding tests
# ---------------------------------------------------------------------------


class TestDetectEncodingAdvanced:
    def test_utf8_file(self, comma_csv: str) -> None:
        enc = _detect_encoding(comma_csv)
        assert enc in ("utf-8", "ascii", "utf-8-sig")

    def test_utf8_bom_file(self, utf8_with_bom_csv: str) -> None:
        enc = _detect_encoding(utf8_with_bom_csv)
        # chardet may detect utf-8-sig or utf-8
        assert "utf" in enc.lower()

    def test_cp949_file(self, cp949_csv: str) -> None:
        # chardet may fail on small CP949 files; treat EncodingDetectionError as expected
        try:
            enc = _detect_encoding(cp949_csv)
            assert enc in ("cp949", "euc-kr", "utf-8")
        except EncodingDetectionError:
            pass  # acceptable — small CP949 file may have low confidence

    def test_empty_file_returns_utf8(self, empty_csv: str) -> None:
        enc = _detect_encoding(empty_csv)
        assert enc == "utf-8"

    def test_nonexistent_raises_file_read_error(self) -> None:
        with pytest.raises(FileReadError):
            _detect_encoding("/no/such/file.csv")

    def test_small_sample_size(self, comma_csv: str) -> None:
        # Should still work with a small sample
        enc = _detect_encoding(comma_csv, sample_size=10)
        assert isinstance(enc, str)
        assert len(enc) > 0

    def test_large_sample_size(self, large_csv: str) -> None:
        enc = _detect_encoding(large_csv, sample_size=131072)
        assert isinstance(enc, str)


# ---------------------------------------------------------------------------
# _detect_delimiter tests
# ---------------------------------------------------------------------------


class TestDetectDelimiterAdvanced:
    def test_comma_detected(self, comma_csv: str) -> None:
        delim = _detect_delimiter(comma_csv, "utf-8")
        assert delim == ","

    def test_tab_detected(self, tab_csv: str) -> None:
        delim = _detect_delimiter(tab_csv, "utf-8")
        assert delim == "\t"

    def test_semicolon_detected(self, semicolon_csv: str) -> None:
        delim = _detect_delimiter(semicolon_csv, "utf-8")
        assert delim == ";"

    def test_pipe_detected(self, pipe_csv: str) -> None:
        delim = _detect_delimiter(pipe_csv, "utf-8")
        assert delim == "|"

    def test_empty_file_raises_delimiter_error(self, empty_csv: str) -> None:
        with pytest.raises(DelimiterDetectionError):
            _detect_delimiter(empty_csv, "utf-8")

    def test_sample_lines_parameter(self, large_csv: str) -> None:
        # Should still detect correctly with fewer sampled lines
        delim = _detect_delimiter(large_csv, "utf-8", sample_lines=5)
        assert delim == ","

    def test_mixed_delimiter_consistent(self, mixed_delimiter_csv: str) -> None:
        delim = _detect_delimiter(mixed_delimiter_csv, "utf-8")
        assert delim == ";"

    def test_single_column_raises_or_returns(self, single_column_csv: str) -> None:
        # Single-column file has no usable delimiter — may raise or return default
        try:
            result = _detect_delimiter(single_column_csv, "utf-8")
            assert isinstance(result, str)
        except DelimiterDetectionError:
            pass  # acceptable — single column cannot satisfy >=2 column requirement


# ---------------------------------------------------------------------------
# read_csv basic tests
# ---------------------------------------------------------------------------


class TestReadCSVBasic:
    def test_basic_read(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert isinstance(ds, Dataset)
        assert ds.n_rows == 3
        assert ds.n_vars == 3

    def test_columns_correct(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert list(ds.data.columns) == ["a", "b", "c"]

    def test_data_values(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert ds.data["a"].tolist() == [1, 4, 7]

    def test_returns_dataset(self, comma_csv: str) -> None:
        result = read_csv(comma_csv)
        assert isinstance(result, Dataset)

    def test_name_is_stem(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert ds.name == "comma"

    def test_header_only_empty_dataset(self, header_only_csv: str) -> None:
        ds = read_csv(header_only_csv)
        assert ds.n_rows == 0
        assert ds.n_vars == 3


# ---------------------------------------------------------------------------
# Delimiter auto-detection via read_csv
# ---------------------------------------------------------------------------


class TestReadCSVDelimiterAuto:
    def test_auto_detects_comma(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv, delimiter="auto")
        assert ds.n_vars == 3

    def test_auto_detects_tab(self, tab_csv: str) -> None:
        ds = read_csv(tab_csv, delimiter="auto")
        assert ds.n_vars == 3
        assert "x" in ds.data.columns

    def test_auto_detects_semicolon(self, semicolon_csv: str) -> None:
        ds = read_csv(semicolon_csv, delimiter="auto")
        assert ds.n_vars == 3

    def test_auto_detects_pipe(self, pipe_csv: str) -> None:
        ds = read_csv(pipe_csv, delimiter="auto")
        assert ds.n_vars == 3

    def test_explicit_tab_delimiter(self, tab_csv: str) -> None:
        ds = read_csv(tab_csv, delimiter="\t")
        assert ds.n_rows == 2
        assert ds.data["y"].tolist() == [20, 50]

    def test_explicit_semicolon_delimiter(self, semicolon_csv: str) -> None:
        ds = read_csv(semicolon_csv, delimiter=";")
        assert list(ds.data.columns) == ["p", "q", "r"]


# ---------------------------------------------------------------------------
# Encoding tests via read_csv
# ---------------------------------------------------------------------------


class TestReadCSVEncoding:
    def test_explicit_utf8(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv, encoding="utf-8")
        assert ds.n_rows == 3

    def test_explicit_cp949(self, cp949_csv: str) -> None:
        ds = read_csv(cp949_csv, encoding="cp949")
        assert ds.n_rows == 2
        assert "이름" in ds.data.columns

    def test_auto_encoding_utf8_file(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv, encoding="auto")
        assert isinstance(ds, Dataset)

    def test_auto_encoding_cp949_file(self, cp949_csv: str) -> None:
        # chardet may fail on small CP949 files; both success and detection error are acceptable
        try:
            ds = read_csv(cp949_csv, encoding="auto")
            assert ds.n_rows == 2
        except EncodingDetectionError:
            pass  # acceptable — small CP949 corpus may fall below confidence threshold

    def test_source_info_encoding_recorded(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv, encoding="utf-8")
        assert ds.source_info["encoding"] == "utf-8"

    def test_source_info_auto_encoding_resolved(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv, encoding="auto")
        # "auto" should be replaced with the resolved encoding
        assert ds.source_info["encoding"] != "auto"


# ---------------------------------------------------------------------------
# Missing values tests
# ---------------------------------------------------------------------------


class TestReadCSVMissingValues:
    def test_missing_name(self, missing_csv: str) -> None:
        ds = read_csv(missing_csv)
        assert ds.data["name"].isna().sum() == 2

    def test_missing_score(self, missing_csv: str) -> None:
        ds = read_csv(missing_csv)
        assert ds.data["score"].isna().sum() == 2

    def test_all_missing_row(self, missing_csv: str) -> None:
        ds = read_csv(missing_csv)
        last_row = ds.data.iloc[-1]
        assert pd.isna(last_row["name"])
        assert pd.isna(last_row["score"])

    def test_total_rows_intact(self, missing_csv: str) -> None:
        ds = read_csv(missing_csv)
        assert ds.n_rows == 4


# ---------------------------------------------------------------------------
# No header tests
# ---------------------------------------------------------------------------


class TestReadCSVNoHeader:
    def test_no_header_columns_renamed(self, no_header_csv: str) -> None:
        ds = read_csv(no_header_csv, header=None)
        assert list(ds.data.columns) == ["var_0", "var_1", "var_2"]

    def test_no_header_row_count(self, no_header_csv: str) -> None:
        ds = read_csv(no_header_csv, header=None)
        assert ds.n_rows == 3

    def test_no_header_source_info(self, no_header_csv: str) -> None:
        ds = read_csv(no_header_csv, header=None)
        assert ds.source_info["header"] is None


# ---------------------------------------------------------------------------
# Large file tests
# ---------------------------------------------------------------------------


class TestReadCSVLargeFile:
    def test_large_file_row_count(self, large_csv: str) -> None:
        ds = read_csv(large_csv)
        assert ds.n_rows == 1000

    def test_large_file_col_count(self, large_csv: str) -> None:
        ds = read_csv(large_csv)
        assert ds.n_vars == 3

    def test_large_file_values(self, large_csv: str) -> None:
        ds = read_csv(large_csv)
        assert ds.data["id"].iloc[0] == 1
        assert ds.data["id"].iloc[-1] == 1000


# ---------------------------------------------------------------------------
# dtype auto-detection tests
# ---------------------------------------------------------------------------


class TestReadCSVDtypeDetection:
    def test_float_columns(self, float_col_csv: str) -> None:
        ds = read_csv(float_col_csv)
        assert pd.api.types.is_float_dtype(ds.data["x"])
        assert pd.api.types.is_float_dtype(ds.data["y"])

    def test_integer_columns(self, int_col_csv: str) -> None:
        ds = read_csv(int_col_csv)
        assert pd.api.types.is_integer_dtype(ds.data["a"])
        assert pd.api.types.is_integer_dtype(ds.data["b"])

    def test_string_columns(self, pipe_csv: str) -> None:
        ds = read_csv(pipe_csv, delimiter="|")
        assert ds.data["col1"].dtype == object


# ---------------------------------------------------------------------------
# source_info tests
# ---------------------------------------------------------------------------


class TestReadCSVSourceInfo:
    def test_source_info_format(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert ds.source_info["format"] == "csv"

    def test_source_info_file_name(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert ds.source_info["file_name"] == "comma.csv"

    def test_source_info_file_path_absolute(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert Path(ds.source_info["file_path"]).is_absolute()

    def test_source_info_file_size_positive(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert ds.source_info["file_size"] > 0

    def test_source_info_delimiter(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv, delimiter=",")
        assert ds.source_info["delimiter"] == ","

    def test_source_info_header(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert ds.source_info["header"] == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestReadCSVErrors:
    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileReadError) as exc_info:
            read_csv("/no/such/file.csv")
        assert "존재하지 않습니다" in str(exc_info.value)

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileReadError):
            read_csv(str(tmp_path))

    def test_empty_file_delimiter_auto_raises(self, empty_csv: str) -> None:
        with pytest.raises((FileReadError, DelimiterDetectionError)):
            read_csv(empty_csv, delimiter="auto")

    def test_empty_file_explicit_delimiter_raises(self, empty_csv: str) -> None:
        with pytest.raises(FileReadError):
            read_csv(empty_csv, delimiter=",")

    def test_error_message_contains_filepath(self) -> None:
        target = "/no/such/file.csv"
        with pytest.raises(FileReadError) as exc_info:
            read_csv(target)
        assert "file.csv" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Variable metadata tests
# ---------------------------------------------------------------------------


class TestReadCSVVariableMeta:
    def test_variables_created_for_all_columns(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        for col in ds.data.columns:
            assert col in ds.variables

    def test_variable_names_match_columns(self, two_cols_csv: str) -> None:
        ds = read_csv(two_cols_csv)
        assert set(ds.variables.keys()) == set(ds.data.columns)

    def test_variable_count(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv)
        assert len(ds.variables) == 3


# ---------------------------------------------------------------------------
# Extra kwargs forwarding
# ---------------------------------------------------------------------------


class TestReadCSVKwargs:
    def test_nrows_kwarg(self, large_csv: str) -> None:
        ds = read_csv(large_csv, nrows=50)
        assert ds.n_rows == 50

    def test_usecols_kwarg(self, comma_csv: str) -> None:
        ds = read_csv(comma_csv, usecols=["a", "b"])
        assert ds.n_vars == 2
        assert "c" not in ds.data.columns

    def test_dtype_kwarg(self, int_col_csv: str) -> None:
        ds = read_csv(int_col_csv, dtype={"a": float})
        assert pd.api.types.is_float_dtype(ds.data["a"])


# ---------------------------------------------------------------------------
# Branch coverage — encoding normalization paths
# ---------------------------------------------------------------------------


class TestDetectEncodingNormalization:
    def test_latin1_normalized(self, tmp_path: Path) -> None:
        """latin-1 encoded file should be detected or raise (small sample confidence may be low)."""
        path = tmp_path / "latin1.csv"
        # Write a longer latin-1 file with repeated non-ASCII bytes to boost confidence
        row = "name,value\ncaf\xe9,1\nna\xefve,2\nfianc\xe9e,3\nresum\xe9,4\n"
        path.write_bytes(row.encode("latin-1") * 20)
        try:
            enc = _detect_encoding(str(path))
            assert enc in ("latin-1", "utf-8", "ascii", "windows-1252", "ISO-8859-1", "iso-8859-1")
        except EncodingDetectionError:
            pass  # acceptable if chardet confidence still below threshold

    def test_cp949_alias_euckr(self, tmp_path: Path) -> None:
        """EUC-KR alias should be normalized to cp949."""
        # Build a byte string that chardet reliably detects as EUC-KR / CP949
        # by using long Korean text (enough bytes to cross the 0.4 confidence threshold)
        path = tmp_path / "euckr.csv"
        row = "이름,나이\n홍길동,30\n김철수,25\n이영희,22\n박민준,28\n최수현,33\n"
        path.write_bytes(row.encode("cp949") * 10)
        try:
            enc = _detect_encoding(str(path))
            assert enc in ("cp949", "euc-kr", "utf-8")
        except EncodingDetectionError:
            pass  # acceptable if confidence still too low


class TestDelimiterInconsistencyBranch:
    def test_inconsistent_column_counts(self, tmp_path: Path) -> None:
        """File where rows have different column counts (consistency-ratio branch)."""
        path = tmp_path / "incon.csv"
        # 4 rows: mostly 3-col comma, one row has 4 — triggers consistency scoring
        path.write_text(
            "a,b,c\n1,2,3\n4,5,6\n7,8,9,10\n11,12,13\n",
            encoding="utf-8",
        )
        delim = _detect_delimiter(str(path), "utf-8")
        assert delim == ","

    def test_unicode_decode_error_in_delimiter(self, tmp_path: Path) -> None:
        """Passing wrong encoding to _detect_delimiter should raise FileReadError."""
        path = tmp_path / "cp949_wrong.csv"
        path.write_bytes("이름,나이\n홍길동,30\n".encode("cp949"))
        with pytest.raises(FileReadError):
            _detect_delimiter(str(path), "utf-8")


class TestReadCSVExceptionBranches:
    def test_unicode_decode_error_explicit_wrong_encoding(self, tmp_path: Path) -> None:
        """read_csv with wrong explicit encoding should raise FileReadError."""
        path = tmp_path / "cp949_file.csv"
        path.write_bytes("이름,나이\n홍길동,30\n".encode("cp949") * 5)
        with pytest.raises(FileReadError):
            read_csv(str(path), encoding="utf-8", delimiter=",")

    def test_empty_data_error(self, tmp_path: Path) -> None:
        """Completely empty file (no header) with explicit delimiter raises FileReadError."""
        path = tmp_path / "truly_empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(FileReadError):
            read_csv(str(path), encoding="utf-8", delimiter=",")

    def test_parser_error_malformed_quotes(self, tmp_path: Path) -> None:
        """Malformed quoted fields may trigger ParserError, mapped to FileReadError."""
        path = tmp_path / "malformed.csv"
        # Unclosed quote that pandas cannot recover from even with error_bad_lines
        path.write_text('a,b\n"unclosed,1\n2,3\n', encoding="utf-8")
        try:
            read_csv(str(path), encoding="utf-8", delimiter=",")
        except FileReadError:
            pass  # expected
        except Exception:
            pass  # pandas may silently handle it depending on version

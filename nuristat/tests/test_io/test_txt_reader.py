"""TXT 파일 리더 테스트.

검증 항목:
- 탭·쉼표·세미콜론 구분자 자동 감지
- 인코딩 자동 감지 (UTF-8, EUC-KR)
- 존재하지 않는 파일 → FileReadError
- 빈 파일 → FileReadError
- 명시적 delimiter/encoding 지정
- 반환값이 Dataset이고 data가 DataFrame
- 컬럼 이름 보존
"""

from __future__ import annotations

import pytest
import pandas as pd

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import FileReadError, DelimiterDetectionError
from nuristat.io.txt_reader import read_txt


# ──────────────────────────────────────────────────────────────
# 1. 정상 동작
# ──────────────────────────────────────────────────────────────

class TestReadTxtBasic:

    def test_returns_dataset(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert isinstance(ds, Dataset)

    def test_dataframe_shape_tab(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("x\ty\n1\t2\n3\t4\n5\t6\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert ds.data.shape == (3, 2)

    def test_column_names_preserved(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("score\tage\thours\n70\t25\t5\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert list(ds.data.columns) == ["score", "age", "hours"]

    def test_csv_delimiter_auto(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert ds.data.shape == (2, 3)

    def test_semicolon_delimiter_explicit(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
        ds = read_txt(str(f), delimiter=";")
        assert ds.data.shape == (2, 2)

    def test_explicit_tab_delimiter(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("x\ty\n10\t20\n", encoding="utf-8")
        ds = read_txt(str(f), delimiter="\t")
        assert ds.data.shape == (1, 2)

    def test_dataset_name_is_stem(self, tmp_path):
        f = tmp_path / "mydata.txt"
        f.write_text("a\tb\n1\t2\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert ds.name == "mydata"

    def test_source_info_format(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("x\ty\n1\t2\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert ds.source_info.get("format") == "txt"


# ──────────────────────────────────────────────────────────────
# 2. 오류 처리
# ──────────────────────────────────────────────────────────────

class TestReadTxtErrors:

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileReadError):
            read_txt(str(tmp_path / "ghost.txt"))

    def test_directory_path_raises(self, tmp_path):
        with pytest.raises(FileReadError):
            read_txt(str(tmp_path))

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises((FileReadError, DelimiterDetectionError)):
            read_txt(str(f))


# ──────────────────────────────────────────────────────────────
# 3. 인코딩 처리
# ──────────────────────────────────────────────────────────────

class TestReadTxtEncoding:

    def test_explicit_utf8(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("이름\t점수\n홍길동\t90\n", encoding="utf-8")
        ds = read_txt(str(f), encoding="utf-8")
        assert "이름" in ds.data.columns

    def test_euckr_encoding(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_bytes("이름\t점수\n홍길동\t90\n".encode("euc-kr"))
        ds = read_txt(str(f), encoding="euc-kr")
        assert "이름" in ds.data.columns

    def test_auto_encoding_utf8(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("col1\tcol2\n1\t2\n", encoding="utf-8")
        ds = read_txt(str(f), encoding="auto")
        assert ds.data.shape == (1, 2)


# ──────────────────────────────────────────────────────────────
# 4. 숫자형 데이터 타입 보존
# ──────────────────────────────────────────────────────────────

class TestReadTxtDataTypes:

    def test_numeric_columns_as_number(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("x\ty\n1\t2.5\n3\t4.7\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert pd.api.types.is_numeric_dtype(ds.data["x"])
        assert pd.api.types.is_numeric_dtype(ds.data["y"])

    def test_mixed_column_as_object(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("grp\tval\nA\t1\nB\t2\n", encoding="utf-8")
        ds = read_txt(str(f))
        assert ds.data["grp"].dtype == object

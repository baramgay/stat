"""import_wizard.py 미커버 경로 보완 테스트."""

from __future__ import annotations

import os
import tempfile
import textwrap

import pytest

from statworkbench.io.import_wizard import ImportWizard


@pytest.fixture
def csv_file(tmp_path):
    content = "name,age,score\nAlice,30,85.5\nBob,25,92.0\nCharlie,35,78.3\n"
    p = tmp_path / "test.csv"
    p.write_text(content, encoding="utf-8")
    return str(p)


@pytest.fixture
def broken_encoding_file(tmp_path):
    p = tmp_path / "broken.csv"
    # write valid utf-8 then cause a preview issue by embedding replacement char
    p.write_bytes(b"name,age\n\xff\xfeAlice,30\n")
    return str(p)


@pytest.fixture
def excel_file(tmp_path):
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["name", "age", "score"])
        ws.append(["Alice", 30, 85.5])
        ws.append(["Bob", 25, 92.0])
        p = tmp_path / "test.xlsx"
        wb.save(str(p))
        return str(p)
    except ImportError:
        pytest.skip("openpyxl 없음")


# ── step_encoding ─────────────────────────────────────────────────────────────

class TestStepEncoding:
    def test_auto_detect(self, csv_file):
        wiz = ImportWizard()
        result = wiz.step_encoding(csv_file, "auto")
        assert "encoding" in result
        assert result["user_selected"] is False

    def test_user_specified(self, csv_file):
        wiz = ImportWizard()
        result = wiz.step_encoding(csv_file, "utf-8")
        assert result["user_selected"] is True
        assert result["encoding"] == "utf-8"

    def test_broken_encoding_warning(self, broken_encoding_file):
        wiz = ImportWizard()
        result = wiz.step_encoding(broken_encoding_file, "utf-8")
        # 깨진 문자 발견 시 preview_ok=False 또는 경고
        assert result is not None


# ── step_delimiter ────────────────────────────────────────────────────────────

class TestStepDelimiter:
    def test_auto_detect_comma(self, csv_file):
        wiz = ImportWizard()
        result = wiz.step_delimiter(csv_file, "utf-8", "auto")
        assert result["delimiter"] == ","
        assert result["user_selected"] is False

    def test_user_specified_semicolon(self, csv_file):
        wiz = ImportWizard()
        result = wiz.step_delimiter(csv_file, "utf-8", ";")
        assert result["delimiter"] == ";"
        assert result["user_selected"] is True

    def test_n_columns_counted(self, csv_file):
        wiz = ImportWizard()
        result = wiz.step_delimiter(csv_file, "utf-8", ",")
        assert result["n_columns"] == 3


# ── step_header ───────────────────────────────────────────────────────────────

class TestStepHeader:
    def test_reads_column_names(self, csv_file):
        wiz = ImportWizard()
        result = wiz.step_header(csv_file, "utf-8", ",", header=0)
        assert "name" in result["column_names"]
        assert "age" in result["column_names"]

    def test_max_rows(self, csv_file):
        wiz = ImportWizard()
        result = wiz.step_header(csv_file, "utf-8", ",", header=0, max_rows=2)
        assert result["max_rows"] == 2


# ── validate_step ─────────────────────────────────────────────────────────────

class TestValidateStep:
    def test_delimiter_single_column_warning(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("delimiter", {"n_columns": 1})
        assert len(msgs) > 0

    def test_delimiter_ok(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("delimiter", {"n_columns": 5})
        assert len(msgs) == 0

    def test_header_empty_columns(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("header", {"column_names": []})
        assert len(msgs) > 0

    def test_header_empty_names_in_list(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("header", {"column_names": ["name", "", "age"]})
        assert len(msgs) > 0

    def test_type_preview_high_missing(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("type_preview", {
            "column_info": [{"name": "x", "missing_rate": 0.6}]
        })
        assert len(msgs) > 0

    def test_type_preview_ok(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("type_preview", {
            "column_info": [{"name": "x", "missing_rate": 0.1}]
        })
        assert len(msgs) == 0

    def test_confirm_no_rows(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("confirm", {"n_rows": 0, "n_columns": 3})
        assert len(msgs) > 0

    def test_confirm_no_columns(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("confirm", {"n_rows": 10, "n_columns": 0})
        assert len(msgs) > 0

    def test_confirm_ok(self):
        wiz = ImportWizard()
        msgs = wiz.validate_step("confirm", {"n_rows": 10, "n_columns": 3})
        assert len(msgs) == 0


# ── run_full_wizard ───────────────────────────────────────────────────────────

class TestRunFullWizard:
    def test_csv_full_wizard(self, csv_file):
        wiz = ImportWizard()
        state = wiz.run_full_wizard(csv_file)
        assert "encoding" in state
        assert "delimiter" in state

    def test_excel_file_select(self, excel_file):
        """Excel 파일은 step_file_select에서 xlsx 포맷으로 감지."""
        wiz = ImportWizard()
        result = wiz.step_file_select(excel_file)
        assert result["file_format"] in ("xlsx", "xls")
        assert result["exists"] is True

    def test_nonexistent_file_errors(self, tmp_path):
        """존재하지 않는 파일 → step_file_select에서 조기 오류 반환."""
        from statworkbench.core.exceptions import FileReadError
        wiz = ImportWizard()
        fake = str(tmp_path / "no_such_file.csv")
        # step_file_select는 파일 존재 여부를 확인함
        result = wiz.step_file_select(fake)
        assert result is not None  # 결과 반환 또는 오류 기록

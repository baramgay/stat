"""ImportWizard 테스트.

검증 항목:
- step_file_select: 파일 존재 여부·형식·크기
- step_encoding: 인코딩 감지, preview_ok
- step_delimiter: 구분자 감지, n_columns
- step_header: 컬럼명 추출
- step_type_preview: 컬럼 정보, 결측치 경고
- step_confirm: 요약 정보, ready 플래그
- validate_step: 각 단계 유효성 검사 메시지
- reset: 상태 초기화
- state / warnings / errors 프로퍼티
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.io.import_wizard import ImportWizard


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def wizard() -> ImportWizard:
    return ImportWizard()


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,score,age\n홍길동,90,25\n김철수,80,30\n이영희,95,22\n", encoding="utf-8")
    return f


@pytest.fixture
def tab_file(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("x\ty\tz\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
    return f


@pytest.fixture
def simple_dataset() -> Dataset:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "score": rng.normal(70, 15, 50),
        "group": ["A", "B"] * 25,
    })
    return Dataset(data=df, name="test")


# ──────────────────────────────────────────────────────────────
# 1. 초기화 및 상태 관리
# ──────────────────────────────────────────────────────────────

class TestImportWizardInit:

    def test_instantiation(self, wizard):
        assert isinstance(wizard, ImportWizard)

    def test_initial_state_empty(self, wizard):
        assert wizard.state == {}

    def test_initial_warnings_empty(self, wizard):
        assert wizard.warnings == []

    def test_initial_errors_empty(self, wizard):
        assert wizard.errors == []

    def test_reset_clears_state(self, wizard, csv_file):
        wizard.step_file_select(str(csv_file))
        wizard.reset()
        assert wizard.state == {}
        assert wizard.warnings == []
        assert wizard.errors == []

    def test_state_returns_copy(self, wizard, csv_file):
        wizard.step_file_select(str(csv_file))
        s = wizard.state
        s["injected"] = True
        assert "injected" not in wizard.state

    def test_encodings_defined(self, wizard):
        assert hasattr(wizard, "ENCODINGS")
        assert "utf-8" in wizard.ENCODINGS
        assert "auto" in wizard.ENCODINGS

    def test_delimiters_defined(self, wizard):
        assert hasattr(wizard, "DELIMITERS")
        assert len(wizard.DELIMITERS) >= 4


# ──────────────────────────────────────────────────────────────
# 2. Step 1: 파일 선택
# ──────────────────────────────────────────────────────────────

class TestStepFileSelect:

    def test_existing_file_exists_true(self, wizard, csv_file):
        result = wizard.step_file_select(str(csv_file))
        assert result["exists"] is True

    def test_nonexistent_file_exists_false(self, wizard, tmp_path):
        result = wizard.step_file_select(str(tmp_path / "ghost.csv"))
        assert result["exists"] is False

    def test_file_format_csv(self, wizard, csv_file):
        result = wizard.step_file_select(str(csv_file))
        assert result["file_format"] == "csv"

    def test_file_format_txt(self, wizard, tab_file):
        result = wizard.step_file_select(str(tab_file))
        assert result["file_format"] == "txt"

    def test_file_size_positive(self, wizard, csv_file):
        result = wizard.step_file_select(str(csv_file))
        assert result["file_size"] > 0

    def test_filename_included(self, wizard, csv_file):
        result = wizard.step_file_select(str(csv_file))
        assert result["filename"] == csv_file.name

    def test_state_updated(self, wizard, csv_file):
        wizard.step_file_select(str(csv_file))
        assert "file" in wizard.state


# ──────────────────────────────────────────────────────────────
# 3. Step 2: 인코딩
# ──────────────────────────────────────────────────────────────

class TestStepEncoding:

    def test_explicit_utf8(self, wizard, csv_file):
        result = wizard.step_encoding(str(csv_file), encoding="utf-8")
        assert result["encoding"] == "utf-8"

    def test_auto_detection(self, wizard, csv_file):
        result = wizard.step_encoding(str(csv_file), encoding="auto")
        assert result["encoding"] != ""
        assert result["preview_ok"] is True

    def test_user_selected_false_for_auto(self, wizard, csv_file):
        result = wizard.step_encoding(str(csv_file), encoding="auto")
        assert result["user_selected"] is False

    def test_user_selected_true_for_explicit(self, wizard, csv_file):
        result = wizard.step_encoding(str(csv_file), encoding="utf-8")
        assert result["user_selected"] is True

    def test_preview_ok_for_valid_encoding(self, wizard, csv_file):
        result = wizard.step_encoding(str(csv_file), encoding="utf-8")
        assert result["preview_ok"] is True

    def test_state_updated(self, wizard, csv_file):
        wizard.step_encoding(str(csv_file), encoding="utf-8")
        assert "encoding" in wizard.state


# ──────────────────────────────────────────────────────────────
# 4. Step 3: 구분자
# ──────────────────────────────────────────────────────────────

class TestStepDelimiter:

    def test_explicit_comma(self, wizard, csv_file):
        result = wizard.step_delimiter(str(csv_file), "utf-8", delimiter=",")
        assert result["delimiter"] == ","

    def test_auto_detects_comma(self, wizard, csv_file):
        result = wizard.step_delimiter(str(csv_file), "utf-8", delimiter="auto")
        assert result["delimiter"] == ","

    def test_explicit_tab(self, wizard, tab_file):
        result = wizard.step_delimiter(str(tab_file), "utf-8", delimiter="\t")
        assert result["delimiter"] == "\t"

    def test_n_columns_csv(self, wizard, csv_file):
        result = wizard.step_delimiter(str(csv_file), "utf-8", delimiter=",")
        assert result["n_columns"] == 3

    def test_state_updated(self, wizard, csv_file):
        wizard.step_delimiter(str(csv_file), "utf-8", delimiter=",")
        assert "delimiter" in wizard.state


# ──────────────────────────────────────────────────────────────
# 5. Step 4: 헤더
# ──────────────────────────────────────────────────────────────

class TestStepHeader:

    def test_column_names_extracted(self, wizard, csv_file):
        result = wizard.step_header(str(csv_file), "utf-8", ",", header=0)
        assert "name" in result["column_names"]
        assert "score" in result["column_names"]
        assert "age" in result["column_names"]

    def test_n_columns_matches(self, wizard, csv_file):
        result = wizard.step_header(str(csv_file), "utf-8", ",", header=0)
        assert result["n_columns"] == 3

    def test_state_updated(self, wizard, csv_file):
        wizard.step_header(str(csv_file), "utf-8", ",", header=0)
        assert "header" in wizard.state


# ──────────────────────────────────────────────────────────────
# 6. Step 5: 타입 미리보기
# ──────────────────────────────────────────────────────────────

class TestStepTypePreview:

    def test_column_info_length(self, wizard, simple_dataset):
        result = wizard.step_type_preview(simple_dataset)
        assert len(result["column_info"]) == len(simple_dataset.data.columns)

    def test_column_info_keys(self, wizard, simple_dataset):
        result = wizard.step_type_preview(simple_dataset)
        col = result["column_info"][0]
        for key in ["name", "storage_type", "measure_type", "n_unique", "missing_count", "missing_rate"]:
            assert key in col

    def test_high_missing_generates_warning(self, wizard):
        df = pd.DataFrame({"x": [1.0] + [np.nan] * 9})
        ds = Dataset(data=df, name="t")
        result = wizard.step_type_preview(ds)
        assert len(result["warnings"]) > 0

    def test_no_warning_for_clean_data(self, wizard, simple_dataset):
        result = wizard.step_type_preview(simple_dataset)
        assert len(result["warnings"]) == 0

    def test_state_updated(self, wizard, simple_dataset):
        wizard.step_type_preview(simple_dataset)
        assert "type_preview" in wizard.state


# ──────────────────────────────────────────────────────────────
# 7. Step 6: 최종 확인
# ──────────────────────────────────────────────────────────────

class TestStepConfirm:

    def test_ready_true_for_valid_data(self, wizard, simple_dataset):
        result = wizard.step_confirm(simple_dataset)
        assert result["ready"] is True

    def test_n_rows_matches(self, wizard, simple_dataset):
        result = wizard.step_confirm(simple_dataset)
        assert result["n_rows"] == len(simple_dataset.data)

    def test_n_columns_matches(self, wizard, simple_dataset):
        result = wizard.step_confirm(simple_dataset)
        assert result["n_columns"] == simple_dataset.n_vars

    def test_state_updated(self, wizard, simple_dataset):
        wizard.step_confirm(simple_dataset)
        assert "confirm" in wizard.state


# ──────────────────────────────────────────────────────────────
# 8. validate_step
# ──────────────────────────────────────────────────────────────

class TestValidateStep:

    def test_file_not_exists_returns_message(self, wizard):
        msgs = wizard.validate_step("file_select", {"exists": False, "file_size": 0, "file_format": "csv"})
        assert len(msgs) > 0

    def test_file_exists_valid(self, wizard):
        msgs = wizard.validate_step("file_select", {"exists": True, "file_size": 100, "file_format": "csv"})
        assert len(msgs) == 0

    def test_unsupported_format_returns_message(self, wizard):
        msgs = wizard.validate_step("file_select", {"exists": True, "file_size": 100, "file_format": "pdf"})
        assert any("형식" in m for m in msgs)

    def test_bad_encoding_preview_returns_message(self, wizard):
        msgs = wizard.validate_step("encoding", {"preview_ok": False})
        assert len(msgs) > 0

    def test_good_encoding_preview_ok(self, wizard):
        msgs = wizard.validate_step("encoding", {"preview_ok": True})
        assert len(msgs) == 0

    def test_one_column_delimiter_warning(self, wizard):
        msgs = wizard.validate_step("delimiter", {"n_columns": 1})
        assert len(msgs) > 0

    def test_multi_column_delimiter_ok(self, wizard):
        msgs = wizard.validate_step("delimiter", {"n_columns": 3})
        assert len(msgs) == 0

    def test_empty_column_names_header_warning(self, wizard):
        msgs = wizard.validate_step("header", {"column_names": []})
        assert len(msgs) > 0

    def test_no_rows_confirm_warning(self, wizard):
        msgs = wizard.validate_step("confirm", {"n_rows": 0, "n_columns": 3})
        assert len(msgs) > 0

    def test_valid_confirm_no_messages(self, wizard):
        msgs = wizard.validate_step("confirm", {"n_rows": 100, "n_columns": 5})
        assert len(msgs) == 0

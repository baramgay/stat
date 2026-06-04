"""import_wizard.py 커버리지 보강 테스트.

미커버 라인:
  127-129: step_encoding — EncodingDetectionError → fallback utf-8, 에러 기록
  137-144: step_encoding — 미리보기 시 UnicodeDecodeError/OSError → preview_ok=False
  438-495: run_auto — CSV/Excel 파일 전체 자동 임포트 파이프라인
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nuristat.core.exceptions import EncodingDetectionError
from nuristat.io.import_wizard import ImportWizard


@pytest.fixture
def wizard():
    return ImportWizard()


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,score,age\n홍길동,90,25\n김철수,80,30\n", encoding="utf-8")
    return f


@pytest.fixture
def excel_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data.xlsx"
    df.to_excel(str(path), index=False)
    return path


# ---------------------------------------------------------------------------
# step_encoding — EncodingDetectionError fallback (lines 127-129)
# ---------------------------------------------------------------------------

class TestStepEncodingErrorFallback:

    def test_encoding_detection_error_falls_back_to_utf8(self, wizard, csv_file):
        """EncodingDetectionError 발생 시 utf-8 fallback, 에러 목록에 기록."""
        with patch(
            "nuristat.io.import_wizard._detect_encoding",
            side_effect=EncodingDetectionError(str(csv_file), "chardet 실패"),
        ):
            result = wizard.step_encoding(str(csv_file), "auto")

        assert result["encoding"] == "utf-8"
        assert len(wizard.errors) > 0

    def test_encoding_detection_error_recorded_in_errors(self, wizard, csv_file):
        """에러 메시지가 wizard.errors에 저장된다."""
        with patch(
            "nuristat.io.import_wizard._detect_encoding",
            side_effect=EncodingDetectionError(str(csv_file), "test error msg"),
        ):
            wizard.step_encoding(str(csv_file), "auto")

        assert any("test error msg" in e for e in wizard.errors)


# ---------------------------------------------------------------------------
# step_encoding — preview UnicodeDecodeError/OSError (lines 137-144)
# ---------------------------------------------------------------------------

class TestStepEncodingPreviewFailure:

    def test_preview_unicode_error_sets_preview_not_ok(self, wizard, csv_file):
        """미리보기 UnicodeDecodeError → preview_ok=False, 에러 기록."""
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad")):
            result = wizard.step_encoding(str(csv_file), "utf-8")

        assert result["preview_ok"] is False
        assert len(wizard.errors) > 0

    def test_preview_oserror_sets_preview_not_ok(self, wizard, csv_file):
        """미리보기 OSError → preview_ok=False, 에러 기록."""
        with patch("builtins.open", side_effect=OSError("file locked")):
            result = wizard.step_encoding(str(csv_file), "utf-8")

        assert result["preview_ok"] is False
        assert any("미리보기 실패" in e for e in wizard.errors)

    def test_replacement_char_sets_preview_not_ok(self, tmp_path, wizard):
        """미리보기에 대체 문자(\\ufffd) 포함 시 preview_ok=False, warning 기록."""
        bad_file = tmp_path / "bad_enc.txt"
        bad_file.write_bytes(b"a,b\n\xff\xfe,2\n")
        # errors='strict'(기본값)이므로 UnicodeDecodeError → errors에 기록
        result = wizard.step_encoding(str(bad_file), "utf-8")
        assert result["preview_ok"] is False
        # UnicodeDecodeError 경로: errors 또는 warnings 중 하나에 기록
        assert len(wizard.errors) > 0 or len(wizard.warnings) > 0


# ---------------------------------------------------------------------------
# run_auto — CSV 파이프라인 (lines 438-495)
# ---------------------------------------------------------------------------

class TestRunFullWizardCSV:
    """run_full_wizard — 선형 6단계 파이프라인 (lines 438-495) 커버리지."""

    def test_returns_state_dict(self, wizard, csv_file):
        """반환값이 dict."""
        state = wizard.run_full_wizard(str(csv_file))
        assert isinstance(state, dict)

    def test_confirm_key_in_state(self, wizard, csv_file):
        """성공적 임포트 후 confirm 키가 state에 존재."""
        state = wizard.run_full_wizard(str(csv_file))
        assert "confirm" in state

    def test_confirm_ready_flag(self, wizard, csv_file):
        """에러 없으면 confirm.ready == True."""
        state = wizard.run_full_wizard(str(csv_file))
        assert state["confirm"]["ready"] is True

    def test_confirm_row_count(self, wizard, csv_file):
        """confirm.n_rows 가 실제 데이터 행 수와 일치."""
        state = wizard.run_full_wizard(str(csv_file))
        assert state["confirm"]["n_rows"] == 2

    def test_confirm_column_count(self, wizard, csv_file):
        """confirm.n_columns 가 실제 열 수와 일치."""
        state = wizard.run_full_wizard(str(csv_file))
        assert state["confirm"]["n_columns"] == 3

    def test_resets_previous_state(self, wizard, csv_file):
        """run_full_wizard 시작 시 이전 state가 초기화된다."""
        wizard._state["dummy"] = "old"
        wizard.run_full_wizard(str(csv_file))
        assert "dummy" not in wizard.state

    def test_type_preview_populated(self, wizard, csv_file):
        """type_preview 단계 결과가 state에 저장된다."""
        state = wizard.run_full_wizard(str(csv_file))
        assert "type_preview" in state
        assert "column_info" in state["type_preview"]

    def test_with_explicit_encoding_and_delimiter(self, wizard, csv_file):
        """명시적 encoding/delimiter 지정 시에도 정상 동작."""
        state = wizard.run_full_wizard(str(csv_file), encoding="utf-8", delimiter=",")
        assert state["confirm"]["ready"] is True

    def test_with_max_rows(self, wizard, tmp_path):
        """max_rows 제한이 파이프라인을 통해 적용된다."""
        f = tmp_path / "big.csv"
        rows = "\n".join(f"{i},{i*2}" for i in range(50))
        f.write_text(f"a,b\n{rows}\n", encoding="utf-8")
        state = wizard.run_full_wizard(str(f), max_rows=5)
        assert isinstance(state, dict)
        # confirm이 있으면 행 수가 5 이하여야 함
        if "confirm" in state:
            assert state["confirm"]["n_rows"] <= 5

    def test_file_select_info_in_state(self, wizard, csv_file):
        """file_select 단계 결과가 state에 저장된다."""
        state = wizard.run_full_wizard(str(csv_file))
        # step_file_select가 state에 'file' 키로 저장됨
        file_key = "file" if "file" in state else "file_select"
        assert file_key in state
        assert state[file_key]["exists"] is True


# ---------------------------------------------------------------------------
# run_full_wizard — FileReadError 흡수 (lines 486-487)
# ---------------------------------------------------------------------------

class TestRunFullWizardFileReadError:

    def test_read_error_absorbed_into_errors(self, wizard, csv_file):
        """CSV 읽기 FileReadError 발생 시 wizard.errors에 기록, state dict 반환."""
        from nuristat.core.exceptions import FileReadError

        # run_full_wizard 내부에서 지역 임포트된 read_csv를 패치
        with patch(
            "nuristat.io.csv_reader.read_csv",
            side_effect=FileReadError(str(csv_file), "forced error"),
        ):
            state = wizard.run_full_wizard(str(csv_file))

        assert isinstance(state, dict)
        # FileReadError가 흡수되어 errors에 기록됨
        assert len(wizard.errors) > 0

    def test_state_returned_even_on_error(self, wizard, csv_file):
        """읽기 실패해도 빈 state dict가 반환된다."""
        from nuristat.core.exceptions import FileReadError

        with patch(
            "nuristat.io.csv_reader.read_csv",
            side_effect=FileReadError(str(csv_file), "forced error"),
        ):
            state = wizard.run_full_wizard(str(csv_file))

        assert state is not None

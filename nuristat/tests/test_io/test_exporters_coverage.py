"""exporters.py 커버리지 보강 테스트.

미커버 라인:
  export_html:
    51-56  : df 변환 ValueError/TypeError → continue (테이블 건너뜀)
    73-74  : text_blocks 항목이 dict인 경우
    121-122: out_path.write_text OSError → FileWriteError
  export_csv_table:
    235-245: dict→DataFrame 변환 실패 → FileWriteError
  export_markdown:
    (dict text_block은 동일 패턴으로 보강)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nuristat.core.exceptions import FileWriteError
from nuristat.io.exporters import export_csv_table, export_html, export_markdown


class TestExportHtmlCoverage:

    def test_table_with_bad_data_skipped(self, tmp_path):
        """df 변환이 실패하는 테이블은 건너뛰고 나머지는 정상 출력."""
        result = {
            "tables": [
                {"title": "bad", "dataframe": None, "data": None},
                {"title": "good", "dataframe": pd.DataFrame({"x": [1, 2]})},
            ],
            "text_blocks": [],
            "notes": [],
            "warnings": [],
        }
        path = tmp_path / "partial.html"
        export_html(result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "good" in content

    def test_table_non_df_conversion_error_skipped(self, tmp_path):
        """데이터가 DataFrame으로 변환 불가한 경우 해당 테이블 건너뜀."""
        unconvertible = object()
        result = {
            "tables": [
                {"title": "unconvertible", "dataframe": unconvertible},
                {"title": "ok", "dataframe": pd.DataFrame({"a": [1]})},
            ],
            "text_blocks": [],
            "notes": [],
            "warnings": [],
        }
        path = tmp_path / "skip.html"
        export_html(result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "ok" in content
        assert path.exists()

    def test_text_block_as_dict(self, tmp_path):
        """text_blocks 항목이 dict인 경우 'text' 키 값이 출력된다."""
        result = {
            "tables": [],
            "text_blocks": [{"text": "딕셔너리형 텍스트 블록"}],
            "notes": [],
            "warnings": [],
        }
        path = tmp_path / "dict_block.html"
        export_html(result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "딕셔너리형 텍스트 블록" in content

    def test_text_block_as_dict_empty_text(self, tmp_path):
        """text_blocks dict 항목에 'text' 키 없으면 빈 문자열 처리."""
        result = {
            "tables": [],
            "text_blocks": [{"other": "ignored"}],
            "notes": [],
            "warnings": [],
        }
        path = tmp_path / "empty_dict_block.html"
        export_html(result, str(path))
        assert path.exists()

    def test_write_oserror_raises_file_write_error(self, tmp_path):
        """파일 쓰기 OSError → FileWriteError."""
        path = tmp_path / "fail.html"
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            with pytest.raises(FileWriteError):
                export_html({"tables": [], "text_blocks": [], "notes": [], "warnings": []}, str(path))


class TestExportMarkdownCoverage:

    def test_text_block_as_dict(self, tmp_path):
        """text_blocks 항목이 dict인 경우 'text' 값이 MD에 포함."""
        result = {
            "tables": [],
            "text_blocks": [{"text": "마크다운 딕셔너리 블록"}],
            "notes": [],
            "warnings": [],
        }
        path = tmp_path / "dict_block.md"
        export_markdown(result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "마크다운 딕셔너리 블록" in content

    def test_table_non_df_skipped(self, tmp_path):
        """변환 불가 테이블 건너뜀 후 나머지 정상 출력."""
        result = {
            "tables": [
                {"title": "bad", "dataframe": object()},
                {"title": "good", "dataframe": pd.DataFrame({"v": [1, 2]})},
            ],
            "text_blocks": [],
            "notes": [],
            "warnings": [],
        }
        path = tmp_path / "mixed.md"
        export_markdown(result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "good" in content


class TestExportCsvCoverage:

    def test_dict_conversion_failure_raises(self, tmp_path):
        """dict→DataFrame 변환 실패 시 FileWriteError."""
        path = tmp_path / "fail.csv"
        with pytest.raises(FileWriteError) as exc_info:
            export_csv_table({"dataframe": None, "data": None}, str(path))
        assert "테이블 데이터가 없습니다" in str(exc_info.value) or "없습니다" in str(exc_info.value)

    def test_unconvertible_data_raises(self, tmp_path):
        """dict의 data가 DataFrame으로 변환 불가인 경우 FileWriteError."""
        path = tmp_path / "unconvertible.csv"
        with pytest.raises(FileWriteError):
            export_csv_table({"dataframe": object()}, str(path))

    def test_write_oserror_raises_file_write_error(self, tmp_path):
        """DataFrame.to_csv OSError → FileWriteError."""
        path = tmp_path / "fail.csv"
        df = pd.DataFrame({"a": [1]})
        with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("permission denied")):
            with pytest.raises(FileWriteError):
                export_csv_table(df, str(path))

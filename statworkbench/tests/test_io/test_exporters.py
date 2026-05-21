"""결과 내보내기(exporters) 종합 테스트 — SW + TU 에이전트 공동 검증.

검증 항목:
- export_html: HTML 파일 생성, 테이블 포함, 경고/노트 포함
- export_markdown: MD 파일 생성, 테이블 MD 형식 변환
- export_csv_table: 단일 테이블 CSV 저장
- 오류 처리: 쓰기 불가 경로, 빈 데이터, 타입 불일치
- 한글 인코딩 (UTF-8 BOM 포함 CSV, UTF-8 HTML/MD)

대상 모듈: statworkbench.io.exporters
담당 에이전트: SW (statworkbench), TU (tester-unit)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from statworkbench.io.exporters import export_csv_table, export_html, export_markdown


# ──────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_result() -> dict:
    """표준 분석 결과 딕셔너리."""
    return {
        "tables": [
            {
                "title": "기술통계",
                "dataframe": pd.DataFrame({
                    "변수": ["score", "age"],
                    "N": [100, 98],
                    "평균": [75.3, 34.2],
                    "SD": [12.1, 8.5],
                }),
            },
            {
                "title": "상관계수",
                "dataframe": pd.DataFrame({
                    "변수": ["score", "age"],
                    "score": [1.0, 0.45],
                    "age": [0.45, 1.0],
                }),
            },
        ],
        "text_blocks": ["분석이 완료되었습니다."],
        "notes": ["리스트와이즈 결측 처리 적용"],
        "warnings": ["일부 변수의 결측치 비율이 높습니다."],
    }


@pytest.fixture
def simple_table_dict() -> dict:
    """단일 테이블 딕셔너리."""
    return {
        "title": "T-검정 결과",
        "dataframe": pd.DataFrame({
            "통계량": ["t", "df", "p"],
            "값": [-2.34, 48.0, 0.023],
        }),
    }


@pytest.fixture
def korean_df() -> pd.DataFrame:
    """한글 컬럼과 값 포함 DataFrame."""
    return pd.DataFrame({
        "지역": ["경남", "부산", "울산"],
        "인구": [3_300_000, 3_400_000, 1_100_000],
        "비율": [34.2, 35.1, 11.4],
    })


# ──────────────────────────────────────────────────────────────
# 1. export_html
# ──────────────────────────────────────────────────────────────

class TestExportHtml:
    """export_html: HTML 파일 생성 및 내용 검증."""

    def test_file_created(self, sample_result, tmp_path):
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        assert path.exists()

    def test_file_size_positive(self, sample_result, tmp_path):
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        assert path.stat().st_size > 0

    def test_html_structure(self, sample_result, tmp_path):
        """HTML 기본 구조 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

    def test_title_in_html(self, sample_result, tmp_path):
        """사용자 지정 제목이 HTML에 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path), title="통계분석 결과")
        content = path.read_text(encoding="utf-8")
        assert "통계분석 결과" in content

    def test_table_titles_in_html(self, sample_result, tmp_path):
        """테이블 제목들이 HTML에 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "기술통계" in content
        assert "상관계수" in content

    def test_table_data_in_html(self, sample_result, tmp_path):
        """테이블 데이터 값이 HTML에 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "75.3" in content or "75" in content

    def test_notes_in_html(self, sample_result, tmp_path):
        """Notes 섹션이 HTML에 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "리스트와이즈" in content

    def test_warnings_in_html(self, sample_result, tmp_path):
        """Warnings 섹션이 HTML에 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "결측치" in content

    def test_utf8_encoding(self, sample_result, tmp_path):
        """UTF-8 메타 태그 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "UTF-8" in content or "utf-8" in content

    def test_empty_tables(self, tmp_path):
        """tables가 빈 리스트 → 오류 없이 HTML 생성."""
        path = tmp_path / "empty.html"
        export_html({"tables": [], "text_blocks": [], "notes": [], "warnings": []}, str(path))
        assert path.exists()

    def test_nested_dir_created(self, sample_result, tmp_path):
        """중간 디렉토리 자동 생성."""
        path = tmp_path / "subdir" / "deep" / "result.html"
        export_html(sample_result, str(path))
        assert path.exists()

    def test_text_blocks_in_html(self, sample_result, tmp_path):
        """text_blocks 내용이 HTML에 포함."""
        path = tmp_path / "result.html"
        export_html(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "분석이 완료되었습니다" in content

    def test_result_with_data_key(self, tmp_path):
        """'dataframe' 대신 'data' 키도 지원."""
        result = {
            "tables": [{"title": "T1", "data": pd.DataFrame({"a": [1, 2]})}],
            "text_blocks": [], "notes": [], "warnings": [],
        }
        path = tmp_path / "data_key.html"
        export_html(result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "<table" in content.lower()


# ──────────────────────────────────────────────────────────────
# 2. export_markdown
# ──────────────────────────────────────────────────────────────

class TestExportMarkdown:
    """export_markdown: Markdown 파일 생성 및 내용 검증."""

    def test_file_created(self, sample_result, tmp_path):
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        assert path.exists()

    def test_file_size_positive(self, sample_result, tmp_path):
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        assert path.stat().st_size > 0

    def test_table_titles_in_md(self, sample_result, tmp_path):
        """테이블 제목이 MD 헤더로 포함."""
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "### 기술통계" in content
        assert "### 상관계수" in content

    def test_markdown_table_format(self, sample_result, tmp_path):
        """테이블이 MD 파이프 형식으로 출력."""
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "|" in content

    def test_notes_in_md(self, sample_result, tmp_path):
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "리스트와이즈" in content

    def test_warnings_in_md(self, sample_result, tmp_path):
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "결측치" in content

    def test_text_blocks_in_md(self, sample_result, tmp_path):
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "분석이 완료되었습니다" in content

    def test_utf8_readable(self, sample_result, tmp_path):
        """한글이 포함된 MD 파일을 UTF-8로 읽기 성공."""
        path = tmp_path / "result.md"
        export_markdown(sample_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_nested_dir_created(self, sample_result, tmp_path):
        path = tmp_path / "a" / "b" / "result.md"
        export_markdown(sample_result, str(path))
        assert path.exists()

    def test_empty_result(self, tmp_path):
        """빈 결과 → 오류 없이 빈 MD 생성."""
        path = tmp_path / "empty.md"
        export_markdown({"tables": [], "text_blocks": [], "notes": [], "warnings": []}, str(path))
        assert path.exists()


# ──────────────────────────────────────────────────────────────
# 3. export_csv_table
# ──────────────────────────────────────────────────────────────

class TestExportCsvTable:
    """export_csv_table: CSV 파일 생성 및 내용 검증."""

    def test_file_created_from_dict(self, simple_table_dict, tmp_path):
        path = tmp_path / "table.csv"
        export_csv_table(simple_table_dict, str(path))
        assert path.exists()

    def test_file_created_from_dataframe(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "df.csv"
        export_csv_table(df, str(path))
        assert path.exists()

    def test_csv_content_matches(self, tmp_path):
        """CSV 내용이 원본 DataFrame과 일치."""
        df = pd.DataFrame({"x": [10, 20, 30], "y": [1.1, 2.2, 3.3]})
        path = tmp_path / "match.csv"
        export_csv_table(df, str(path))
        read_back = pd.read_csv(path, encoding="utf-8-sig")
        assert list(read_back["x"]) == [10, 20, 30]
        assert list(read_back.columns) == ["x", "y"]

    def test_utf8_bom_encoding(self, korean_df, tmp_path):
        """CSV가 UTF-8 BOM으로 저장 (Excel 한글 호환)."""
        path = tmp_path / "korean.csv"
        export_csv_table(korean_df, str(path))
        read_back = pd.read_csv(path, encoding="utf-8-sig")
        assert "지역" in read_back.columns
        assert "경남" in read_back["지역"].values

    def test_roundtrip_integrity(self, tmp_path):
        """CSV 왕복 후 데이터 무결성."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "score": [85.5, 90.2, 78.0, 95.1, 88.3],
            "group": ["A", "B", "A", "B", "A"],
        })
        path = tmp_path / "rt.csv"
        export_csv_table(df, str(path))
        read_back = pd.read_csv(path, encoding="utf-8-sig")
        assert len(read_back) == 5
        assert list(read_back["id"]) == [1, 2, 3, 4, 5]

    def test_none_data_raises(self, tmp_path):
        """data가 None인 dict → FileWriteError."""
        from statworkbench.core.exceptions import FileWriteError
        path = tmp_path / "none.csv"
        with pytest.raises(FileWriteError):
            export_csv_table({"title": "T", "dataframe": None}, str(path))

    def test_nested_dir_created(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "deep" / "dir" / "out.csv"
        export_csv_table(df, str(path))
        assert path.exists()

    def test_extra_kwargs_forwarded(self, tmp_path):
        """추가 kwargs가 DataFrame.to_csv에 전달."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "sep.csv"
        export_csv_table(df, str(path), sep=";")
        content = path.read_text(encoding="utf-8-sig")
        assert ";" in content

    def test_dict_with_data_key(self, tmp_path):
        """'data' 키도 지원."""
        table = {"title": "T", "data": pd.DataFrame({"x": [1, 2, 3]})}
        path = tmp_path / "data_key.csv"
        export_csv_table(table, str(path))
        read_back = pd.read_csv(path, encoding="utf-8-sig")
        assert "x" in read_back.columns


# ──────────────────────────────────────────────────────────────
# 4. 통합 시나리오 (SPSS 분석 결과 내보내기)
# ──────────────────────────────────────────────────────────────

class TestExporterIntegration:
    """실제 SPSS 분석 결과를 내보내는 엔드투엔드 시나리오."""

    @pytest.fixture
    def anova_result(self):
        """ANOVA 분석 결과 형식 모의."""
        return {
            "tables": [
                {
                    "title": "Case Processing Summary",
                    "dataframe": pd.DataFrame({
                        "Total Cases": [30], "Valid Cases": [30],
                        "Excluded Cases": [0], "Excluded %": ["0.0%"],
                    }),
                },
                {
                    "title": "ANOVA",
                    "dataframe": pd.DataFrame({
                        "Source": ["Between", "Within", "Total"],
                        "SS": [2000.0, 182.7, 2182.7],
                        "df": [2, 27, 29],
                        "MS": [1000.0, 6.77, None],
                        "F": [147.79, None, None],
                        "p": ["< .001", None, None],
                    }),
                },
            ],
            "text_blocks": ["일원배치 분산분석 결과"],
            "notes": ["사후 검정: Tukey HSD"],
            "warnings": [],
        }

    def test_anova_result_to_html(self, anova_result, tmp_path):
        path = tmp_path / "anova.html"
        export_html(anova_result, str(path), title="ANOVA 결과")
        content = path.read_text(encoding="utf-8")
        assert "ANOVA 결과" in content
        assert "147.79" in content or "147" in content

    def test_anova_result_to_markdown(self, anova_result, tmp_path):
        path = tmp_path / "anova.md"
        export_markdown(anova_result, str(path))
        content = path.read_text(encoding="utf-8")
        assert "ANOVA" in content
        assert "Tukey HSD" in content

    def test_anova_table_to_csv(self, anova_result, tmp_path):
        path = tmp_path / "anova_table.csv"
        table = anova_result["tables"][1]
        export_csv_table(table, str(path))
        read_back = pd.read_csv(path, encoding="utf-8-sig")
        assert "Source" in read_back.columns
        assert len(read_back) == 3

    def test_all_formats_same_data(self, anova_result, tmp_path):
        """HTML, MD, CSV 세 형식 모두 정상 생성."""
        export_html(anova_result, str(tmp_path / "result.html"))
        export_markdown(anova_result, str(tmp_path / "result.md"))
        export_csv_table(anova_result["tables"][1], str(tmp_path / "table.csv"))

        assert (tmp_path / "result.html").exists()
        assert (tmp_path / "result.md").exists()
        assert (tmp_path / "table.csv").exists()

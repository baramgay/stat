"""ReportEngine 테스트.

검증 항목:
- generate_html_report: HTML 구조, 제목 포함, 데이터셋 정보
- generate_data_quality_report: 결측치 섹션, 중복 행 섹션, HTML 유효성
- 빈 분석 목록 / 결측치 없는 데이터 처리
- 보고서 제목·작성자 반영
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.reporting.report_engine import ReportEngine


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> ReportEngine:
    return ReportEngine()


@pytest.fixture
def simple_dataset() -> Dataset:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "score": rng.normal(70, 15, 50),
        "age":   rng.integers(20, 60, 50).astype(float),
    })
    return Dataset(data=df, name="sample")


@pytest.fixture
def missing_dataset() -> Dataset:
    df = pd.DataFrame({
        "x": [1.0, np.nan, np.nan, np.nan, 5.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    return Dataset(data=df, name="missing_test")


@pytest.fixture
def duplicate_dataset() -> Dataset:
    df = pd.DataFrame({
        "a": [1, 1, 2, 3],
        "b": [4, 4, 5, 6],
    })
    return Dataset(data=df, name="dup_test")


# ──────────────────────────────────────────────────────────────
# 1. HTML 보고서 생성
# ──────────────────────────────────────────────────────────────

class TestGenerateHtmlReport:

    def test_returns_string(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [])
        assert isinstance(html, str)

    def test_html_doctype(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [])
        assert "<!DOCTYPE html>" in html

    def test_html_charset_utf8(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [])
        assert "UTF-8" in html

    def test_default_title_in_html(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [])
        assert "StatWorkbench" in html

    def test_custom_title_in_html(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [], title="내 보고서")
        assert "내 보고서" in html

    def test_author_in_html(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [], author="홍길동")
        assert "홍길동" in html

    def test_dataset_name_in_html(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [])
        assert simple_dataset.name in html

    def test_analysis_section_included(self, engine, simple_dataset):
        analyses = [{"type": "t-검정", "result": "p = 0.023"}]
        html = engine.generate_html_report(simple_dataset, analyses)
        assert "t-검정" in html

    def test_empty_analyses_no_error(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [])
        assert len(html) > 0

    def test_html_has_body_tag(self, engine, simple_dataset):
        html = engine.generate_html_report(simple_dataset, [])
        assert "<body>" in html and "</body>" in html


# ──────────────────────────────────────────────────────────────
# 2. 데이터 품질 보고서
# ──────────────────────────────────────────────────────────────

class TestGenerateDataQualityReport:

    def test_returns_string(self, engine, simple_dataset):
        html = engine.generate_data_quality_report(simple_dataset)
        assert isinstance(html, str)

    def test_html_structure(self, engine, simple_dataset):
        html = engine.generate_data_quality_report(simple_dataset)
        assert "<!DOCTYPE html>" in html
        assert "UTF-8" in html

    def test_missing_section_included_when_missing(self, engine, missing_dataset):
        html = engine.generate_data_quality_report(missing_dataset)
        assert "결측치" in html

    def test_no_missing_section_when_clean(self, engine, simple_dataset):
        html = engine.generate_data_quality_report(simple_dataset)
        assert "결측치가 없습니다" in html

    def test_duplicate_section_included(self, engine, duplicate_dataset):
        html = engine.generate_data_quality_report(duplicate_dataset)
        assert "중복 행" in html

    def test_row_count_in_report(self, engine, simple_dataset):
        html = engine.generate_data_quality_report(simple_dataset)
        assert str(len(simple_dataset.data)) in html

    def test_column_count_in_report(self, engine, simple_dataset):
        html = engine.generate_data_quality_report(simple_dataset)
        assert str(len(simple_dataset.data.columns)) in html

    def test_dataset_name_in_report(self, engine, missing_dataset):
        html = engine.generate_data_quality_report(missing_dataset)
        assert missing_dataset.name in html


# ──────────────────────────────────────────────────────────────
# 3. 엔진 초기화
# ──────────────────────────────────────────────────────────────

class TestReportEngineInit:

    def test_instantiation(self):
        eng = ReportEngine()
        assert isinstance(eng, ReportEngine)

    def test_css_style_set(self, engine):
        assert hasattr(engine, "_css_style")
        assert isinstance(engine._css_style, str)
        assert len(engine._css_style) > 0

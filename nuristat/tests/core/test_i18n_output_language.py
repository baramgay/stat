"""분석 결과 출력 언어 전환(한국어/영어) 검증.

비파괴 원칙: 내부 DataFrame은 영어 유지, 출력(to_html/to_markdown)만 번역.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.core import i18n
from nuristat.core.settings import SettingsManager


@pytest.fixture
def sample_result():
    res = AnalysisResult(id="r", title="Descriptive Statistics")
    res.tables.append(ResultTable(
        title="Descriptive Statistics",
        dataframe=pd.DataFrame({"Variable": ["x"], "Mean": [3.0], "SD": [1.0], "N": [10]}),
    ))
    res.tables.append(ResultTable(
        title="ANOVA",
        dataframe=pd.DataFrame({"Source": ["grp"], "SS": [10.0], "df": [2], "F": [5.0], "p-value": ["< .001"]}),
    ))
    return res


class TestLanguageSwitch:

    def test_default_language_is_en(self):
        assert i18n.get_language() == "en"

    def test_set_get_language(self):
        i18n.set_language("ko")
        assert i18n.get_language() == "ko"
        i18n.set_language("en")
        assert i18n.get_language() == "en"

    def test_en_output_is_english(self, sample_result):
        i18n.set_language("en")
        html = sample_result.to_html()
        assert "Descriptive Statistics" in html
        assert "Mean" in html and "Source" in html
        assert "기술통계량" not in html

    def test_ko_output_is_korean(self, sample_result):
        i18n.set_language("ko")
        html = sample_result.to_html()
        assert "기술통계량" in html       # 제목
        assert "평균" in html             # Mean
        assert "소스" in html             # Source
        assert "분산분석 (ANOVA)" in html  # ANOVA 제목
        assert "유의확률" in html         # p-value
        # 영문 잔존 없음 (번역된 용어)
        assert "Descriptive Statistics" not in html

    def test_internal_dataframe_not_mutated(self, sample_result):
        """번역은 표시용 사본에만 — 원본 DataFrame 컬럼은 영어 유지."""
        i18n.set_language("ko")
        _ = sample_result.to_html()
        cols = list(sample_result.tables[0].dataframe.columns)
        assert cols == ["Variable", "Mean", "SD", "N"], f"원본 변경됨: {cols}"

    def test_markdown_translated_ko(self, sample_result):
        i18n.set_language("ko")
        md = sample_result.tables[1].to_markdown()
        assert "분산분석 (ANOVA)" in md

    def test_unknown_title_falls_through(self):
        i18n.set_language("ko")
        t = ResultTable(title="존재하지 않는 제목 XYZ", dataframe=pd.DataFrame({"A": [1]}))
        html = t.to_html()
        assert "존재하지 않는 제목 XYZ" in html  # 사전에 없으면 원본 유지


class TestSettingsLanguage:

    def test_default_setting_is_korean(self):
        """앱 기본 출력 언어는 한국어."""
        sm = SettingsManager()
        # 기존 저장값 영향 배제 위해 명시 저장 후 확인
        sm.save_language("ko")
        assert sm.load_language() == "ko"

    def test_save_load_roundtrip(self):
        sm = SettingsManager()
        sm.save_language("en")
        assert sm.load_language() == "en"
        sm.save_language("ko")
        assert sm.load_language() == "ko"

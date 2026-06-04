"""텍스트 마이닝 분석 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from nuristat.analysis.text_mining import run_analysis, _tokenize
from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType


# ── 픽스처 ──────────────────────────────────────────────────────────────────

_TEXTS_KO = [
    "경남 빅데이터 센터에서 데이터 분석을 수행합니다",
    "공공데이터를 활용한 정책 분석 사례 연구",
    "데이터 기반 의사결정 지원 시스템 구축",
    "빅데이터 플랫폼 운영 및 분석 서비스 제공",
    "경남 지역 주요 현안 데이터 분석 결과 보고",
    "통계 패키지를 이용한 분석 방법론 연구",
    "데이터 수집 정제 분석 시각화 단계별 수행",
    "인공지능 기반 빅데이터 분석 모델 개발",
    "공공 서비스 품질 향상을 위한 데이터 활용",
    "지역 경제 현황 분석 및 정책 제언 연구",
]

_TEXTS_EN = [
    "data analysis with Python and pandas",
    "statistical methods for data science research",
    "machine learning algorithms in data analysis",
    "data visualization and reporting techniques",
    "public data analysis for policy making",
]


def _make_dataset(texts: list, col: str = "text") -> Dataset:
    df = pd.DataFrame({col: texts})
    meta = {col: VariableMeta(name=col, measure=MeasureType.NOMINAL, storage_type=StorageType.STRING)}
    return Dataset(data=df, variables=meta)


def _default_spec(**overrides) -> dict:
    spec: dict = {
        "variables": {"text_column": "text"},
        "options": {
            "top_n": 10,
            "min_word_len": 2,
            "ngram": "bigram",
            "tfidf": False,
            "wordcloud": False,  # 테스트에서는 이미지 생성 비활성
            "language": "ko",
        },
    }
    for k, v in overrides.items():
        if k == "options":
            spec["options"].update(v)
        elif k == "variables":
            spec["variables"].update(v)
        else:
            spec[k] = v
    return spec


# ── 구조 검증 ────────────────────────────────────────────────────────────────

class TestTextMiningStructure:
    def setup_method(self):
        self.ds = _make_dataset(_TEXTS_KO)
        self.res = run_analysis(self.ds, _default_spec())

    def test_result_id(self):
        assert self.res.id == "text_mining"

    def test_no_warnings_on_clean_data(self):
        assert not self.res.warnings

    def test_has_summary_table(self):
        assert any("요약" in t.title for t in self.res.tables)

    def test_summary_has_correct_doc_count(self):
        summary = next(t for t in self.res.tables if "요약" in t.title)
        row = summary.dataframe.iloc[0]
        assert int(row["문서(행) 수"]) == len(_TEXTS_KO)

    def test_has_frequency_table(self):
        assert any("빈도" in t.title for t in self.res.tables)

    def test_frequency_top_n(self):
        freq = next(t for t in self.res.tables if "빈도" in t.title)
        assert len(freq.dataframe) <= 10

    def test_frequency_has_rank_column(self):
        freq = next(t for t in self.res.tables if "빈도" in t.title)
        assert "순위" in freq.dataframe.columns

    def test_frequency_rank_starts_at_1(self):
        freq = next(t for t in self.res.tables if "빈도" in t.title)
        assert int(freq.dataframe.iloc[0]["순위"]) == 1

    def test_has_bigram_table_by_default(self):
        assert any("바이그램" in t.title for t in self.res.tables)

    def test_has_notes(self):
        assert len(self.res.notes) >= 1


# ── 통계 검증 ────────────────────────────────────────────────────────────────

class TestTextMiningStatistics:
    def setup_method(self):
        self.ds = _make_dataset(_TEXTS_KO)
        self.res = run_analysis(self.ds, _default_spec())

    def test_most_frequent_word_is_data(self):
        freq = next(t for t in self.res.tables if "빈도" in t.title)
        top_word = freq.dataframe.iloc[0]["단어"]
        # "데이터" 또는 "분석"이 최상위 단어여야 함
        assert top_word in ("데이터", "분석", "빅데이터", "공공데이터")

    def test_frequency_descending_order(self):
        freq = next(t for t in self.res.tables if "빈도" in t.title)
        counts = freq.dataframe["빈도"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_relative_frequency_sums_reasonably(self):
        freq = next(t for t in self.res.tables if "빈도" in t.title)
        total_pct = sum(float(v) for v in freq.dataframe["상대빈도(%)"])
        assert 0 < total_pct <= 100

    def test_total_tokens_in_summary(self):
        summary = next(t for t in self.res.tables if "요약" in t.title)
        total = int(summary.dataframe.iloc[0]["총 토큰 수"])
        assert total > 0

    def test_unique_words_le_total_tokens(self):
        summary = next(t for t in self.res.tables if "요약" in t.title)
        total = int(summary.dataframe.iloc[0]["총 토큰 수"])
        unique = int(summary.dataframe.iloc[0]["고유 단어 수"])
        assert unique <= total

    def test_bigram_count(self):
        bigram = next(t for t in self.res.tables if "바이그램" in t.title)
        assert len(bigram.dataframe) >= 1


# ── 옵션 검증 ────────────────────────────────────────────────────────────────

class TestTextMiningOptions:
    def test_trigram(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(options={"ngram": "trigram"}))
        assert any("트라이그램" in t.title for t in res.tables)
        assert not any("바이그램" in t.title for t in res.tables)

    def test_no_ngram(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(options={"ngram": "none"}))
        assert not any("바이그램" in t.title or "트라이그램" in t.title for t in res.tables)

    def test_tfidf_on(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(options={"tfidf": True}))
        assert any("TF-IDF" in t.title for t in res.tables)

    def test_tfidf_off(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(options={"tfidf": False}))
        assert not any("TF-IDF" in t.title for t in res.tables)

    def test_english_language(self):
        ds = _make_dataset(_TEXTS_EN)
        res = run_analysis(ds, _default_spec(options={"language": "en"}))
        assert not res.warnings
        freq = next(t for t in res.tables if "빈도" in t.title)
        assert len(freq.dataframe) >= 1

    def test_custom_top_n(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(options={"top_n": 5}))
        freq = next(t for t in res.tables if "빈도" in t.title)
        assert len(freq.dataframe) <= 5

    def test_extra_stopwords(self):
        ds = _make_dataset(_TEXTS_KO)
        res_before = run_analysis(ds, _default_spec())
        freq_before = next(t for t in res_before.tables if "빈도" in t.title)
        # "데이터"를 불용어로 추가하면 빈도 표에서 사라져야 함
        res_after = run_analysis(ds, _default_spec(options={"stopwords": ["데이터"]}))
        freq_after = next(t for t in res_after.tables if "빈도" in t.title)
        words_after = freq_after.dataframe["단어"].tolist()
        assert "데이터" not in words_after

    def test_min_word_len(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(options={"min_word_len": 4}))
        freq = next(t for t in res.tables if "빈도" in t.title)
        for word in freq.dataframe["단어"]:
            assert len(str(word)) >= 4


# ── 입력 검증 ────────────────────────────────────────────────────────────────

class TestTextMiningInputValidation:
    def test_missing_text_column(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(variables={"text_column": ""}))
        assert res.warnings

    def test_nonexistent_column(self):
        ds = _make_dataset(_TEXTS_KO)
        res = run_analysis(ds, _default_spec(variables={"text_column": "nosuchcol"}))
        assert res.warnings

    def test_empty_dataset(self):
        import pandas as pd
        ds = Dataset(data=pd.DataFrame())
        res = run_analysis(ds, _default_spec())
        assert res.warnings

    def test_all_nan_text(self):
        ds = _make_dataset([None, None, None])
        res = run_analysis(ds, _default_spec())
        assert res.warnings


# ── 토크나이저 단위 테스트 ──────────────────────────────────────────────────

class TestTokenizer:
    def test_removes_numbers_only_tokens(self):
        tokens = _tokenize("12345 hello world", min_len=2, stopwords=set())
        assert "12345" not in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("a ab abc abcd", min_len=3, stopwords=set())
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens

    def test_removes_stopwords(self):
        tokens = _tokenize("이것은 데이터 분석 결과입니다", min_len=2, stopwords={"이것은", "결과입니다"})
        assert "이것은" not in tokens
        assert "데이터" in tokens

    def test_removes_special_chars(self):
        tokens = _tokenize("hello, world! 안녕하세요?", min_len=2, stopwords=set())
        # 구두점이 제거된 형태여야 함
        assert "hello," not in tokens
        assert "hello" in tokens

    def test_empty_string(self):
        tokens = _tokenize("", min_len=2, stopwords=set())
        assert tokens == []

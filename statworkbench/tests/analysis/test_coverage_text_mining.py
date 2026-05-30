"""text_mining.py 미커버 경로 보완 테스트 (wordcloud, _find_font, TF-IDF 경계)."""

from __future__ import annotations

import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType
from statworkbench.analysis.text_mining import run_analysis, _tokenize, _compute_tfidf, _find_font


def _str_var(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.STRING, measure=MeasureType.NOMINAL)


def _make_dataset(texts: list[str]) -> Dataset:
    df = pd.DataFrame({"text": texts})
    ds = Dataset(df, name="text_test")
    ds.variables["text"] = _str_var("text")
    return ds


SAMPLE_TEXTS = [
    "데이터 분석은 중요한 기술입니다 데이터 처리가 필요합니다",
    "통계적 방법론을 활용한 데이터 분석이 핵심입니다",
    "기계학습과 딥러닝으로 데이터 패턴을 발견합니다",
    "공공 데이터를 활용한 정책 분석이 중요합니다",
    "빅데이터 환경에서의 통계 분석 방법론을 연구합니다",
]


# ── _find_font ────────────────────────────────────────────────────────────────

class TestFindFont:
    def test_english_returns_none(self):
        assert _find_font("en") is None

    def test_korean_returns_str_or_none(self):
        result = _find_font("ko")
        assert result is None or isinstance(result, str)

    def test_unknown_language_tries_candidates(self):
        result = _find_font("fr")
        assert result is None or isinstance(result, str)


# ── _compute_tfidf 경계 ───────────────────────────────────────────────────────

class TestComputeTFIDF:
    def test_single_document_returns_empty(self):
        """문서 1개는 TF-IDF 불가 — 빈 리스트 반환."""
        result = _compute_tfidf([["데이터", "분석"]], top_n=5)
        assert result == []

    def test_empty_vocab_returns_empty(self):
        result = _compute_tfidf([[], []], top_n=5)
        assert result == []

    def test_two_docs_returns_ranked_words(self):
        doc_tokens = [
            ["데이터", "분석", "데이터"],
            ["통계", "분석", "방법론"],
        ]
        result = _compute_tfidf(doc_tokens, top_n=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        if result:
            assert "단어" in result[0]
            assert "TF-IDF (평균)" in result[0]


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_removes_stopwords(self):
        tokens = _tokenize("이것은 테스트입니다", min_len=2, stopwords={"이것은"})
        assert "이것은" not in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("a 데이터 분석", min_len=2, stopwords=set())
        assert "a" not in tokens

    def test_removes_digits(self):
        tokens = _tokenize("123 데이터 456", min_len=2, stopwords=set())
        assert "123" not in tokens
        assert "456" not in tokens

    def test_removes_punctuation(self):
        tokens = _tokenize("데이터, 분석.", min_len=2, stopwords=set())
        assert all("," not in t and "." not in t for t in tokens)


# ── run_analysis (워드클라우드 포함) ─────────────────────────────────────────

class TestRunAnalysisWordcloud:
    def _spec(self, do_wordcloud: bool = True, do_tfidf: bool = True) -> dict:
        return {
            "variables": {"text_column": "text"},
            "options": {
                "wordcloud": do_wordcloud,
                "tfidf": do_tfidf,
                "top_n": 5,
                "min_len": 2,
            },
        }

    def test_basic_run(self):
        ds = _make_dataset(SAMPLE_TEXTS)
        result = run_analysis(ds, self._spec(do_wordcloud=False))
        assert result is not None
        assert len(result.tables) > 0

    def test_tfidf_table_present(self):
        ds = _make_dataset(SAMPLE_TEXTS)
        result = run_analysis(ds, self._spec(do_wordcloud=False, do_tfidf=True))
        titles = [t.title for t in result.tables]
        assert any("TF-IDF" in t for t in titles)

    def test_wordcloud_does_not_crash(self):
        """wordcloud 라이브러리 유무 관계없이 crash 없어야 함."""
        ds = _make_dataset(SAMPLE_TEXTS)
        result = run_analysis(ds, self._spec(do_wordcloud=True))
        assert result is not None

    def test_wordcloud_missing_library_graceful(self):
        """wordcloud 없어도 결과 정상 반환."""
        import unittest.mock as mock
        ds = _make_dataset(SAMPLE_TEXTS)
        spec = self._spec(do_wordcloud=True)
        with mock.patch.dict("sys.modules", {"wordcloud": None}):
            result = run_analysis(ds, spec)
        assert result is not None

    def test_single_text_no_tfidf(self):
        """텍스트 1건 — TF-IDF 테이블 없어도 정상 반환."""
        ds = _make_dataset(["데이터 분석 통계"])
        result = run_analysis(ds, self._spec(do_tfidf=True, do_wordcloud=False))
        assert result is not None

    def test_empty_texts_graceful(self):
        """빈 텍스트 행 포함 처리."""
        ds = _make_dataset(["", "데이터 분석", ""])
        result = run_analysis(ds, self._spec(do_wordcloud=False))
        assert result is not None

    def test_notes_appended(self):
        ds = _make_dataset(SAMPLE_TEXTS)
        result = run_analysis(ds, self._spec(do_wordcloud=False))
        assert len(result.notes) >= 1

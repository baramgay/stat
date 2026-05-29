"""텍스트 마이닝(Text Mining) 분석 모듈.

지원 기능:
  - 단어 빈도 분석 (Top-N)
  - N-gram 빈도 분석 (바이그램, 트라이그램)
  - TF-IDF 분석 (문서-단어 행렬)
  - 워드클라우드 이미지 생성
  - 불용어 처리 (내장 한국어/영어 불용어 + 사용자 정의)
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset


# ── 내장 불용어 ───────────────────────────────────────────────────────────────
_KO_STOPWORDS = {
    "이", "그", "저", "것", "수", "있", "하", "되", "않", "없", "나", "우리",
    "같이", "같은", "이런", "그런", "저런", "이렇게", "그렇게", "저렇게",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로", "으로",
    "에서", "에게", "한테", "께", "부터", "까지", "만", "뿐", "라도", "이라도",
    "하고", "이고", "며", "이며", "거나", "이거나", "든지", "이든지",
    "는데", "은데", "인데", "지만", "이지만", "지", "고", "서", "어서", "아서",
    "때문에", "으로써", "으로서", "에도", "에서도",
    "그리고", "그러나", "그런데", "그래서", "따라서", "또한", "또", "및",
    "즉", "즉", "등", "등등", "등의", "대한", "관한", "위한", "통해", "통하여",
    "위해", "위하여", "대해", "대하여", "관해", "관하여",
    "많은", "많이", "모든", "각각", "각", "더", "덜", "매우", "아주", "너무",
    "정말", "진짜", "참", "꽤", "상당히",
    "것이", "것은", "것을", "것도", "것의", "것에", "것과",
    "제", "이제", "그제", "지금", "오늘", "이번", "그때", "바로",
}

_EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "when", "where", "how",
    "if", "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "more", "most", "other", "some", "such", "than", "then",
    "too", "very", "just", "also", "than", "there",
}


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """텍스트 마이닝 분석을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.text_column:  텍스트 컬럼 이름
            options.top_n:          상위 N개 단어 (기본 30)
            options.min_word_len:   최소 단어 길이 (기본 2)
            options.ngram:          N-gram 크기 ("bigram" | "trigram" | "none", 기본 "bigram")
            options.tfidf:          True=TF-IDF 분석 포함 (기본 False)
            options.wordcloud:      True=워드클라우드 이미지 생성 (기본 True)
            options.stopwords:      추가 불용어 목록 (기본 [])
            options.language:       "ko" | "en" | "both" (기본 "ko")
            options.wc_width:       워드클라우드 너비 px (기본 800)
            options.wc_height:      워드클라우드 높이 px (기본 400)
            options.wc_max_words:   워드클라우드 최대 단어 수 (기본 100)
            missing_policy:         결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. 분석 요약 (케이스 수, 총 어절 수, 고유 단어 수)
            2. 단어 빈도 Top-N
            3. N-gram 빈도 (선택)
            4. TF-IDF 상위 단어 (선택)
            5. 워드클라우드 이미지 (PNG bytes, 선택)
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})

    text_col: str = variables.get("text_column", "")
    top_n: int = int(options.get("top_n", 30))
    min_len: int = int(options.get("min_word_len", 2))
    ngram_type: str = options.get("ngram", "bigram")
    do_tfidf: bool = options.get("tfidf", False)
    do_wordcloud: bool = options.get("wordcloud", True)
    extra_stopwords: list[str] = list(options.get("stopwords", []))
    language: str = options.get("language", "ko").lower()
    wc_width: int = int(options.get("wc_width", 800))
    wc_height: int = int(options.get("wc_height", 400))
    wc_max_words: int = int(options.get("wc_max_words", 100))

    result = AnalysisResult(id="text_mining", title="텍스트 마이닝 (Text Mining)")

    # ── 입력 검증 ─────────────────────────────────────────────────────────────
    if dataset.data is None or len(dataset.data) == 0:
        result.add_warning("데이터셋이 비어 있습니다.")
        return result
    if not text_col:
        result.add_warning("텍스트 컬럼을 지정하세요.")
        return result
    if text_col not in dataset.data.columns:
        result.add_warning(f"컬럼 '{text_col}'이(가) 데이터셋에 없습니다.")
        return result

    # ── 텍스트 준비 ──────────────────────────────────────────────────────────
    series = dataset.data[text_col].dropna().astype(str)
    total_docs = len(series)
    if total_docs == 0:
        result.add_warning("유효한 텍스트 행이 없습니다.")
        return result

    # 불용어 집합
    stopwords: set[str] = set(s.lower() for s in extra_stopwords)
    if language in ("ko", "both"):
        stopwords |= _KO_STOPWORDS
    if language in ("en", "both"):
        stopwords |= _EN_STOPWORDS

    # 토큰화 (공백 기반 + 특수문자 제거)
    all_tokens: list[str] = []
    doc_tokens: list[list[str]] = []
    for text in series:
        tokens = _tokenize(text, min_len, stopwords)
        all_tokens.extend(tokens)
        doc_tokens.append(tokens)

    total_tokens = len(all_tokens)
    unique_words = len(set(all_tokens))

    # ── 요약 테이블 ──────────────────────────────────────────────────────────
    summary_table = ResultTable(
        title="분석 요약",
        dataframe=pd.DataFrame([{
            "분석 컬럼": text_col,
            "문서(행) 수": total_docs,
            "총 토큰 수": total_tokens,
            "고유 단어 수": unique_words,
            "언어 설정": language,
        }]),
    )
    result.add_table(summary_table)

    # ── 단어 빈도 ─────────────────────────────────────────────────────────────
    freq = Counter(all_tokens)
    top_words = freq.most_common(top_n)
    freq_rows = [
        {"순위": i + 1, "단어": w, "빈도": cnt, "상대빈도(%)": round(cnt / total_tokens * 100, 2) if total_tokens else 0}
        for i, (w, cnt) in enumerate(top_words)
    ]
    result.add_table(ResultTable(
        title=f"단어 빈도 Top-{top_n}",
        dataframe=pd.DataFrame(freq_rows),
    ))

    # ── N-gram ────────────────────────────────────────────────────────────────
    if ngram_type in ("bigram", "trigram"):
        n = 2 if ngram_type == "bigram" else 3
        ngram_counter: Counter = Counter()
        for tokens in doc_tokens:
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i: i + n])
                ngram_counter[gram] += 1
        top_ngrams = ngram_counter.most_common(top_n)
        label = "바이그램" if n == 2 else "트라이그램"
        ngram_rows = [
            {"순위": i + 1, label: g, "빈도": cnt}
            for i, (g, cnt) in enumerate(top_ngrams)
        ]
        result.add_table(ResultTable(
            title=f"{label} Top-{top_n}",
            dataframe=pd.DataFrame(ngram_rows),
        ))

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    if do_tfidf and len(doc_tokens) >= 2:
        tfidf_rows = _compute_tfidf(doc_tokens, top_n)
        result.add_table(ResultTable(
            title=f"TF-IDF 상위 단어 Top-{top_n}",
            dataframe=pd.DataFrame(tfidf_rows),
        ))

    # ── 워드클라우드 ─────────────────────────────────────────────────────────
    if do_wordcloud and freq:
        wc_bytes = _generate_wordcloud(
            freq,
            max_words=wc_max_words,
            width=wc_width,
            height=wc_height,
            language=language,
        )
        if wc_bytes:
            result.add_table(ResultTable(
                title="워드클라우드",
                dataframe=pd.DataFrame([{"image_bytes": wc_bytes}]),
                metadata={"type": "wordcloud_image"},
            ))

    for note in [
        f"텍스트 컬럼: {text_col}",
        f"최소 단어 길이: {min_len}자, 상위 {top_n}개",
        "불용어 처리 포함 (내장 + 사용자 정의)",
    ]:
        result.notes.append(note)
    return result


# ─────────────────────────── helpers ────────────────────────────────────────

def _tokenize(text: str, min_len: int, stopwords: set[str]) -> list[str]:
    """텍스트를 토큰 목록으로 변환 (공백 기반, 특수문자 제거)."""
    # 특수문자·숫자만으로 된 토큰 제거
    tokens = re.split(r"[\s\t\n\r]+", text.strip())
    result_tokens: list[str] = []
    for tok in tokens:
        # 구두점·특수문자 제거
        tok = re.sub(r"[^\w가-힣a-zA-Z]", "", tok)
        tok = tok.strip()
        if not tok:
            continue
        # 숫자만으로 구성된 토큰 제외
        if tok.isdigit():
            continue
        # 최소 길이 필터
        if len(tok) < min_len:
            continue
        # 불용어 제거 (소문자 비교)
        if tok.lower() in stopwords:
            continue
        result_tokens.append(tok)
    return result_tokens


def _compute_tfidf(doc_tokens: list[list[str]], top_n: int) -> list[dict]:
    """TF-IDF 계산 후 상위 단어 반환."""
    vocab = sorted({w for tokens in doc_tokens for w in tokens})
    N = len(doc_tokens)
    if N < 2 or not vocab:
        return []

    # DF (문서 빈도)
    df_count: dict[str, int] = {}
    for tokens in doc_tokens:
        for w in set(tokens):
            df_count[w] = df_count.get(w, 0) + 1

    # TF-IDF 평균
    tfidf_avg: dict[str, float] = {}
    for w in vocab:
        idf = np.log((N + 1) / (df_count.get(w, 0) + 1)) + 1.0
        tf_vals = [tokens.count(w) / len(tokens) if tokens else 0 for tokens in doc_tokens]
        tfidf_avg[w] = float(np.mean(tf_vals)) * idf

    top = sorted(tfidf_avg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {"순위": i + 1, "단어": w, "TF-IDF (평균)": round(v, 4)}
        for i, (w, v) in enumerate(top)
    ]


def _generate_wordcloud(
    freq: Counter,
    max_words: int,
    width: int,
    height: int,
    language: str,
) -> Optional[bytes]:
    """wordcloud 라이브러리로 이미지 생성 후 PNG bytes 반환."""
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 한국어 폰트 경로 탐색
        font_path = _find_font(language)

        wc_kwargs: dict = {
            "width": width,
            "height": height,
            "max_words": max_words,
            "background_color": "white",
            "collocations": False,
            "prefer_horizontal": 0.9,
        }
        if font_path:
            wc_kwargs["font_path"] = font_path

        wc = WordCloud(**wc_kwargs)
        wc.generate_from_frequencies(dict(freq.most_common(max_words)))

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning("워드클라우드 생성 실패: %s", e)
        return None


def _find_font(language: str) -> Optional[str]:
    """시스템에서 한글 지원 폰트 경로를 탐색합니다."""
    import os

    candidates = [
        # Windows 기본 한글 폰트
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        r"C:\Windows\Fonts\batang.ttc",
        r"C:\Windows\Fonts\dotum.ttc",
        # 나눔 폰트
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NanumGothicBold.ttf",
        # Linux/Mac 한글 폰트
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
    ]

    if language == "en":
        return None  # 영어는 기본 폰트 사용

    for path in candidates:
        if os.path.exists(path):
            return path
    return None

"""CSV/TSV reader for NuriStat.

Supports automatic encoding detection (via chardet), delimiter detection,
and variable metadata creation on import.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import chardet
import pandas as pd

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import (
    DelimiterDetectionError,
    EncodingDetectionError,
    FileReadError,
)

# Candidate delimiters for auto-detection, in order of likelihood
_DELIMITER_CANDIDATES = [",", "\t", ";", "|"]


def _detect_encoding(filepath: str, sample_size: int = 65536) -> str:
    """Detect file encoding using chardet.

    Parameters
    ----------
    filepath : str
        Path to the file.
    sample_size : int
        Number of bytes to sample for detection.

    Returns
    -------
    str
        Detected encoding name.

    Raises
    ------
    EncodingDetectionError
        If encoding cannot be detected.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileReadError(filepath, "파일이 존재하지 않습니다")

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileReadError(filepath, str(exc)) from exc

    if not raw:
        return "utf-8"

    # Use at most sample_size bytes for detection
    sample = raw[:sample_size]
    result = chardet.detect(sample)
    encoding = result.get("encoding")

    if encoding is None or result.get("confidence", 0.0) < 0.4:
        # Low confidence fallback: try cp949 for Korean text
        if b'\xb0' in sample or b'\xa1' in sample:
            return "cp949"
        raise EncodingDetectionError(
            filepath,
            f"인코딩 감지 실패 (confidence={result.get('confidence')})",
        )

    # Normalize common encoding aliases
    encoding = encoding.lower()
    if encoding in ("ascii", "us-ascii"):
        return "utf-8"
    if encoding in ("iso-8859-1", "iso8859-1", "latin1", "latin-1"):
        return "latin-1"
    if encoding in ("euc-kr", "euckr", "cp949"):
        return "cp949"
    if encoding == "utf-8-sig":
        return "utf-8-sig"

    return encoding


def _detect_delimiter(
    filepath: str,
    encoding: str,
    sample_lines: int = 20,
) -> str:
    """Detect delimiter by sampling the file.

    Counts occurrences of candidate delimiters in the first *sample_lines*
    non-empty lines and picks the one with the most consistent column
    structure.

    Parameters
    ----------
    filepath : str
        Path to the file.
    encoding : str
        File encoding.
    sample_lines : int
        Number of lines to sample.

    Returns
    -------
    str
        Detected delimiter character.

    Raises
    ------
    DelimiterDetectionError
        If no suitable delimiter is found.
    """
    try:
        text = Path(filepath).read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise FileReadError(filepath, f"디코딩 실패: {exc}") from exc

    if not text.strip():
        raise DelimiterDetectionError(filepath, "파일이 비어 있습니다")

    lines = [ln for ln in text.splitlines() if ln.strip()][:sample_lines]
    if not lines:  # pragma: no cover
        raise DelimiterDetectionError(filepath, "빈 파일입니다")

    best_delimiter = ","
    best_score = -1.0

    for delim in _DELIMITER_CANDIDATES:
        try:
            reader = csv.reader(StringIO("\n".join(lines)), delimiter=delim)
            rows = list(reader)
        except csv.Error:  # pragma: no cover
            continue

        if not rows:  # pragma: no cover
            continue

        col_counts = [len(row) for row in rows]
        if max(col_counts) < 2:
            # Need at least 2 columns to be a useful delimiter
            continue

        # Prefer delimiters that produce consistent column counts
        if len(set(col_counts)) == 1:
            # Perfect consistency – strong signal
            score = 100.0 + col_counts[0]
        else:
            # Score by consistency ratio
            most_common = max(set(col_counts), key=col_counts.count)
            consistency = col_counts.count(most_common) / len(col_counts)
            score = consistency * most_common

        if score > best_score:
            best_score = score
            best_delimiter = delim

    if best_score < 0:
        raise DelimiterDetectionError(filepath, "적절한 구분자를 찾을 수 없습니다")

    return best_delimiter


def read_csv(
    filepath: str,
    encoding: str = "auto",
    delimiter: str = ",",
    header: int = 0,
    **kwargs: Any,
) -> Dataset:
    """Read a CSV or TSV file into a Dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV/TSV file.
    encoding : str, default "auto"
        File encoding.  ``"auto"`` triggers chardet-based detection.
    delimiter : str, default ","
        Field delimiter.  ``"auto"`` triggers heuristic detection from
        ``",", "\\t", ";", "|"`` candidates.
    header : int, default 0
        Row index to use as column names.  ``None`` means no header.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    Dataset
        Imported dataset with auto-generated variable metadata.

    Raises
    ------
    FileReadError
        If the file cannot be read.
    EncodingDetectionError
        If encoding auto-detection fails.
    DelimiterDetectionError
        If delimiter auto-detection fails.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileReadError(filepath, "파일이 존재하지 않습니다")
    if not path.is_file():
        raise FileReadError(filepath, "경로가 파일이 아닙니다")

    # Detect encoding if requested
    if encoding == "auto":
        encoding = _detect_encoding(filepath)

    # Detect delimiter if requested
    if delimiter == "auto":
        delimiter = _detect_delimiter(filepath, encoding)

    try:
        df = pd.read_csv(
            filepath,
            encoding=encoding,
            delimiter=delimiter,
            header=header,
            **kwargs,
        )
    except UnicodeDecodeError as exc:
        raise FileReadError(
            filepath, f"인코딩 오류 ({encoding}): {exc}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise FileReadError(filepath, "파일에 데이터가 없습니다") from exc
    except pd.errors.ParserError as exc:
        raise FileReadError(filepath, f"파싱 오류: {exc}") from exc
    except OSError as exc:
        raise FileReadError(filepath, str(exc)) from exc

    source_info = {
        "file_path": str(path.absolute()),
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "encoding": encoding,
        "delimiter": delimiter,
        "header": header,
        "format": "csv",
    }

    return Dataset(data=df, name=path.stem, source_info=source_info)

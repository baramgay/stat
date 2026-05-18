"""Clipboard reader for StatWorkbench.

Reads tabular data from the system clipboard and converts it to a Dataset.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import IORError


def read_clipboard(**kwargs: Any) -> Dataset:
    """Read tabular data from the system clipboard into a Dataset.

    This function uses ``pandas.read_clipboard`` which attempts to
    automatically detect the delimiter and parse the clipboard content.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_clipboard``.

    Returns
    -------
    Dataset
        Imported dataset with auto-generated variable metadata.

    Raises
    ------
    IORError
        If the clipboard cannot be read or contains no tabular data.
    """
    try:
        df = pd.read_clipboard(**kwargs)
    except pd.errors.EmptyDataError as exc:
        raise IORError("클립보드에 데이터가 없습니다") from exc
    except pd.errors.ParserError as exc:
        raise IORError(f"클립보드 데이터 파싱 오류: {exc}") from exc
    except OSError as exc:
        raise IORError(f"클립보드 접근 오류: {exc}") from exc
    except RuntimeError as exc:
        raise IORError(f"클립보드를 읽을 수 없습니다: {exc}") from exc

    if df.empty:
        raise IORError("클립보드에 유효한 테이블 데이터가 없습니다")

    source_info = {
        "format": "clipboard",
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }

    return Dataset(data=df, name="ClipboardData", source_info=source_info)


def read_clipboard_from_qt(qt_text: str, **kwargs: Any) -> Dataset:
    """Read tabular data from a Qt clipboard text string.

    Parameters
    ----------
    qt_text : str
        The text content obtained from QApplication.clipboard().text().
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``
        (the text is parsed as tab-separated by default).

    Returns
    -------
    Dataset
        Imported dataset with auto-generated variable metadata.

    Raises
    ------
    IORError
        If the text cannot be parsed as tabular data.
    """
    from io import StringIO

    delimiter = kwargs.pop("delimiter", "\t")
    try:
        df = pd.read_csv(
            StringIO(qt_text),
            delimiter=delimiter,
            **kwargs,
        )
    except pd.errors.EmptyDataError as exc:
        raise IORError("클립보드에 데이터가 없습니다") from exc
    except pd.errors.ParserError as exc:
        raise IORError(f"클립보드 데이터 파싱 오류: {exc}") from exc

    if df.empty:
        raise IORError("클립보드에 유효한 테이블 데이터가 없습니다")

    source_info = {
        "format": "clipboard_qt",
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }

    return Dataset(data=df, name="ClipboardData", source_info=source_info)

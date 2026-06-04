"""Excel (.xlsx) reader for NuriStat.

Uses openpyxl (via pandas.read_excel) to import Excel workbooks.
Supports sheet selection by name or index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import FileReadError


def read_excel(
    filepath: str,
    sheet_name: str | int = 0,
    header: int = 0,
    **kwargs: Any,
) -> Dataset:
    """Read an Excel (.xlsx) file into a Dataset.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    sheet_name : str or int, default 0
        Sheet to read.  Can be a zero-based index or the sheet name.
    header : int, default 0
        Row index to use as column names.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_excel``.

    Returns
    -------
    Dataset
        Imported dataset with auto-generated variable metadata.

    Raises
    ------
    FileReadError
        If the file cannot be read.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileReadError(filepath, "파일이 존재하지 않습니다")
    if not path.is_file():
        raise FileReadError(filepath, "경로가 파일이 아닙니다")

    try:
        df = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=header,
            engine="openpyxl",
            **kwargs,
        )
    except FileNotFoundError as exc:
        raise FileReadError(filepath, "파일을 찾을 수 없습니다") from exc
    except PermissionError as exc:
        raise FileReadError(filepath, "파일 접근 권한이 없습니다") from exc
    except ValueError as exc:
        raise FileReadError(filepath, f"Excel 파싱 오류: {exc}") from exc
    except OSError as exc:
        raise FileReadError(filepath, str(exc)) from exc

    # pandas may return a dict of DataFrames when sheet_name is a list
    if isinstance(df, dict):
        if not df:
            raise FileReadError(filepath, "시트가 비어 있습니다")
        # Use the first sheet
        df = next(iter(df.values()))

    source_info = {
        "file_path": str(path.absolute()),
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "sheet_name": sheet_name,
        "header": header,
        "format": "xlsx",
    }

    return Dataset(data=df, name=path.stem, source_info=source_info)

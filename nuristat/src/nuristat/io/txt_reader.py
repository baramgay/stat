"""TXT reader with delimiter detection for NuriStat.

Reads plain-text files that are not explicitly CSV/TSV by sampling the
first chunk of the file to heuristically determine the delimiter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import FileReadError
from nuristat.io.csv_reader import _detect_delimiter, _detect_encoding


def read_txt(
    filepath: str,
    encoding: str = "auto",
    delimiter: str = "auto",
    header: int = 0,
    **kwargs: Any,
) -> Dataset:
    """Read a plain-text delimited file into a Dataset.

    Parameters
    ----------
    filepath : str
        Path to the text file.
    encoding : str, default "auto"
        File encoding.  ``"auto"`` triggers chardet-based detection.
    delimiter : str, default "auto"
        Field delimiter.  ``"auto"`` triggers heuristic sampling from
        the first 5 KB of the file.
    header : int, default 0
        Row index to use as column names.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``
        (the file is ultimately read as a delimited text file).

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

    # Detect encoding
    if encoding == "auto":
        encoding = _detect_encoding(filepath)

    # Detect delimiter
    if delimiter == "auto":
        delimiter = _detect_delimiter(filepath, encoding)

    # Delegate to pandas read_csv with detected parameters
    import pandas as pd

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
        "format": "txt",
    }

    return Dataset(data=df, name=path.stem, source_info=source_info)

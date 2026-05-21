"""SPSS .sav 파일 쓰기.

pyreadstat을 사용하여 DataFrame을 SPSS .sav 파일로 저장합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import ImportError as SWBImportError


def write_sav(dataset: Dataset, path: str) -> None:
    """Dataset을 SPSS .sav 파일로 저장합니다.

    Args:
        dataset: 저장할 데이터셋
        path: 저장 경로

    Raises:
        SWBImportError: pyreadstat이 설치되지 않은 경우
        ValueError: 데이터가 없는 경우
    """
    try:
        import pyreadstat
    except ImportError:
        raise SWBImportError(
            "pyreadstat이 설치되지 않았습니다. "
            "'pip install pyreadstat'로 설치하세요."
        )

    path = Path(path)

    if dataset is None or dataset.data.empty:
        raise ValueError("저장할 데이터가 없습니다.")

    df = dataset.data.copy()

    # 변수 레이블
    column_labels = {}
    for var_name, var_meta in dataset.variables.items():
        if var_name in df.columns:
            column_labels[var_name] = var_meta.label or var_name

    # 값 레이블
    value_labels = {}
    for var_name, var_meta in dataset.variables.items():
        if var_name in df.columns and var_meta.value_labels:
            value_labels[var_name] = var_meta.value_labels

    # 측정 척도 매핑 (pyreadstat: variable_measure)
    measure_map = {
        "SCALE": "scale",
        "ORDINAL": "ordinal",
        "NOMINAL": "nominal",
    }
    variable_measure: dict[str, str] = {}
    for var_name, var_meta in dataset.variables.items():
        if var_name in df.columns:
            variable_measure[var_name] = measure_map.get(
                var_meta.measure.value.upper(), "nominal"
            )

    # 저장 (pyreadstat 1.x API)
    pyreadstat.write_sav(
        df,
        str(path),
        column_labels=column_labels if column_labels else None,
        variable_value_labels=value_labels if value_labels else None,
        variable_measure=variable_measure if variable_measure else None,
    )

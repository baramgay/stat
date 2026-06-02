"""SPSS .sav 파일 읽기.

pyreadstat을 사용하여 SPSS .sav 파일을 읽고 Dataset으로 변환합니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import ImportError as SWBImportError
from statworkbench.core.variable import MeasureType, StorageType, VariableMeta


def read_sav(path: str | Path) -> Dataset:
    """SPSS .sav 파일을 읽어 Dataset으로 변환합니다.

    Args:
        path: .sav 파일 경로

    Returns:
        Dataset 객체

    Raises:
        DataError: 파일 읽기 실패 시
    """
    path = Path(path)

    if not path.exists():
        raise SWBImportError(f"파일을 찾을 수 없습니다: {path}")

    if not path.suffix.lower() == ".sav":
        raise SWBImportError(f"SPSS .sav 파일이 아닙니다: {path.suffix}")

    try:
        import pyreadstat

        df, meta = pyreadstat.read_sav(str(path))

        # 변수 메타데이터 변환
        _MEASURE_MAP = {
            "scale":   MeasureType.SCALE,
            "ordinal": MeasureType.ORDINAL,
            "nominal": MeasureType.NOMINAL,
            "unknown": MeasureType.NOMINAL,
        }
        variables = {}
        for i, col in enumerate(df.columns):
            # 측정 척도: 파일에 저장된 값 우선, 없으면 dtype 추정
            if meta.variable_measure and col in meta.variable_measure:
                measure = _MEASURE_MAP.get(
                    meta.variable_measure[col].lower(), MeasureType.NOMINAL
                )
            else:
                measure = _guess_measure_type(df[col])

            var_meta = VariableMeta(
                name=col,
                label=meta.column_labels[i] if meta.column_labels and i < len(meta.column_labels) else col,
                storage_type=_guess_storage_type(df[col]),
                measure=measure,
            )

            # 값 레이블 (dict 키 타입 불변성으로 인한 mypy 경계 — 런타임 정상)
            if meta.variable_value_labels and col in meta.variable_value_labels:
                var_meta.value_labels = meta.variable_value_labels[col]  # type: ignore[assignment]

            # 결측치 정의
            if meta.missing_ranges and col in meta.missing_ranges:
                var_meta.missing_values = meta.missing_ranges[col]

            variables[col] = var_meta

        dataset = Dataset(
            data=df,
            variables=variables,
            name=path.stem,
            description=f"SPSS 파일: {path.name}",
        )

        return dataset

    except ImportError:
        raise SWBImportError(
            "pyreadstat이 설치되지 않았습니다. "
            "'pip install pyreadstat'로 설치하세요."
        )
    except Exception as exc:
        raise SWBImportError(f"SPSS 파일 읽기 실패: {exc}") from exc


def _guess_storage_type(series: pd.Series) -> StorageType:
    """시리즈 타입에 맞는 StorageType 추정.

    pyreadstat은 SPSS 정수 변수도 float64로 읽어오는 경우가 있다.
    유효 값이 모두 정수인 경우 INTEGER로 처리해 NOMINAL/ORDINAL 호환성을 유지한다.
    """
    dtype = series.dtype

    if pd.api.types.is_integer_dtype(dtype):
        return StorageType.INTEGER
    elif pd.api.types.is_float_dtype(dtype):
        non_null = series.dropna()
        if len(non_null) > 0 and non_null.apply(lambda v: float(v).is_integer()).all():
            return StorageType.INTEGER
        return StorageType.FLOAT
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return StorageType.DATETIME
    else:
        return StorageType.STRING


def _guess_measure_type(series: pd.Series) -> MeasureType:
    """시리즈 타입에 맞는 MeasureType 추정.

    SPSS 29 기본 규칙:
      - 수치형 → SCALE
      - 문자형 / 범주형 → NOMINAL
    ORDINAL/BINARY는 사용자가 Variable View에서 직접 지정한다.
    """
    dtype = series.dtype
    if pd.api.types.is_numeric_dtype(dtype):
        return MeasureType.SCALE
    return MeasureType.NOMINAL

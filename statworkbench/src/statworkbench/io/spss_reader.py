"""SPSS .sav 파일 읽기.

pyreadstat을 사용하여 SPSS .sav 파일을 읽고 Dataset으로 변환합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta, StorageType, MeasureType
from statworkbench.core.exceptions import ImportError as SWBImportError


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
        variables = {}
        for i, col in enumerate(df.columns):
            var_meta = VariableMeta(
                name=col,
                label=meta.column_labels[i] if meta.column_labels and i < len(meta.column_labels) else col,
                storage_type=_guess_storage_type(df[col]),
                measure=_guess_measure_type(df[col]),
            )
            
            # 값 레이블
            if meta.variable_value_labels and col in meta.variable_value_labels:
                var_meta.value_labels = meta.variable_value_labels[col]
            
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
    """시리즈 타입에 맞는 StorageType 추정."""
    dtype = series.dtype
    
    if pd.api.types.is_integer_dtype(dtype):
        return StorageType.INTEGER
    elif pd.api.types.is_float_dtype(dtype):
        return StorageType.FLOAT
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return StorageType.DATETIME
    else:
        return StorageType.STRING


def _guess_measure_type(series: pd.Series) -> MeasureType:
    """시리즈 타입에 맞는 MeasureType 추정."""
    dtype = series.dtype
    
    if pd.api.types.is_numeric_dtype(dtype):
        return MeasureType.SCALE
    elif pd.api.types.is_categorical_dtype(dtype):
        return MeasureType.ORDINAL
    else:
        return MeasureType.NOMINAL

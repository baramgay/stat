"""SPSS 스타일 분석 다이얼로그 공통 유틸리티.

변수 측정 척도 기반 필터링 — SPSS Variable View 의 Measure 속성을 기준으로
분석에 적합한 변수 목록을 반환한다.
"""

from __future__ import annotations

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, StorageType


# ── 변수 목록 필터 ─────────────────────────────────────────────────────────


def scale_vars(dataset: Dataset) -> list[str]:
    """척도(Scale) 변수 목록 — 연속형 측정에 사용.

    MeasureType.SCALE 우선; 메타데이터 없으면 StorageType.FLOAT/INTEGER 로 대체.
    """
    result = []
    vars_source = (
        list(dataset.variables.keys())
        if dataset.variables
        else list(dataset.data.columns)
    )
    for v in vars_source:
        meta = dataset.variables.get(v)
        if meta is not None:
            if meta.measure == MeasureType.SCALE:
                result.append(v)
        elif v in dataset.data.columns:
            import pandas as pd
            if pd.api.types.is_numeric_dtype(dataset.data[v]):
                result.append(v)
    return result


def numeric_vars(dataset: Dataset) -> list[str]:
    """수치형 저장 타입 변수 목록 — StorageType.FLOAT 또는 INTEGER.

    MeasureType 무관. scale_vars 보다 느슨한 기준.
    """
    result = []
    vars_source = (
        list(dataset.variables.keys())
        if dataset.variables
        else list(dataset.data.columns)
    )
    for v in vars_source:
        meta = dataset.variables.get(v)
        if meta is not None:
            if meta.storage_type in (StorageType.FLOAT, StorageType.INTEGER):
                result.append(v)
        elif v in dataset.data.columns:
            import pandas as pd
            if pd.api.types.is_numeric_dtype(dataset.data[v]):
                result.append(v)
    return result


def categorical_vars(dataset: Dataset) -> list[str]:
    """범주형 변수 목록 — 명목(Nominal) 또는 순서형(Ordinal).

    그룹 변수, 요인 변수 선택 시 사용.
    """
    result = []
    vars_source = (
        list(dataset.variables.keys())
        if dataset.variables
        else list(dataset.data.columns)
    )
    for v in vars_source:
        meta = dataset.variables.get(v)
        if meta is not None:
            if meta.measure in (MeasureType.NOMINAL, MeasureType.ORDINAL, MeasureType.BINARY):
                result.append(v)
        elif v in dataset.data.columns:
            import pandas as pd
            col = dataset.data[v]
            if not pd.api.types.is_numeric_dtype(col) or col.nunique() <= 10:
                result.append(v)
    return result


def all_vars(dataset: Dataset) -> list[str]:
    """전체 변수 목록 — dataset.variables 순서 우선."""
    if dataset.variables:
        return list(dataset.variables.keys())
    return list(dataset.data.columns)


def ordinal_or_higher_vars(dataset: Dataset) -> list[str]:
    """순서형 이상 변수 — Ordinal, Scale. 비모수 상관(Spearman/Kendall) 용."""
    result = []
    for v in all_vars(dataset):
        meta = dataset.variables.get(v)
        if meta is not None:
            if meta.measure in (MeasureType.ORDINAL, MeasureType.SCALE):
                result.append(v)
        elif v in dataset.data.columns:
            import pandas as pd
            if pd.api.types.is_numeric_dtype(dataset.data[v]):
                result.append(v)
    return result


# ── 표시 이름 ──────────────────────────────────────────────────────────────


def display_label(dataset: Dataset, var_name: str) -> str:
    """변수 표시 이름: '라벨 (변수명)' 형식, 라벨 없으면 변수명만."""
    meta = dataset.variables.get(var_name)
    if meta and meta.label:
        return f"{meta.label} ({var_name})"
    return var_name


def var_from_display(display: str) -> str:
    """display_label 의 역방향 — '라벨 (변수명)' 에서 변수명 추출."""
    if display.endswith(")") and " (" in display:
        return display.rsplit(" (", 1)[-1][:-1]
    return display


# ── 값 레이블 해석 ─────────────────────────────────────────────────────────


def apply_value_labels(dataset: Dataset, var_name: str, series) -> object:
    """Series 값에 값 레이블 적용 — 레이블 없는 값은 원래 값 유지."""
    meta = dataset.variables.get(var_name)
    if meta is None or not meta.value_labels:
        return series
    import pandas as pd
    labels = meta.value_labels

    def _label(v):
        if pd.isna(v):
            return v
        key = int(v) if isinstance(v, float) and v == int(v) else v
        return labels.get(key, labels.get(str(key), v))

    return series.map(_label)


# ── 측정 척도 아이콘 ───────────────────────────────────────────────────────

_MEASURE_ICON = {
    MeasureType.SCALE:     "📏",
    MeasureType.ORDINAL:   "🔢",
    MeasureType.NOMINAL:   "🏷️",
    MeasureType.BINARY:    "⚖️",
    MeasureType.DATE_TIME: "📅",
    MeasureType.TEXT:      "📝",
}


def measure_icon(dataset: Dataset, var_name: str) -> str:
    """변수의 측정 척도 아이콘 반환."""
    meta = dataset.variables.get(var_name)
    if meta:
        return _MEASURE_ICON.get(meta.measure, "")
    return ""

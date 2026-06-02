# SPEC.md — StatWorkbench

> HERMES.md 기반 상세 명세. 모든 에이전트는 이 문서를 single source of truth로 따른다.

---

## 1. 프로젝트 개요

**제품명:** StatWorkbench  
**유형:** 데스크톱 통계 패키지 (PySide6 기반)  
**목표:** SPSS/MedCalc 스타일의 메뉴 기반 통계 분석 도구

---

## 2. 기술 스택

| 영역 | 기술 |
|---|---|
| GUI | PySide6 |
| 데이터 처리 | pandas, numpy |
| 통계 | scipy, statsmodels |
| Excel | openpyxl |
| 저장 | ZIP 기반 .swb + Parquet/CSV + JSON |
| 메타데이터 검증 | pydantic |
| 테스트 | pytest, pytest-qt |
| 패키징 | pyproject.toml, hatchling |

---

## 3. 프로젝트 구조

```
statworkbench/
├── pyproject.toml
├── README.md
├── HERMES.md
├── src/statworkbench/
│   ├── __init__.py
│   ├── app.py              # QApplication 진입점
│   ├── main.py             # CLI 진입점
│   ├── core/
│   │   ├── __init__.py
│   │   ├── typing.py       # enums: StorageType, MeasureType, Role, MissingPolicy
│   │   ├── variable.py     # VariableMeta (pydantic)
│   │   ├── dataset.py      # Dataset 클래스
│   │   ├── project.py      # Project 저장/로드
│   │   ├── validation.py   # 변수명 검증, 범위 검증
│   │   ├── exceptions.py   # 커스텀 예외
│   │   └── audit.py        # 감사 로그
│   ├── io/
│   │   ├── __init__.py
│   │   ├── csv_reader.py   # CSV/TSV 임포트
│   │   ├── txt_reader.py   # 구분자 감지 TXT 임포트
│   │   ├── excel_reader.py # xlsx 임포트
│   │   ├── clipboard_reader.py
│   │   ├── import_wizard.py # Import Wizard 로직
│   │   ├── project_store.py # .swb 저장/불러오기
│   │   └── exporters.py   # 결과 납품포트
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── base.py         # AnalysisPlugin Protocol
│   │   ├── registry.py     # AnalysisRegistry
│   │   ├── result.py       # AnalysisResult, ResultTable
│   │   ├── formatting.py   # p-value, 숫자 포맷팅
│   │   ├── assumptions.py  # 결측 처리, 가정 검정
│   │   ├── descriptive.py  # 기술통계
│   │   ├── frequencies.py  # 빈도분석
│   │   ├── normality.py    # 정규성 검정
│   │   ├── crosstab.py     # 교차분석
│   │   ├── ttests.py       # 독립/대응 t검정
│   │   ├── anova.py        # 일원분산분석
│   │   ├── nonparametric.py # Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman
│   │   ├── correlation.py  # 상관분석
│   │   └── regression.py   # 선형회귀
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py  # 메인 윈도우
│   │   ├── menu_builder.py # 메뉴 생성
│   │   ├── data_view.py    # Data View (스프레드시트)
│   │   ├── variable_view.py # Variable View
│   │   ├── output_view.py  # Output View (결과 표시)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── dataframe_table_model.py
│   │   │   └── variable_table_model.py
│   │   └── dialogs/
│   │       ├── __init__.py
│   │       ├── import_dialog.py
│   │       ├── variable_editor.py
│   │       ├── analysis_dialog_base.py
│   │       ├── descriptive_dialog.py
│   │       ├── frequency_dialog.py
│   │       ├── crosstab_dialog.py
│   │       ├── ttest_dialog.py
│   │       ├── anova_dialog.py
│   │       ├── correlation_dialog.py
│   │       └── regression_dialog.py
│   ├── syntax/
│   │   ├── __init__.py
│   │   ├── command.py      # SyntaxCommand 모델
│   │   └── writer.py       # syntax 로그 작성
│   ├── viz_bridge/
│   │   ├── __init__.py
│   │   └── spec.py
│   └── resources/
│       └── icons/
├── tests/
│   ├── conftest.py
│   ├── core/
│   │   ├── test_variable.py
│   │   ├── test_dataset.py
│   │   ├── test_validation.py
│   │   └── test_project.py
│   ├── io/
│   │   ├── test_csv_reader.py
│   │   ├── test_excel_reader.py
│   │   └── test_project_store.py
│   ├── analysis/
│   │   ├── test_descriptive.py
│   │   ├── test_frequencies.py
│   │   ├── test_normality.py
│   │   ├── test_crosstab.py
│   │   ├── test_ttests.py
│   │   ├── test_anova.py
│   │   ├── test_nonparametric.py
│   │   ├── test_correlation.py
│   │   └── test_regression.py
│   └── fixtures/
│       ├── sample_clinical.csv
│       ├── sample_survey.csv
│       └── sample_regression.csv
├── examples/
└── scripts/
    └── make_sample_data.py
```

---

## 4. 인터페이스 계약

### 4.1 Enums (core/typing.py)

```python
from enum import Enum

class StorageType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"

class MeasureType(str, Enum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    SCALE = "scale"
    BINARY = "binary"
    DATE_TIME = "date_time"
    TEXT = "text"

class Role(str, Enum):
    INPUT = "input"
    TARGET = "target"
    WEIGHT = "weight"
    ID = "id"
    SPLIT = "split"
    FREQUENCY = "frequency"
    NONE = "none"

class MissingPolicy(str, Enum):
    LISTWISE = "listwise"
    PAIRWISE = "pairwise"
    ANALYSIS_DEFAULT = "analysis_default"
    INCLUDE_MISSING = "include_missing"
    USER_ONLY = "user_only"
    SYSTEM_ONLY = "system_only"
```

### 4.2 VariableMeta (core/variable.py)

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional
from .typing import StorageType, MeasureType, Role

class VariableMeta(BaseModel):
    name: str
    label: str = ""
    storage_type: StorageType = StorageType.FLOAT
    measure: MeasureType = MeasureType.SCALE
    role: Role = Role.INPUT
    width: int = 8
    decimals: int = 2
    value_labels: dict = Field(default_factory=dict)
    missing_values: list = Field(default_factory=list)
    unit: str = ""
    allowed_min: Optional[float] = None
    allowed_max: Optional[float] = None
    format_pattern: str = ""
    datetime_format: str = ""
    description: str = ""
    source_column: str = ""
    derived: bool = False
    formula: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("변수명은 비어있을 수 없습니다")
        v = v.strip().replace(" ", "_")
        return v
```

### 4.3 Dataset (core/dataset.py)

```python
import pandas as pd
from typing import Dict, Optional, Any
from .variable import VariableMeta
from .audit import AuditLog

class Dataset:
    def __init__(self, data: pd.DataFrame, name: str = "Dataset"):
        self._data = data.copy()
        self.name = name
        self.variables: Dict[str, VariableMeta] = {}
        self.source_info: dict = {}
        self._dirty = False
        self.audit_log = AuditLog()
        self.syntax_log: list[str] = []
        self._init_variables()

    def _init_variables(self) -> None:
        for col in self._data.columns:
            self.variables[col] = VariableMeta(
                name=col,
                source_column=col,
            )
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        self._data = value
        self._dirty = True
    
    @property
    def shape(self) -> tuple:
        return self._data.shape
    
    @property
    def n_rows(self) -> int:
        return len(self._data)
    
    @property
    def n_vars(self) -> int:
        return len(self._data.columns)
    
    def get_variable(self, name: str) -> VariableMeta:
        if name not in self.variables:
            raise KeyError(f"변수를 찾을 수 없습니다: {name}")
        return self.variables[name]
    
    def rename_variable(self, old_name: str, new_name: str) -> None:
        if old_name not in self._data.columns:
            raise KeyError(f"변수를 찾을 수 없습니다: {old_name}")
        self._data = self._data.rename(columns={old_name: new_name})
        meta = self.variables.pop(old_name)
        meta.name = new_name
        meta.updated_at = datetime.now()
        self.variables[new_name] = meta
        self._dirty = True
    
    def to_dict(self) -> dict:
        return {
            "data": self._data.to_dict(),
            "variables": {k: v.model_dump() for k, v in self.variables.items()},
            "name": self.name,
            "source_info": self.source_info,
        }
```

### 4.4 AnalysisPlugin Protocol (analysis/base.py)

```python
from typing import Protocol, runtime_checkable
from ..core.dataset import Dataset
from .result import AnalysisResult

@runtime_checkable
class AnalysisPlugin(Protocol):
    id: str
    name: str
    category: str
    description: str
    variable_requirements: list[dict]

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        ...

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        ...
```

### 4.5 AnalysisResult (analysis/result.py)

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any
import pandas as pd

class ResultTable(BaseModel):
    title: str
    dataframe: pd.DataFrame
    footnotes: list[str] = []
    format_rules: dict = {}
    export_options: dict = {}
    
    class Config:
        arbitrary_types_allowed = True

class AnalysisResult(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
    spec: dict = {}
    notes: list[str] = []
    warnings: list[str] = []
    tables: list[ResultTable] = []
    text_blocks: list[str] = []
    assumptions: list[ResultTable] = []
    diagnostics: list[ResultTable] = []
    figures: list[Any] = []
    syntax: str = ""
    metadata: dict = {}
    
    class Config:
        arbitrary_types_allowed = True
```

---

## 5. 구현 우선순위

### Phase 0: 프로젝트 기반
- pyproject.toml, 패키지 구조
- 기본 main window (빈 창)
- pytest 설정
- 샘플 데이터

### Phase 1: 데이터/변수 모델
- VariableMeta pydantic 모델
- Dataset 클래스
- 변수명 검증
- 타입/척도 enum

### Phase 2: 임포트
- CSV/TXT/Excel import
- 인코딩 감지
- 변수 타입 추론
- Import Wizard

### Phase 3: Data/Variable View
- 스프레드시트 UI
- 편집 기능
- undo/redo 구조

### Phase 4: 기본 분석 엔진
- AnalysisPlugin, Registry, Result
- formatting, assumptions
- descriptive, frequencies, normality, crosstab

### Phase 5: 분석 구현
- ttests, anova, nonparametric
- correlation, regression

### Phase 6: 메뉴 UI
- Analysis Dialogs
- Output View
- Syntax log

---

## 6. 구현 시 주의사항

1. 모든 함수에 type hints 사용
2. 통계 함수는 side effect 없음 (순수 함수)
3. 분석은 Dataset을 직접 변경하지 않음
4. 결측 처리는 assumptions.py의 공통 유틸 사용
5. p-value는 HERMES.md 표기 규칙 준수
6. 모든 분석 결과에 Case Processing Summary 포함
7. 테스트는 각 분석당 최소 3개

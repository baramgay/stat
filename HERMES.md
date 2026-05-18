# HERMES.md — 메뉴 기반 스프레드시트형 통계 패키지 개발 지시서

> **프로젝트 코드명:** `StatWorkbench`  
> **목표 제품:** SPSS/MedCalc처럼 데이터를 스프레드시트 방식으로 입력·관리하고, 메뉴 기반으로 통계분석을 실행할 수 있는 데스크톱 통계 패키지  
> **개발 에이전트:** Hermes Agent  
> **작성일:** 2026-05-17  
> **핵심 원칙:** 단순 표 계산기가 아니라, **변수 메타데이터를 중심으로 분석 가능성을 판단하고 올바른 분석 절차를 안내하는 통계 워크벤치**를 만든다.

---

## 0. Hermes Agent에게 주는 최상위 지시

이 문서는 프로젝트 루트에 위치하는 `HERMES.md`로 사용한다. Hermes Agent는 작업을 시작할 때 이 파일을 프로젝트의 기준 문서로 간주하고, 아래 원칙을 항상 따른다.

### 0.1 작업 태도

1. 사용자가 매 단계 확인하지 않아도, 합리적인 기본값을 정해 자율적으로 구현한다.
2. 단순히 UI 껍데기만 만들지 말고, **데이터 모델 → 변수 관리 → 분석 엔진 → UI 연결 → 테스트** 순서로 실제 동작하는 기능을 완성한다.
3. 불확실한 요구사항은 통계 패키지의 일반적인 관례와 재현성 원칙에 따라 결정한다.
4. 모든 핵심 기능은 테스트를 포함한다.
5. “나중에 구현”이라는 빈 껍데기 코드를 남기지 않는다. 아직 구현하지 않는 기능은 명확히 `NotImplemented`나 비활성 메뉴로 처리하고, 로드맵에 남긴다.
6. 분석 결과는 항상 재현 가능해야 하며, 동일 데이터와 동일 옵션에서 동일 결과가 나와야 한다.
7. 사용자의 데이터가 의료·임상·연구 데이터일 수 있으므로, 기본 저장·로그 정책은 로컬 우선, 개인정보 최소 노출 원칙을 따른다.

### 0.2 구현 우선순위

Hermes Agent는 다음 순서를 우선한다.

1. 프로젝트 골격, 패키징, 테스트 실행 환경
2. 데이터셋/변수 메타데이터 도메인 모델
3. CSV/TXT/Excel 계열 데이터 임포트
4. Data View / Variable View 스프레드시트 UI
5. 기본 분석 엔진
6. 메뉴 기반 분석 실행
7. 결과 출력 뷰
8. 프로젝트 저장/불러오기
9. 고급 분석 확장
10. Python 기반 시각화 브리지

---

## 1. 제품 비전

### 1.1 한 문장 정의

`StatWorkbench`는 연구자, 임상의, 데이터 분석가가 코드를 직접 작성하지 않고도 데이터를 불러오고 변수 특성을 정의한 뒤, 변수의 척도와 역할에 맞는 통계분석을 메뉴 방식으로 실행할 수 있는 데스크톱 통계 패키지다.

### 1.2 핵심 차별점

이 제품은 단순 스프레드시트가 아니다.

일반 스프레드시트는 셀 중심이다.  
StatWorkbench는 **변수 중심**이다.

즉, 각 열은 단순 문자열/숫자 열이 아니라 다음 정보를 가진 통계 변수다.

- 변수명
- 변수 라벨
- 데이터 타입
- 측정 척도
- 값 라벨
- 결측값 규칙
- 분석 역할
- 단위
- 허용 범위
- 표시 형식
- 파생 변수 여부
- 원본 데이터 출처
- 분석 적합성 정보

분석 메뉴는 이 변수 메타데이터를 이용해 사용자가 선택 가능한 분석과 불가능한 분석을 자동 판단해야 한다.

### 1.3 장기 비전

최종적으로는 다음 수준을 목표로 한다.

1. SPSS와 유사한 Data View / Variable View 구조
2. MedCalc처럼 임상·의학 통계 분석에 강한 메뉴
3. Python 생태계와 연결되는 확장 가능한 분석·시각화 엔진
4. 결과 테이블, 분석 로그, 재현 가능한 syntax 기록
5. 연구 보고서 작성에 바로 사용할 수 있는 출력 포맷
6. GUI 사용자와 Python 고급 사용자 모두를 위한 이중 인터페이스

---

## 2. MVP 범위

### 2.1 MVP에서 반드시 구현할 기능

MVP는 “분석 가능한 첫 제품”이어야 한다. 단순 UI 목업은 실패로 간주한다.

#### 데이터 입력

- CSV 파일 불러오기
- TXT 파일 불러오기
- TSV 파일 불러오기
- Excel `.xlsx` 불러오기
- 클립보드 붙여넣기
- 빈 데이터셋 생성
- 셀 직접 입력
- 행/열 추가 및 삭제
- 데이터 타입 자동 추론
- 인코딩 자동 감지 또는 선택
- 구분자 자동 감지 또는 선택
- 첫 행을 변수명으로 사용할지 선택

#### 변수 관리

- Variable View 제공
- 변수명 수정
- 변수 라벨 설정
- 데이터 타입 설정
- 측정 척도 설정
- 값 라벨 설정
- 결측값 규칙 설정
- 표시 소수점 자리 설정
- 변수 역할 설정
- 분석 가능 여부 검증

#### 분석 기능

MVP에서는 다음 분석을 반드시 메뉴로 제공한다.

1. 빈도분석
2. 기술통계
3. 정규성 검정
4. 교차분석
5. 독립표본 t 검정
6. 대응표본 t 검정
7. 일원분산분석
8. 상관분석
9. 단순/다중 선형회귀
10. 비모수 기본 분석
    - Mann-Whitney U
    - Wilcoxon signed-rank
    - Kruskal-Wallis
    - Friedman

#### 결과 출력

- Output View 제공
- 분석별 결과 테이블 출력
- p-value, 통계량, 자유도, 신뢰구간 표시
- 분석 옵션과 사용 변수 기록
- 결과를 HTML/Markdown/CSV로 내보내기
- 분석 로그 저장

#### 프로젝트 저장

- 자체 프로젝트 파일 저장
- 데이터와 변수 메타데이터 함께 저장
- 결과 출력 문서 저장
- 분석 syntax 로그 저장

### 2.2 MVP에서 명시적으로 제외하는 기능

다음은 MVP에서 제외하되, 구조적으로 나중에 추가 가능하게 설계한다.

- 복잡한 그래프 UI
- 대시보드 빌더
- 클라우드 협업
- 실시간 다중 사용자 편집
- 대규모 분산 처리
- Bayesian 분석
- 구조방정식 모형
- 자동 논문 작성
- LIMS/EMR 직접 연동
- SAS/SPSS 파일 완전 호환

---

## 3. 권장 기술 스택

### 3.1 기본 선택

Python 중심으로 구현한다.

| 영역 | 권장 기술 |
|---|---|
| GUI | PySide6 / Qt |
| 테이블 모델 | Qt Model/View + pandas-backed model |
| 데이터 처리 | pandas, numpy |
| 통계 | scipy, statsmodels |
| Excel | openpyxl |
| 저장 포맷 | ZIP 기반 프로젝트 파일 + Parquet/CSV + JSON metadata |
| 메타데이터 검증 | pydantic |
| 테스트 | pytest, pytest-qt |
| 패키징 | pyproject.toml, hatchling 또는 setuptools |
| 데스크톱 배포 | PyInstaller 또는 Briefcase |
| 향후 시각화 | matplotlib, plotly, seaborn은 선택적 연동. 기본 브리지는 Python script runner |

### 3.2 Python + PySide6를 기본안으로 정하는 이유

1. 향후 시각화가 Python 기반이므로 통합 비용이 낮다.
2. pandas/scipy/statsmodels 생태계를 직접 사용할 수 있다.
3. PySide6는 Data View/Variable View 같은 데스크톱형 UI에 적합하다.
4. 통계 계산 엔진과 UI를 같은 언어로 시작할 수 있어 MVP 속도가 빠르다.
5. 분석 엔진을 나중에 CLI/API로 분리하기 쉽다.

### 3.3 대안 아키텍처

추후 UI 규모가 커지면 다음 구조로 전환할 수 있다.

- Frontend: Tauri 또는 Electron
- Backend: Python FastAPI
- Analysis Engine: 독립 Python 패키지
- Visualization Engine: Python subprocess 또는 local API

단, 초기 구현에서는 복잡도를 줄이기 위해 PySide6 단일 데스크톱 앱으로 시작한다.

---

## 4. 프로젝트 구조

Hermes Agent는 아래 구조를 기본으로 생성한다.

```text
statworkbench/
├── HERMES.md
├── README.md
├── pyproject.toml
├── src/
│   └── statworkbench/
│       ├── __init__.py
│       ├── app.py
│       ├── main.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── variable.py
│       │   ├── project.py
│       │   ├── typing.py
│       │   ├── validation.py
│       │   ├── exceptions.py
│       │   └── audit.py
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── import_wizard.py
│       │   ├── csv_reader.py
│       │   ├── txt_reader.py
│       │   ├── excel_reader.py
│       │   ├── clipboard_reader.py
│       │   ├── project_store.py
│       │   └── exporters.py
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── result.py
│       │   ├── formatting.py
│       │   ├── assumptions.py
│       │   ├── descriptive.py
│       │   ├── frequencies.py
│       │   ├── normality.py
│       │   ├── crosstab.py
│       │   ├── ttests.py
│       │   ├── anova.py
│       │   ├── nonparametric.py
│       │   ├── correlation.py
│       │   ├── regression.py
│       │   └── reliability.py
│       │
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── main_window.py
│       │   ├── data_view.py
│       │   ├── variable_view.py
│       │   ├── output_view.py
│       │   ├── menu_builder.py
│       │   ├── dialogs/
│       │   │   ├── __init__.py
│       │   │   ├── import_dialog.py
│       │   │   ├── variable_editor.py
│       │   │   ├── analysis_dialog_base.py
│       │   │   ├── descriptive_dialog.py
│       │   │   ├── frequency_dialog.py
│       │   │   ├── crosstab_dialog.py
│       │   │   ├── ttest_dialog.py
│       │   │   ├── anova_dialog.py
│       │   │   ├── correlation_dialog.py
│       │   │   └── regression_dialog.py
│       │   └── models/
│       │       ├── __init__.py
│       │       ├── dataframe_table_model.py
│       │       ├── variable_table_model.py
│       │       └── output_tree_model.py
│       │
│       ├── syntax/
│       │   ├── __init__.py
│       │   ├── command.py
│       │   ├── parser.py
│       │   └── writer.py
│       │
│       ├── viz_bridge/
│       │   ├── __init__.py
│       │   ├── spec.py
│       │   ├── python_runner.py
│       │   └── registry.py
│       │
│       └── resources/
│           ├── icons/
│           └── translations/
│
├── tests/
│   ├── conftest.py
│   ├── core/
│   ├── io/
│   ├── analysis/
│   ├── syntax/
│   └── ui/
│
├── examples/
│   ├── sample_clinical.csv
│   ├── sample_survey.csv
│   └── sample_regression.csv
│
├── docs/
│   ├── architecture.md
│   ├── variable_model.md
│   ├── analysis_registry.md
│   ├── statistical_validation.md
│   └── user_manual_mvp.md
│
└── scripts/
    ├── run_app.py
    ├── make_sample_data.py
    └── package_app.py
```

---

## 5. 도메인 모델

### 5.1 핵심 개념

제품의 중심은 `Dataset`과 `VariableMeta`다.

`Dataset`은 실제 데이터 프레임과 변수 메타데이터 컬렉션을 함께 가진다.

```text
Dataset
├── data: pandas.DataFrame
├── variables: dict[str, VariableMeta]
├── row_labels: optional
├── source_info
├── dirty_state
├── audit_log
└── syntax_log
```

### 5.2 VariableMeta

변수 메타데이터는 다음 필드를 가진다.

| 필드 | 타입 | 설명 |
|---|---|---|
| name | str | 실제 변수명. 고유해야 한다. |
| label | str | 사용자에게 보이는 설명. |
| storage_type | enum | integer, float, string, boolean, datetime, categorical |
| measure | enum | nominal, ordinal, scale, binary, date_time, text |
| role | enum | input, target, weight, id, split, frequency, none |
| width | int | 문자열/표시 폭 |
| decimals | int | 숫자 표시 소수점 자리 |
| value_labels | dict | 값과 라벨 매핑 |
| missing_values | list | 사용자 정의 결측값 규칙 |
| unit | str | 단위 |
| allowed_min | optional float | 허용 최소값 |
| allowed_max | optional float | 허용 최대값 |
| format_pattern | str | 표시 형식 |
| datetime_format | str | 날짜 파싱 형식 |
| description | str | 긴 설명 |
| source_column | str | 원본 파일의 열 이름 |
| derived | bool | 파생 변수 여부 |
| formula | optional str | 파생 변수 식 |
| created_at | datetime | 생성 시각 |
| updated_at | datetime | 수정 시각 |

### 5.3 측정 척도

측정 척도는 분석 가능성을 판단하는 핵심이다.

| 척도 | 설명 | 예 |
|---|---|---|
| nominal | 순서 없는 범주형 | 성별, 혈액형, 진단군 |
| ordinal | 순서 있는 범주형 | Likert 1~5, 병기 |
| scale | 연속형 또는 등간/비율형 | 나이, 혈압, 키, 점수 |
| binary | 이진 변수 | 사망 여부, 양성/음성 |
| date_time | 날짜/시간 | 검사일, 등록일 |
| text | 자유 텍스트 | 비고, 소견 |

### 5.4 저장 타입과 측정 척도의 분리

저장 타입과 측정 척도는 분리해야 한다.

예를 들어 `1`, `2`, `3`으로 저장된 변수는 숫자 타입이지만, 실제 의미는 범주형일 수 있다.

```text
storage_type = integer
measure = nominal
value_labels = {1: "Control", 2: "Treatment A", 3: "Treatment B"}
```

이 경우 평균 계산은 기본적으로 권장하지 않고, 빈도분석/교차분석/카이제곱 검정을 우선 제안한다.

### 5.5 Dataset 불변성 원칙

분석 함수는 원본 Dataset을 직접 변경하지 않는다.

- 데이터 변환 기능만 Dataset을 변경한다.
- 분석 함수는 필요한 데이터를 복사하거나 view로 가져와 계산한다.
- 분석 결과는 `AnalysisResult`로 반환한다.
- 모든 변환은 syntax log에 기록한다.

---

## 6. 변수 관리 기능

### 6.1 Variable View 컬럼

Variable View는 SPSS처럼 행 하나가 변수 하나를 의미한다.

| 컬럼 | 설명 |
|---|---|
| Name | 변수명 |
| Label | 변수 라벨 |
| Type | 저장 타입 |
| Measure | 측정 척도 |
| Role | 분석 역할 |
| Values | 값 라벨 편집 |
| Missing | 결측값 규칙 |
| Width | 표시 폭 |
| Decimals | 소수점 자리 |
| Align | 표시 정렬 |
| Unit | 단위 |
| Range | 허용 범위 |
| Formula | 파생 변수 식 |
| Notes | 비고 |

### 6.2 변수명 규칙

기본 변수명 규칙은 다음과 같다.

1. 비어 있을 수 없다.
2. 같은 Dataset 안에서 고유해야 한다.
3. 영문자, 숫자, `_`를 권장한다.
4. 첫 글자는 영문자 또는 `_`를 권장한다.
5. 공백은 `_`로 자동 변환할 수 있다.
6. 원본 열 이름은 `source_column`에 보존한다.
7. 사용자가 한글 변수명을 원하면 허용하되, 내부 syntax에서는 quoting 처리한다.

### 6.3 값 라벨

값 라벨은 범주형 변수의 핵심이다.

예:

```json
{
  "0": "No",
  "1": "Yes"
}
```

UI에서는 다음 방식으로 편집한다.

- 값 입력
- 라벨 입력
- 추가/삭제
- 중복 값 검증
- 값 타입과 storage_type 일치 여부 검증

### 6.4 결측값 규칙

결측값은 두 종류를 지원한다.

#### 시스템 결측

- pandas `NaN`
- 빈 셀
- 빈 문자열
- 파싱 실패 값

#### 사용자 정의 결측

- 특정 값: `999`, `-99`, `"NA"`
- 범위: `-999 <= x <= -900`
- 문자열 패턴: `"Unknown"`, `"N/A"`

분석 대화상자에서는 다음 옵션을 제공한다.

- listwise deletion
- pairwise deletion
- analysis-specific default
- 결측 범주로 포함
- 사용자 정의 결측만 제외
- 시스템 결측만 제외

MVP 기본값은 listwise deletion이다. 상관분석은 pairwise 옵션을 제공한다.

### 6.5 변수 자동 추론

데이터 임포트 시 다음 순서로 변수 타입과 척도를 추론한다.

1. 모든 값이 날짜 형식이면 `datetime`, `date_time`
2. 모든 값이 숫자이고 고유값 수가 작으면 `integer`, `nominal` 또는 `ordinal` 후보
3. 숫자 고유값 비율이 높으면 `float`, `scale`
4. 문자열 고유값 수가 작으면 `categorical`, `nominal`
5. 문자열 고유값 수가 많으면 `string`, `text`
6. 0/1, yes/no, true/false이면 `binary`
7. 추론 신뢰도가 낮으면 사용자 확인 필요 표시

척도 자동 추론은 항상 수정 가능해야 한다.

---

## 7. 데이터 입력 및 임포트

### 7.1 임포트 대상

MVP에서 지원할 파일 형식:

| 형식 | 확장자 | 처리 |
|---|---|---|
| CSV | `.csv` | pandas read_csv |
| TSV | `.tsv` | read_csv sep=`\t` |
| TXT | `.txt` | delimiter 감지 |
| Excel | `.xlsx` | pandas read_excel/openpyxl |
| Clipboard | N/A | pandas read_clipboard 또는 Qt clipboard parser |

Phase 2:

- `.sav` SPSS 파일
- `.dta` Stata 파일
- `.sas7bdat` SAS 파일
- `.ods` OpenDocument Spreadsheet
- JSON/JSONL
- Parquet

### 7.2 Import Wizard 단계

Import Wizard는 다음 단계로 구성한다.

#### Step 1. 파일 선택

- 파일 경로
- 파일 형식 자동 감지
- 파일 크기 표시
- 마지막 수정일 표시

#### Step 2. 인코딩 선택

- UTF-8
- UTF-8-SIG
- CP949
- EUC-KR
- Latin-1
- 자동 감지 결과 표시
- 미리보기 깨짐 여부 확인

#### Step 3. 구분자 선택

- comma
- tab
- semicolon
- pipe
- space
- custom delimiter
- 자동 감지 결과 표시

#### Step 4. 헤더 및 행 설정

- 첫 행을 변수명으로 사용
- 건너뛸 행 수
- 불러올 최대 행 수
- 빈 행 처리 방식

#### Step 5. 데이터 타입 미리보기

- 각 열의 추론 타입
- 측정 척도 후보
- 결측값 후보
- 날짜 형식 후보
- 사용자가 수정 가능

#### Step 6. 최종 확인

- 행 수
- 열 수
- 경고 목록
- 가져오기 실행

### 7.3 임포트 경고

다음 상황은 경고로 표시한다.

- 중복 변수명
- 빈 변수명
- 타입 추론 실패
- 날짜 파싱 실패
- 일부 행의 열 개수 불일치
- 숫자 열에 문자열 혼입
- 매우 높은 결측률
- 범주 수가 너무 많은 nominal 변수
- 파일 인코딩 불확실

---

## 8. 스프레드시트 UI

### 8.1 기본 레이아웃

메인 윈도우는 다음 구조를 가진다.

```text
┌─────────────────────────────────────────────────────────┐
│ Menu Bar                                                │
├─────────────────────────────────────────────────────────┤
│ Tool Bar                                                │
├───────────────────────┬─────────────────────────────────┤
│ Project / Output Tree │ Main Workspace                  │
│                       │ ┌─────────────────────────────┐ │
│                       │ │ Data View / Variable View   │ │
│                       │ │ Output View                 │ │
│                       │ └─────────────────────────────┘ │
├───────────────────────┴─────────────────────────────────┤
│ Status Bar: rows, columns, selected variable, messages   │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Data View

Data View는 실제 데이터를 보여준다.

필수 기능:

- 셀 편집
- 행 추가
- 행 삭제
- 열 추가
- 열 삭제
- 복사/붙여넣기
- 다중 셀 선택
- 열 정렬
- 간단 필터
- 결측값 표시
- 값 라벨 표시 모드 토글
- 원값 표시 모드 토글
- 변수 라벨 툴팁
- 잘못된 값 강조

### 8.3 Variable View

Variable View는 변수 속성을 관리한다.

필수 기능:

- 변수 행 추가/삭제
- 변수명 변경 시 DataFrame 컬럼 동기화
- Type 콤보박스
- Measure 콤보박스
- Role 콤보박스
- Values 편집 버튼
- Missing 편집 버튼
- Range 편집 버튼
- 변경 즉시 검증
- 변경 내용 undo/redo

### 8.4 Output View

Output View는 분석 결과를 표시한다.

구조:

```text
Output Document
├── Analysis 1: Descriptives
│   ├── Notes
│   ├── Case Processing Summary
│   ├── Descriptive Statistics
│   └── Assumption Checks
├── Analysis 2: Independent t-test
│   ├── Syntax
│   ├── Group Statistics
│   ├── Test Results
│   └── Effect Size
└── Analysis 3: Linear Regression
    ├── Model Summary
    ├── ANOVA
    ├── Coefficients
    └── Diagnostics
```

Output View 요구사항:

- 결과별 접기/펼치기
- 표 복사
- 표 CSV 저장
- 전체 HTML 내보내기
- Markdown 내보내기
- 결과 노트 표시
- 분석 실행 시간 표시
- syntax 표시

---

## 9. 메뉴 구조

### 9.1 File

- New Project
- Open Project
- Save Project
- Save Project As
- Import Data
  - CSV
  - TXT/Delimited
  - Excel
  - Clipboard
- Export Data
  - CSV
  - Excel
  - JSON metadata
- Export Output
  - HTML
  - Markdown
  - CSV tables
- Recent Files
- Exit

### 9.2 Edit

- Undo
- Redo
- Cut
- Copy
- Paste
- Clear
- Find
- Replace
- Select All
- Preferences

### 9.3 Data

- Define Variable Properties
- Sort Cases
- Filter Cases
- Select Cases
- Split File
- Weight Cases
- Recode into Same Variable
- Recode into Different Variable
- Compute Variable
- Rank Cases
- Aggregate
- Merge Files
- Reshape
- Validate Data

MVP에서는 `Sort`, `Filter`, `Compute Variable` 정도만 우선 구현한다.

### 9.4 Analyze

- Descriptive Statistics
  - Frequencies
  - Descriptives
  - Explore
  - Crosstabs
- Compare Means
  - One-Sample t Test
  - Independent-Samples t Test
  - Paired-Samples t Test
  - One-Way ANOVA
- Nonparametric Tests
  - Mann-Whitney U
  - Wilcoxon Signed-Rank
  - Kruskal-Wallis
  - Friedman
  - Chi-square Goodness-of-Fit
- Correlate
  - Bivariate
  - Partial
- Regression
  - Linear
  - Logistic
- Scale
  - Reliability Analysis
- Diagnostic Tests
  - ROC Analysis
  - Sensitivity/Specificity
- Survival
  - Kaplan-Meier
  - Cox Regression
- Agreement
  - Cohen's Kappa
  - ICC
  - Bland-Altman numeric summary

MVP 메뉴는 완성해도 되지만, 구현되지 않은 항목은 비활성화하거나 “planned”로 표시한다.

### 9.5 Transform

- Compute Variable
- Recode
- Standardize
- Categorize
- Date/Time Functions
- String Functions

### 9.6 Graphs

MVP에서는 기본 비활성화한다.

- Chart Builder
- Histogram
- Boxplot
- Scatterplot
- Bar Chart
- Line Chart
- ROC Curve
- Python Visualization Script

향후 Python 연동 방식으로 구현한다.

### 9.7 Tools

- Variable Audit
- Missing Data Summary
- Syntax Log
- Python Bridge Settings
- Plugin Manager
- Options

### 9.8 Help

- User Manual
- Statistical Method Notes
- About
- Check for Updates

---

## 10. 분석 엔진 설계

### 10.1 분석 엔진의 기본 원칙

분석 엔진은 UI와 분리한다.

UI는 사용자 입력을 `AnalysisSpec`으로 변환하고, 분석 엔진은 `AnalysisResult`를 반환한다.

```text
UI Dialog → AnalysisSpec → AnalysisPlugin.run() → AnalysisResult → Output View
```

### 10.2 AnalysisPlugin 인터페이스

모든 분석은 동일한 플러그인 규격을 따른다.

```python
from typing import Protocol
from statworkbench.core.dataset import Dataset
from statworkbench.analysis.result import AnalysisResult

class AnalysisPlugin(Protocol):
    id: str
    name: str
    category: str
    description: str
    variable_requirements: list

    def validate(self, dataset: Dataset, spec: dict) -> list[str]:
        ...

    def run(self, dataset: Dataset, spec: dict) -> AnalysisResult:
        ...
```

### 10.3 AnalysisSpec

`AnalysisSpec`는 분석 실행 명세다.

필드:

| 필드 | 설명 |
|---|---|
| analysis_id | 실행할 분석 id |
| variables | 선택 변수 |
| groups | 그룹 변수 |
| weights | 가중치 변수 |
| filters | 적용 필터 |
| options | 분석 옵션 |
| missing_policy | 결측 처리 |
| confidence_level | 신뢰수준 |
| created_at | 생성 시각 |
| user_note | 사용자 노트 |

예:

```json
{
  "analysis_id": "independent_t_test",
  "variables": {
    "dependent": ["systolic_bp"],
    "group": "treatment_group"
  },
  "options": {
    "equal_var": "auto",
    "effect_size": true,
    "normality_test": true,
    "levene_test": true
  },
  "missing_policy": "listwise",
  "confidence_level": 0.95
}
```

### 10.4 AnalysisResult

`AnalysisResult`는 다음 구조를 가진다.

```text
AnalysisResult
├── id
├── title
├── created_at
├── spec
├── notes
├── warnings
├── tables
├── text_blocks
├── assumptions
├── diagnostics
├── figures
├── syntax
└── metadata
```

표는 `ResultTable`로 저장한다.

```text
ResultTable
├── title
├── dataframe
├── footnotes
├── format_rules
└── export_options
```

### 10.5 분석 Registry

`AnalysisRegistry`는 분석 플러그인을 등록하고 조회한다.

기능:

- 전체 분석 목록 반환
- 메뉴 카테고리별 분석 목록 반환
- 변수 타입에 따른 추천 분석 반환
- 분석 id로 플러그인 실행
- 구현/미구현 상태 관리
- 테스트용 mock plugin 등록

---

## 11. 변수 타입 기반 분석 추천

### 11.1 분석 적합성 매트릭스

| 분석 | 종속 변수 | 독립/그룹 변수 | 조건 |
|---|---|---|---|
| 빈도분석 | nominal/ordinal/binary | 없음 | 모든 범주형 |
| 기술통계 | scale | 없음 | 숫자형 연속 변수 |
| 정규성 검정 | scale | 선택 group | n >= 3 권장 |
| 교차분석 | nominal/ordinal/binary | nominal/ordinal/binary | 2개 이상 범주형 |
| 독립 t 검정 | scale | binary | 그룹 2개 |
| 대응 t 검정 | scale pair | 없음 | 두 연속 변수 |
| 일원 ANOVA | scale | nominal/ordinal | 그룹 3개 이상 |
| Mann-Whitney U | ordinal/scale | binary | 독립 2그룹 |
| Wilcoxon | ordinal/scale pair | 없음 | 대응 2변수 |
| Kruskal-Wallis | ordinal/scale | nominal/ordinal | 독립 3그룹 이상 |
| Friedman | ordinal/scale repeated | 없음 | 반복측정 3개 이상 |
| Pearson correlation | scale + scale | 없음 | 선형 관계 |
| Spearman correlation | ordinal/scale | 없음 | 단조 관계 |
| Linear regression | scale target | scale/categorical predictors | 연속 종속 |
| Logistic regression | binary target | scale/categorical predictors | 이진 종속 |

### 11.2 UI 선택 제약

분석 대화상자에서는 선택 가능한 변수 목록을 역할별로 필터링한다.

예:

- 독립표본 t 검정의 Dependent Variable 영역에는 `scale` 변수만 표시
- Grouping Variable 영역에는 `binary` 또는 범주 수가 2인 nominal 변수 우선 표시
- 상관분석에는 `scale` 또는 `ordinal` 변수 표시
- 선형회귀 종속변수는 `scale`만 표시
- 로지스틱 회귀 종속변수는 `binary`만 표시

사용자가 부적합한 변수를 강제로 선택하려 하면 경고를 표시한다.

---

## 12. 통계 분석 상세 명세

### 12.1 공통 출력 규칙

모든 분석은 다음을 출력한다.

1. 분석 제목
2. 사용 데이터셋 이름
3. 분석 실행 시각
4. 선택 변수
5. 결측 처리 방법
6. 유효 사례 수
7. 제외 사례 수
8. 주요 결과 표
9. 해석 경고
10. syntax 기록

### 12.2 p-value 표기

기본 표기 규칙:

| 값 | 표시 |
|---|---|
| p < 0.001 | `< .001` |
| 0.001 <= p < 1 | 소수점 3자리 |
| p >= 1 | `1.000` |
| NaN | 빈칸 또는 `NA` |

별표 표기는 옵션으로 제공한다.

- `*` p < .05
- `**` p < .01
- `***` p < .001

단, 별표만으로 해석하지 않도록 footnote를 함께 출력한다.

### 12.3 신뢰구간

기본 신뢰수준은 95%다.

모든 가능한 분석에는 신뢰구간을 제공한다.

- 평균 신뢰구간
- 평균 차이 신뢰구간
- 상관계수 신뢰구간
- 회귀계수 신뢰구간
- odds ratio 신뢰구간
- effect size 신뢰구간은 가능한 경우 제공

### 12.4 기술통계

#### 입력

- scale 변수 1개 이상
- 선택 그룹 변수 0개 또는 1개

#### 통계량

- N
- Missing
- Mean
- Median
- Standard deviation
- Standard error
- Variance
- Minimum
- Maximum
- Range
- Q1
- Q3
- IQR
- Skewness
- Kurtosis
- 95% CI for mean

#### 구현

- pandas/numpy 기반
- scipy.stats로 skew/kurtosis 계산
- 결측 처리 옵션 반영

#### 출력 표

`Descriptive Statistics`

| Variable | N | Missing | Mean | SD | Median | IQR | Min | Max | Skewness | Kurtosis |

### 12.5 빈도분석

#### 입력

- nominal/ordinal/binary/text 변수
- text는 고유값이 너무 많으면 경고

#### 출력

- Frequency
- Percent
- Valid Percent
- Cumulative Percent
- Missing Count

#### 옵션

- 결측값을 표에 포함
- 누적 퍼센트 표시
- 값 라벨 표시
- 빈도 기준 정렬
- 값 기준 정렬

### 12.6 정규성 검정

#### 입력

- scale 변수

#### 검정

- Shapiro-Wilk
- Kolmogorov-Smirnov 또는 Lilliefors는 Phase 2
- Anderson-Darling은 Phase 2

#### 출력

- N
- Statistic
- p-value
- 해석 경고

#### 경고

- 표본 수가 매우 크면 작은 차이도 유의할 수 있음을 표시
- 표본 수가 너무 작으면 검정력이 낮음을 표시

### 12.7 교차분석

#### 입력

- row variable: nominal/ordinal/binary
- column variable: nominal/ordinal/binary
- optional layer variable

#### 출력

- count
- row percent
- column percent
- total percent
- expected count
- residual
- standardized residual

#### 검정

- Pearson chi-square
- likelihood ratio chi-square
- Fisher exact test for 2x2
- continuity correction 옵션
- Cramer's V
- Phi for 2x2

#### 경고

- 기대빈도 5 미만 셀이 많으면 Fisher/exact test 권고
- 범주 수가 너무 많으면 표 해석 어려움 경고

### 12.8 독립표본 t 검정

#### 입력

- dependent: scale
- group: binary 또는 고유 그룹 2개

#### 출력

- 그룹별 N, Mean, SD, SE
- Levene's test
- t statistic
- df
- p-value
- mean difference
- standard error difference
- 95% CI
- Cohen's d
- Hedges' g

#### 옵션

- 등분산 가정
- Welch correction
- 양측/단측 검정
- 정규성 검정 포함

#### 기본 정책

`equal_var = auto`이면 Levene 검정 결과를 참고하되, 기본 출력에는 등분산/비등분산 결과를 모두 표시한다.

### 12.9 대응표본 t 검정

#### 입력

- paired variable 2개 이상 pair
- 같은 행이 같은 subject라고 가정

#### 출력

- 각 변수 평균/SD
- 차이 평균/SD
- t
- df
- p
- 95% CI
- Cohen's dz
- 상관계수

### 12.10 일원분산분석

#### 입력

- dependent: scale
- factor: nominal/ordinal, 그룹 3개 이상

#### 출력

- 그룹별 기술통계
- Levene test
- ANOVA table
- Welch ANOVA 옵션
- eta squared
- omega squared
- post-hoc test 결과

#### Post-hoc

MVP:

- Tukey HSD 가능하면 구현
- Bonferroni pairwise t-test 대안

Phase 2:

- Games-Howell
- Dunnett
- Scheffe

### 12.11 비모수 검정

#### Mann-Whitney U

- 독립 2그룹
- ordinal/scale 종속
- U statistic
- p-value
- rank-biserial correlation

#### Wilcoxon signed-rank

- 대응 2변수
- W statistic
- p-value
- effect size r

#### Kruskal-Wallis

- 독립 3그룹 이상
- H statistic
- p-value
- epsilon squared
- Dunn post-hoc는 Phase 2

#### Friedman

- 반복측정 3조건 이상
- chi-square statistic
- p-value
- Kendall's W

### 12.12 상관분석

#### 입력

- 변수 2개 이상
- scale 또는 ordinal

#### 방법

- Pearson
- Spearman
- Kendall

#### 출력

- correlation matrix
- p-value matrix
- N matrix
- confidence interval for Pearson
- pairwise/listwise deletion 옵션

#### 옵션

- two-tailed
- one-tailed
- flag significant correlations
- diagonal 표시 여부

### 12.13 선형회귀

#### 입력

- dependent: scale
- predictors: scale, binary, nominal, ordinal

#### 범주형 처리

- nominal predictor는 자동 dummy coding
- 기준 범주 선택 가능
- value_labels를 계수표에 반영

#### 출력

- Model Summary
  - R
  - R²
  - Adjusted R²
  - RMSE
- ANOVA table
- Coefficients
  - B
  - SE
  - beta
  - t
  - p
  - CI
- Diagnostics
  - residual summary
  - multicollinearity VIF
  - Durbin-Watson
  - influential cases는 Phase 2

#### 구현

statsmodels OLS를 사용한다.

### 12.14 로지스틱 회귀

MVP 후반 또는 Phase 2에서 구현한다.

#### 입력

- dependent: binary
- predictors: scale/categorical

#### 출력

- coefficients
- odds ratio
- CI for odds ratio
- Wald test
- likelihood ratio test
- pseudo R²
- classification table
- ROC numeric summary

---

## 13. MedCalc 성격의 임상 통계 확장

MedCalc 계열 사용자를 고려하여 Phase 2 이후 다음 기능을 추가한다.

### 13.1 진단 정확도

- sensitivity
- specificity
- positive predictive value
- negative predictive value
- accuracy
- likelihood ratio positive
- likelihood ratio negative
- diagnostic odds ratio
- confidence intervals

### 13.2 ROC 분석

MVP에서는 숫자 결과만 우선 구현 가능하다.

- AUC
- standard error
- 95% CI
- optimal cutoff
- Youden index
- sensitivity/specificity by cutoff
- DeLong test는 Phase 3

### 13.3 생존분석

- Kaplan-Meier
- log-rank test
- Cox proportional hazards
- hazard ratio
- median survival
- censoring 정보 표시

### 13.4 일치도 분석

- Cohen's kappa
- weighted kappa
- ICC
- Bland-Altman numeric summary
- Passing-Bablok regression
- Deming regression

---

## 14. Syntax Log

### 14.1 목적

GUI로 실행한 모든 작업은 재현 가능한 syntax로 기록한다.

예:

```text
FREQUENCIES VARIABLES=sex diagnosis
  /ORDER=VALUE
  /MISSING=EXCLUDE.

TTEST GROUPS=treatment(0 1)
  /VARIABLES=systolic_bp
  /MISSING=ANALYSIS
  /CRITERIA=CI(.95).

REGRESSION
  /DEPENDENT outcome_score
  /METHOD=ENTER age sex baseline_score
  /STATISTICS COEFF R ANOVA CI(95).
```

### 14.2 MVP 정책

MVP에서는 완전한 parser를 먼저 만들지 않아도 된다.

우선순위:

1. GUI 실행 내용을 syntax 문자열로 기록
2. syntax log를 저장
3. output에 syntax 표시
4. 나중에 parser로 재실행 가능하게 확장

### 14.3 SyntaxCommand 모델

```python
class SyntaxCommand(BaseModel):
    command: str
    parameters: dict
    raw_text: str
    created_at: datetime
    dataset_id: str
    result_id: str | None = None
```

---

## 15. 프로젝트 파일 포맷

### 15.1 확장자

기본 프로젝트 확장자는 `.swb`로 한다.

`StatWorkbench Project Bundle`의 약자다.

### 15.2 내부 구조

`.swb`는 ZIP 파일이다.

```text
project.swb
├── manifest.json
├── data/
│   ├── active.parquet
│   └── backup.csv
├── metadata/
│   ├── variables.json
│   ├── dataset.json
│   └── project.json
├── output/
│   ├── output.json
│   ├── output.html
│   └── tables/
├── syntax/
│   └── syntax.log
└── audit/
    └── audit.jsonl
```

### 15.3 저장 원칙

- 데이터는 Parquet 우선
- Parquet 저장 실패 시 CSV fallback
- 메타데이터는 JSON
- 사람이 읽을 수 있는 백업을 포함
- 프로젝트 버전 기록
- future migration을 위한 schema_version 기록

---

## 16. 결과 포맷팅 기준

### 16.1 숫자 포맷

| 항목 | 기본 |
|---|---|
| 평균 | 소수점 3자리 |
| 표준편차 | 소수점 3자리 |
| 통계량 | 소수점 3자리 |
| p-value | p-value 규칙 적용 |
| 퍼센트 | 소수점 1자리 |
| 신뢰구간 | `[lower, upper]` |

### 16.2 표 footnote

모든 표는 필요한 경우 footnote를 가진다.

예:

```text
Note. Missing values were excluded listwise.
Note. Equal variances not assumed results use Welch's correction.
Note. p-values are two-tailed unless otherwise specified.
```

### 16.3 경고 메시지

경고는 분석 결과 상단에 표시한다.

예:

- “Group variable has more than two groups. Independent t-test requires exactly two groups.”
- “More than 20% of expected cell counts are below 5. Interpret chi-square results with caution.”
- “The selected variable is marked nominal but numeric summaries were requested.”

---

## 17. UI 대화상자 설계

### 17.1 공통 Analysis Dialog

모든 분석 대화상자는 같은 기본 구조를 사용한다.

```text
┌──────────────────────────────────────────────┐
│ Available Variables       Selected Variables │
│ ┌─────────────────┐      ┌────────────────┐ │
│ │ age             │  ->  │ dependent      │ │
│ │ sex             │      │ group          │ │
│ │ treatment       │      │ covariates     │ │
│ └─────────────────┘      └────────────────┘ │
│                                              │
│ Options                                      │
│ [x] Include confidence intervals             │
│ [x] Exclude missing values listwise          │
│ Confidence level: 95%                        │
│                                              │
│ [OK] [Paste Syntax] [Cancel] [Help]          │
└──────────────────────────────────────────────┘
```

### 17.2 필수 버튼

- OK: 분석 실행
- Paste Syntax: 실행하지 않고 syntax log/editor에 추가
- Cancel: 닫기
- Help: 분석 설명 열기

### 17.3 변수 목록 표시

변수 목록에는 다음 정보를 표시한다.

- 변수명
- 변수 라벨
- 측정 척도 아이콘
- 타입 아이콘
- 결측률
- 값 라벨 여부

### 17.4 분석별 유효성 검증

OK 버튼을 누르면 다음 검증을 수행한다.

1. 필수 변수 선택 여부
2. 변수 척도 적합성
3. 그룹 수 조건
4. 유효 사례 수
5. 결측 처리 후 데이터 존재 여부
6. 분석 옵션 충돌 여부

오류는 실행을 막고, 경고는 사용자가 확인 후 실행 가능하게 한다.

---

## 18. 데이터 변환

### 18.1 Compute Variable

수식 기반 파생 변수를 만든다.

MVP에서는 안전한 expression evaluator를 사용한다.

허용 예:

```text
bmi = weight_kg / (height_m ** 2)
log_crp = log(crp)
age_group = cut(age, bins=[0, 40, 65, 120])
```

금지:

- 임의 파일 접근
- 임의 시스템 명령
- 네트워크 접근
- Python builtins 직접 노출

### 18.2 Recode

두 가지를 지원한다.

#### Recode into Same Variable

기존 변수 값 변경. 위험하므로 확인 필요.

#### Recode into Different Variable

새 변수 생성. 기본 권장.

예:

```text
0 -> "No"
1 -> "Yes"
else -> missing
```

### 18.3 Filter Cases

필터 조건을 Dataset에 적용하되, 원본 데이터를 삭제하지 않는다.

- 활성 필터 상태 저장
- Output에 필터 조건 표시
- Data View에서 필터된 행 표시 방식 제공

---

## 19. 결측값 처리 정책

### 19.1 결측값의 일관성

모든 분석 함수는 동일한 결측 처리 유틸리티를 사용한다.

`analysis/assumptions.py` 또는 `core/validation.py`에 공통 함수를 둔다.

```python
def prepare_analysis_frame(
    dataset: Dataset,
    variables: list[str],
    missing_policy: MissingPolicy,
    include_user_missing: bool = False,
) -> PreparedAnalysisFrame:
    ...
```

### 19.2 결측 처리 결과 보고

모든 분석 결과에는 `Case Processing Summary`를 포함한다.

| Total Cases | Valid Cases | Excluded Cases | Excluded % |
|---:|---:|---:|---:|

### 19.3 사용자 정의 결측

사용자 정의 결측값은 DataFrame 내부 값을 반드시 NaN으로 바꾸지 않아도 된다.

분석 준비 단계에서 제외 처리할 수 있다.

장점:

- 원본 값 보존
- 결측 규칙 변경 가능
- syntax 재현 가능

---

## 20. 분석 정확도 검증 전략

### 20.1 Golden Test

각 분석은 검증용 데이터셋과 기대 결과를 가진다.

예:

```text
tests/fixtures/
├── ttest_independent_case1.csv
├── anova_oneway_case1.csv
├── correlation_case1.csv
└── regression_case1.csv
```

기대 결과는 JSON으로 저장한다.

```text
tests/expected/
├── ttest_independent_case1.json
├── anova_oneway_case1.json
└── regression_case1.json
```

### 20.2 허용 오차

통계 계산 비교에는 tolerance를 둔다.

- 일반 통계량: `1e-6`
- p-value: `1e-6`
- 복잡한 모델 계수: `1e-5`
- 표시 문자열: 별도 snapshot test

### 20.3 외부 기준 비교

가능하면 다음과 비교한다.

- scipy 공식 계산
- statsmodels 결과
- R 결과
- 검증된 수작업 예제
- 문헌 예제

단, 프로젝트 내부 테스트는 외부 프로그램 실행 없이 통과해야 한다.

### 20.4 테스트 범위

| 영역 | 테스트 |
|---|---|
| 변수 메타데이터 | 타입/척도/결측 규칙 검증 |
| 임포트 | CSV/TXT/Excel parsing |
| 분석 | 통계량 정확도 |
| 결과 포맷 | p-value/CI 문자열 |
| Syntax | GUI 옵션 → syntax 변환 |
| 프로젝트 저장 | 저장 후 재로드 동일성 |
| UI | 기본 모델 동작과 dialog validation |

---

## 21. 품질 기준

Hermes Agent는 다음 기준을 만족하지 못하면 작업을 완료로 간주하지 않는다.

### 21.1 코드 품질

- type hints 사용
- domain logic과 UI logic 분리
- 함수는 단일 책임 원칙 준수
- 통계 함수는 side effect 없음
- 오류 메시지는 사용자 친화적
- 내부 예외와 사용자 메시지를 분리

### 21.2 테스트 품질

- 새 분석 기능은 최소 3개 테스트
- 정상 케이스
- 결측 포함 케이스
- 부적합 변수 케이스
- 작은 표본 케이스
- 결과 포맷 테스트

### 21.3 UI 품질

- 메뉴 항목은 실제 기능과 연결
- 미구현 기능은 명확히 비활성화
- 사용자가 잘못된 분석을 실행하지 않도록 사전 검증
- 긴 계산 중 UI freeze 방지
- 상태바에 작업 진행 표시

### 21.4 통계 품질

- 분석 가정 표시
- 결측 처리 명시
- 표본 수 명시
- effect size 가능한 경우 표시
- 신뢰구간 가능한 경우 표시
- 경고를 숨기지 않음

---

## 22. 개발 로드맵

### Phase 0 — 프로젝트 기반

목표: 실행 가능한 빈 앱과 테스트 환경 구축

작업:

- pyproject.toml 작성
- 패키지 구조 생성
- 기본 main window 생성
- pytest 설정
- sample data 생성
- README 작성
- lint/type check 도입

완료 기준:

- `python -m statworkbench` 실행 가능
- 빈 MainWindow 표시
- `pytest` 통과

### Phase 1 — 데이터/변수 모델

목표: Dataset과 VariableMeta 구현

작업:

- VariableMeta pydantic 모델
- Dataset 클래스
- 변수명 검증
- 타입/척도 enum
- value labels
- missing rules
- audit log
- 기본 테스트

완료 기준:

- DataFrame과 변수 메타데이터가 동기화됨
- 변수명 변경 시 컬럼명 변경
- 결측 규칙 테스트 통과

### Phase 2 — 임포트

목표: CSV/TXT/Excel 불러오기

작업:

- CSV reader
- TXT delimiter detection
- Excel reader
- encoding detection
- import preview model
- variable inference
- import wizard UI

완료 기준:

- sample CSV/TXT/XLSX 로드 가능
- 변수 메타데이터 자동 생성
- 잘못된 파일에 사용자 오류 메시지 표시

### Phase 3 — Data View / Variable View

목표: 스프레드시트식 편집

작업:

- pandas-backed QTableModel
- Data View 편집
- Variable View 편집
- undo/redo 기본 구조
- 값 라벨 편집 dialog
- 결측값 편집 dialog

완료 기준:

- UI에서 데이터와 변수 속성 편집 가능
- Data View와 Variable View 동기화
- 저장 전 dirty state 표시

### Phase 4 — 기본 분석 엔진

목표: UI 없이도 분석 함수가 정확히 동작

작업:

- AnalysisPlugin base
- AnalysisRegistry
- AnalysisResult
- formatting
- descriptive
- frequencies
- normality
- crosstab
- t-tests
- correlation

완료 기준:

- 모든 분석 테스트 통과
- 결과가 구조화된 ResultTable로 반환

### Phase 5 — 메뉴 기반 분석 UI

목표: 사용자가 메뉴로 분석 실행

작업:

- Analyze 메뉴 구성
- 공통 analysis dialog
- 분석별 dialog
- result rendering
- output tree
- syntax log

완료 기준:

- GUI에서 분석 선택 → 변수 선택 → 결과 출력 가능
- 잘못된 변수 선택 시 실행 차단 또는 경고

### Phase 6 — 프로젝트 저장/내보내기

목표: 작업 재개 가능

작업:

- `.swb` 저장
- `.swb` 열기
- output export
- data export
- metadata export
- syntax export

완료 기준:

- 저장 후 다시 열어도 데이터/변수/결과 유지
- HTML/Markdown output export 가능

### Phase 7 — 회귀/고급분석

목표: 실사용 분석 범위 확대

작업:

- ANOVA 고도화
- linear regression
- logistic regression
- reliability
- ROC numeric
- diagnostic accuracy

완료 기준:

- 회귀 분석 결과 표가 statsmodels 기준과 일치
- 범주형 predictor 처리 가능

### Phase 8 — Python 시각화 브리지

목표: Python 기반 시각화 확장

작업:

- VisualizationSpec
- Python script runner
- plot output capture
- figure registry
- chart menu prototype

완료 기준:

- 히스토그램/박스플롯/산점도 prototype 생성
- 결과 output에 figure 등록 가능

---

## 23. Hermes Agent 작업 분할 전략

### 23.1 첫 실행 프롬프트 예시

Hermes Agent에게 처음 줄 수 있는 지시:

```text
이 저장소는 HERMES.md의 지시를 따른다. 
우선 Phase 0과 Phase 1을 구현하라.

요구사항:
1. pyproject.toml을 만들고 src/statworkbench 패키지 구조를 생성한다.
2. PySide6 기반으로 빈 MainWindow가 실행되게 한다.
3. core/variable.py와 core/dataset.py에 VariableMeta, Dataset 모델을 구현한다.
4. 변수명 검증, 측정척도 enum, 저장 타입 enum, 결측값 규칙을 구현한다.
5. tests/core에 단위 테스트를 작성한다.
6. pytest가 통과해야 한다.
7. README에는 실행 방법을 작성한다.
```

### 23.2 두 번째 프롬프트 예시

```text
Phase 2를 구현하라.

1. CSV/TXT/XLSX import 기능을 구현한다.
2. encoding, delimiter, header 옵션을 지원한다.
3. 변수 타입과 척도를 자동 추론한다.
4. ImportResult와 ImportWarning 모델을 만든다.
5. sample data를 examples에 추가한다.
6. tests/io에 CSV/TXT/XLSX 테스트를 작성한다.
7. UI import dialog는 최소 기능으로 연결한다.
```

### 23.3 세 번째 프롬프트 예시

```text
Phase 3과 Phase 4 일부를 구현하라.

1. Data View와 Variable View를 PySide6 QTableView로 구현한다.
2. pandas-backed table model을 만든다.
3. 변수 메타데이터 변경이 DataFrame과 동기화되게 한다.
4. descriptive, frequency 분석 엔진을 구현한다.
5. 분석 엔진은 UI와 독립적으로 테스트한다.
6. Analyze > Descriptive Statistics > Frequencies, Descriptives 메뉴를 연결한다.
```

### 23.4 작업 중 Hermes가 항상 확인할 체크리스트

작업 완료 전 다음을 확인한다.

```text
- 앱이 실행되는가?
- pytest가 통과하는가?
- 새 기능의 테스트가 있는가?
- UI에서 오류가 사용자 친화적으로 표시되는가?
- 분석 결과에 N, 결측, 옵션이 기록되는가?
- syntax log가 남는가?
- 미구현 메뉴는 비활성화되어 있는가?
- README 또는 docs가 업데이트되었는가?
```

---

## 24. pyproject.toml 기본안

Hermes Agent는 초기 구현 시 다음 형태를 기준으로 작성한다.

```toml
[project]
name = "statworkbench"
version = "0.1.0"
description = "Spreadsheet-style statistical analysis workbench with variable metadata management."
readme = "README.md"
requires-python = ">=3.11"
authors = [
  { name = "StatWorkbench Contributors" }
]
dependencies = [
  "numpy>=1.26",
  "pandas>=2.2",
  "scipy>=1.12",
  "statsmodels>=0.14",
  "pydantic>=2.6",
  "PySide6>=6.6",
  "openpyxl>=3.1",
  "pyarrow>=15.0",
  "charset-normalizer>=3.3",
  "python-dateutil>=2.8"
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-qt>=4.4",
  "pytest-cov>=4.1",
  "ruff>=0.3",
  "mypy>=1.8"
]
viz = [
  "matplotlib>=3.8",
  "plotly>=5.20"
]

[project.scripts]
statworkbench = "statworkbench.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
```

---

## 25. 핵심 클래스 설계 초안

### 25.1 variable.py

```python
from enum import Enum
from pydantic import BaseModel, Field

class StorageType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"

class MeasureLevel(str, Enum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    SCALE = "scale"
    BINARY = "binary"
    DATE_TIME = "date_time"
    TEXT = "text"

class VariableRole(str, Enum):
    INPUT = "input"
    TARGET = "target"
    WEIGHT = "weight"
    ID = "id"
    SPLIT = "split"
    FREQUENCY = "frequency"
    NONE = "none"

class MissingRule(BaseModel):
    kind: str
    value: object | None = None
    min_value: float | None = None
    max_value: float | None = None

class VariableMeta(BaseModel):
    name: str
    label: str = ""
    storage_type: StorageType
    measure: MeasureLevel
    role: VariableRole = VariableRole.INPUT
    width: int = 8
    decimals: int = 2
    value_labels: dict[str, str] = Field(default_factory=dict)
    missing_values: list[MissingRule] = Field(default_factory=list)
    unit: str = ""
    allowed_min: float | None = None
    allowed_max: float | None = None
    format_pattern: str = ""
    datetime_format: str = ""
    description: str = ""
    source_column: str = ""
    derived: bool = False
    formula: str | None = None
```

### 25.2 dataset.py

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    variables: dict[str, VariableMeta]

    def validate(self) -> list[str]:
        ...

    def rename_variable(self, old: str, new: str) -> None:
        ...

    def add_variable(self, meta: VariableMeta, values=None) -> None:
        ...

    def remove_variable(self, name: str) -> None:
        ...

    def get_analysis_frame(self, variable_names: list[str], missing_policy: str) -> pd.DataFrame:
        ...
```

### 25.3 result.py

```python
from pydantic import BaseModel
import pandas as pd

class ResultTable(BaseModel):
    title: str
    dataframe: pd.DataFrame
    footnotes: list[str] = []
    metadata: dict = {}

    class Config:
        arbitrary_types_allowed = True

class AnalysisResult(BaseModel):
    analysis_id: str
    title: str
    tables: list[ResultTable]
    warnings: list[str] = []
    notes: list[str] = []
    syntax: str = ""
    metadata: dict = {}

    class Config:
        arbitrary_types_allowed = True
```

---

## 26. 분석별 구현 우선순위

### Priority A — MVP 필수

1. Frequencies
2. Descriptives
3. Normality
4. Crosstabs
5. Independent t-test
6. Paired t-test
7. One-way ANOVA
8. Correlation
9. Mann-Whitney U
10. Wilcoxon
11. Kruskal-Wallis
12. Friedman

### Priority B — 초기 실사용 강화

1. Linear regression
2. Logistic regression
3. Reliability analysis
4. ROC numeric summary
5. Diagnostic accuracy
6. Chi-square goodness-of-fit
7. Partial correlation

### Priority C — MedCalc 성격 강화

1. Kaplan-Meier
2. Cox regression
3. Bland-Altman
4. Passing-Bablok
5. Deming regression
6. ICC
7. Weighted kappa

### Priority D — 고급/장기

1. Repeated-measures ANOVA
2. Mixed models
3. Generalized linear models
4. Meta-analysis
5. Power analysis
6. Sample size calculator
7. Bayesian modules

---

## 27. Python 시각화 연동 계획

### 27.1 기본 원칙

시각화는 MVP의 핵심이 아니다.  
그러나 초기부터 구조는 마련한다.

시각화는 별도 `viz_bridge` 모듈로 분리한다.

```text
VisualizationSpec → PythonRunner → FigureArtifact → OutputView
```

### 27.2 VisualizationSpec

필드:

| 필드 | 설명 |
|---|---|
| chart_type | histogram, boxplot, scatter, bar, roc |
| variables | 사용 변수 |
| group | 그룹 변수 |
| options | 그래프 옵션 |
| theme | 스타일 |
| output_format | png, svg, html |
| code | 생성된 Python 코드 |

### 27.3 PythonRunner

역할:

- 안전한 임시 디렉토리 생성
- 분석용 DataFrame을 임시 파일로 저장
- Python 스크립트 실행
- 결과 이미지/HTML 수집
- 오류 메시지 반환

### 27.4 초기 구현 대상

Phase 8에서 다음만 구현한다.

- Histogram
- Boxplot
- Scatterplot
- Bar chart
- ROC curve plot

### 27.5 보안

Python 시각화 브리지는 사용자가 임의 코드를 실행할 수 있으므로 다음 정책을 둔다.

- 기본 GUI chart builder는 안전한 템플릿 코드만 사용
- Advanced Python script는 명시적 경고 후 실행
- 실행 로그 저장
- 외부 네트워크 접근 기본 금지 옵션 제공
- 실행 디렉토리 제한

---

## 28. 사용자 경험 세부 원칙

### 28.1 초보자 친화성

사용자는 통계 용어에 익숙하지 않을 수 있다.

따라서 대화상자는 다음 정보를 제공한다.

- 이 분석은 언제 쓰는가?
- 필요한 변수 유형은 무엇인가?
- 현재 선택한 변수는 적합한가?
- 결과에서 무엇을 봐야 하는가?
- 어떤 가정이 필요한가?

### 28.2 전문가 친화성

전문가는 빠른 작업과 재현성을 원한다.

따라서 다음 기능을 제공한다.

- syntax log
- keyboard shortcuts
- 최근 분석 재실행
- 옵션 preset
- 결과 테이블 복사
- batch analysis는 Phase 2 이후

### 28.3 오류 메시지 원칙

나쁜 메시지:

```text
ValueError: array must not contain infs or NaNs
```

좋은 메시지:

```text
선택한 변수 `age`와 `blood_pressure`에서 결측값을 제외한 후 유효 사례가 2개뿐입니다.
상관분석에는 최소 3개 이상의 유효 사례가 필요합니다.
결측 처리 방식을 변경하거나 데이터를 확인하세요.
```

---

## 29. 국제화와 한국어 지원

초기 UI 언어는 한국어를 우선하되, 내부 코드는 영어로 작성한다.

정책:

- 코드 식별자: 영어
- UI 문자열: 번역 리소스 사용
- 문서: 한국어 우선, 영어 병기 가능
- 통계 용어: 한국어 + 영어 병기

예:

```text
독립표본 t 검정 (Independent-Samples t Test)
상관분석 (Correlation Analysis)
기술통계 (Descriptive Statistics)
```

---

## 30. 개인정보 및 보안

### 30.1 로컬 우선

기본적으로 모든 데이터는 로컬에서 처리한다.

- 사용자의 데이터 자동 업로드 금지
- 외부 API 전송 금지
- 오류 리포트 자동 전송 금지
- 샘플 데이터와 사용자 데이터 구분

### 30.2 민감 데이터

의료/임상 데이터 가능성을 고려한다.

- 최근 파일 목록에 전체 경로 노출 여부 옵션
- 프로젝트 파일 암호화는 Phase 3
- export 시 식별자 포함 경고
- ID role 변수 지정 가능

### 30.3 Audit Log

다음 이벤트를 기록한다.

- 데이터 임포트
- 변수 속성 변경
- 데이터 변환
- 분석 실행
- 결과 export
- 프로젝트 저장

Audit log는 재현성과 추적성을 위한 것이며, 민감한 셀 값을 과도하게 기록하지 않는다.

---

## 31. 플러그인 아키텍처

### 31.1 분석 플러그인

분석은 플러그인 구조로 추가할 수 있게 한다.

```text
analysis/
├── base.py
├── registry.py
└── plugins/
    ├── descriptive.py
    ├── ttest.py
    └── regression.py
```

초기에는 내부 모듈로 구현하되, 나중에 외부 플러그인을 로드할 수 있게 설계한다.

### 31.2 플러그인 메타데이터

각 분석 플러그인은 다음 정보를 제공한다.

```json
{
  "id": "independent_t_test",
  "name": "Independent-Samples t Test",
  "category": "Compare Means",
  "implemented": true,
  "required_variables": [
    {"slot": "dependent", "measure": ["scale"], "min": 1, "max": null},
    {"slot": "group", "measure": ["binary", "nominal"], "min": 1, "max": 1}
  ],
  "options_schema": {}
}
```

### 31.3 메뉴 자동 생성

장기적으로 Analyze 메뉴는 registry에서 자동 생성한다.

장점:

- 분석 추가 시 UI 메뉴 자동 반영
- 구현/미구현 상태 관리 쉬움
- 도움말 자동 연결 가능

---

## 32. Output Export

### 32.1 HTML Export

HTML 출력은 논문/보고서에 붙여넣기 쉽도록 깔끔한 테이블 스타일을 사용한다.

요구사항:

- 분석별 섹션
- 목차
- 표 caption
- footnote
- syntax block optional
- CSS inline 또는 별도 파일 옵션

### 32.2 Markdown Export

Markdown 출력은 GitHub/문서화용이다.

- 표는 GitHub-flavored Markdown
- 긴 표는 CSV 파일로 분리 가능
- footnote 포함
- code block에 syntax 포함

### 32.3 CSV Tables Export

분석 결과 표를 각각 CSV로 저장한다.

```text
exported_output/
├── 001_descriptives_table.csv
├── 002_ttest_group_statistics.csv
├── 003_ttest_results.csv
└── manifest.json
```

---

## 33. 예제 데이터셋

Hermes Agent는 개발 중 최소 3개의 예제 데이터를 만든다.

### 33.1 sample_clinical.csv

컬럼:

- patient_id
- age
- sex
- treatment_group
- baseline_bp
- followup_bp
- cholesterol
- diabetes
- outcome
- survival_time
- event

용도:

- t-test
- paired t-test
- ANOVA
- logistic regression
- survival later

### 33.2 sample_survey.csv

컬럼:

- respondent_id
- gender
- age_group
- education
- q1
- q2
- q3
- q4
- q5
- satisfaction
- purchase_intent

용도:

- frequencies
- crosstabs
- reliability
- ordinal/nonparametric

### 33.3 sample_regression.csv

컬럼:

- y
- x1
- x2
- x3
- group
- weight

용도:

- correlation
- linear regression
- diagnostics

---

## 34. 문서화

### 34.1 README.md

README에는 다음을 포함한다.

- 제품 소개
- 설치 방법
- 실행 방법
- 지원 파일 형식
- 지원 분석 목록
- 개발 상태
- 테스트 실행 방법
- 라이선스

### 34.2 docs/architecture.md

- 전체 구조
- 데이터 흐름
- 분석 엔진 구조
- UI 구조
- 저장 포맷

### 34.3 docs/variable_model.md

- 변수 메타데이터 설명
- 측정 척도
- 값 라벨
- 결측값
- 타입 추론

### 34.4 docs/statistical_validation.md

- 분석별 공식
- 사용 라이브러리
- 검증 데이터
- 허용 오차
- 알려진 차이

### 34.5 docs/user_manual_mvp.md

- 데이터 불러오기
- 변수 속성 지정
- 빈도분석 실행
- t-test 실행
- 결과 내보내기

---

## 35. CI/CD

초기에는 GitHub Actions를 기준으로 설정한다.

### 35.1 기본 workflow

```yaml
name: tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install -U pip
      - run: pip install -e ".[dev]"
      - run: pytest
      - run: ruff check .
```

GUI 테스트는 headless 환경에서 문제가 생길 수 있으므로 `pytest-qt`와 xvfb 설정을 나중에 추가한다.

---

## 36. 주요 리스크와 대응

### 36.1 통계 결과 신뢰성 리스크

위험:

- scipy/statsmodels 사용법 오류
- 결측 처리 차이
- 범주형 인코딩 오류
- 자유도 계산 오류

대응:

- golden test
- 외부 기준 결과 비교
- 모든 분석에 N과 결측 보고
- 분석 옵션 명시

### 36.2 UI 복잡도 리스크

위험:

- 스프레드시트 UI가 복잡해져 개발 지연
- pandas와 Qt model 동기화 오류

대응:

- MVP에서는 필수 편집만 구현
- undo/redo는 작은 범위부터 시작
- DataFrame 변경 API를 중앙화
- UI model 테스트 작성

### 36.3 범위 폭발 리스크

위험:

- SPSS/MedCalc 전체 기능을 한 번에 구현하려 함

대응:

- Priority A부터 완성
- 미구현 메뉴는 비활성화
- 분석 plugin registry로 확장성 확보
- 각 Phase마다 완료 기준 준수

### 36.4 성능 리스크

위험:

- 큰 데이터셋에서 UI 느려짐

대응:

- Qt model에서 필요한 셀만 로드
- DataFrame copy 최소화
- 대용량 데이터는 Phase 2 이후 lazy 전략
- 상태바에 row/column count 표시
- 긴 분석은 worker thread 사용

---

## 37. Worker Thread 정책

GUI가 멈추지 않도록 분석 실행은 worker로 처리한다.

### 37.1 MVP

- 빠른 분석은 직접 실행 가능
- 1초 이상 걸릴 수 있는 분석은 QThread/QRunnable 사용
- 실행 중 progress dialog 표시

### 37.2 장기

- AnalysisJobQueue
- cancel support
- progress callback
- background result rendering

---

## 38. 내부 에러 처리

### 38.1 예외 계층

```python
class StatWorkbenchError(Exception):
    pass

class DataImportError(StatWorkbenchError):
    pass

class VariableValidationError(StatWorkbenchError):
    pass

class AnalysisValidationError(StatWorkbenchError):
    pass

class AnalysisExecutionError(StatWorkbenchError):
    pass

class ProjectStorageError(StatWorkbenchError):
    pass
```

### 38.2 사용자 메시지

내부 예외는 UI에서 사용자 메시지로 변환한다.

```python
try:
    result = analysis.run(dataset, spec)
except AnalysisValidationError as exc:
    show_user_error("분석 조건을 확인하세요", str(exc))
except Exception as exc:
    log.exception("Unexpected analysis error")
    show_user_error(
        "분석 중 오류가 발생했습니다",
        "데이터와 변수 설정을 확인한 뒤 다시 실행하세요. 자세한 정보는 로그를 확인하세요."
    )
```

---

## 39. 성능 목표

MVP 기준 목표:

| 작업 | 목표 |
|---|---|
| 10,000행 x 50열 CSV 로드 | 5초 이내 |
| Data View 스크롤 | 체감상 즉시 |
| 빈도분석 | 1초 이내 |
| 기술통계 50변수 | 2초 이내 |
| 상관행렬 30변수 | 3초 이내 |
| 프로젝트 저장 | 5초 이내 |

Phase 2 기준:

| 작업 | 목표 |
|---|---|
| 100,000행 x 100열 CSV 로드 | 15초 이내 |
| 회귀분석 100,000행 | 10초 이내 |
| Output HTML export | 5초 이내 |

---

## 40. 접근성

기본 접근성 요구사항:

- 키보드로 메뉴 접근 가능
- 대화상자 tab order 정리
- 색상만으로 오류를 표시하지 않음
- 상태 메시지 텍스트 제공
- 표 복사 가능
- 폰트 크기 설정

---

## 41. 라이선스와 배포

라이선스는 프로젝트 성격에 따라 선택한다.

권장:

- 오픈소스 지향: Apache-2.0 또는 MIT
- 상용 가능성 고려: 내부 비공개 후 추후 결정

패키징:

- 개발 실행: `python -m statworkbench`
- CLI 실행: `statworkbench`
- 데스크톱 배포: PyInstaller
- Windows/macOS/Linux 빌드 스크립트는 Phase 3

---

## 42. Definition of Done

어떤 기능이 완료되려면 다음을 모두 만족해야 한다.

1. 사용자 시나리오가 실제로 동작한다.
2. 단위 테스트가 있다.
3. 오류 케이스 테스트가 있다.
4. UI 연결이 필요한 기능은 메뉴/대화상자에서 접근 가능하다.
5. 결과가 Output View에 표시된다.
6. syntax log가 기록된다.
7. 문서가 업데이트된다.
8. 미구현 또는 제한 사항이 명확히 표시된다.
9. `pytest`가 통과한다.
10. 실행 중 치명적 예외가 발생하지 않는다.

---

## 43. MVP 성공 시나리오

MVP가 성공하려면 다음 시나리오가 모두 가능해야 한다.

### 시나리오 1 — CSV 불러오기와 변수 설정

1. 사용자가 `sample_clinical.csv`를 연다.
2. Import Wizard에서 인코딩과 구분자를 확인한다.
3. 데이터가 Data View에 표시된다.
4. 사용자가 `sex` 변수를 nominal로 설정한다.
5. 사용자가 `treatment_group`에 값 라벨을 지정한다.
6. 사용자가 프로젝트를 저장한다.
7. 다시 열었을 때 변수 설정이 유지된다.

### 시나리오 2 — 빈도분석

1. 사용자가 Analyze > Descriptive Statistics > Frequencies를 선택한다.
2. `sex`, `treatment_group`을 선택한다.
3. OK를 누른다.
4. Output View에 빈도표가 표시된다.
5. 결측값과 valid percent가 표시된다.
6. syntax log가 남는다.

### 시나리오 3 — 독립표본 t 검정

1. 사용자가 Analyze > Compare Means > Independent-Samples t Test를 선택한다.
2. Dependent에 `followup_bp`를 선택한다.
3. Group에 `treatment_group`을 선택한다.
4. 그룹이 2개인지 검증된다.
5. Output View에 그룹 통계, Levene test, t-test 결과가 표시된다.
6. Cohen's d가 표시된다.

### 시나리오 4 — 상관분석

1. 사용자가 Analyze > Correlate > Bivariate를 선택한다.
2. 연속형 변수 여러 개를 선택한다.
3. Pearson/Spearman 옵션을 선택한다.
4. 상관계수 행렬과 p-value 행렬이 출력된다.

### 시나리오 5 — 프로젝트 내보내기

1. 사용자가 분석 결과를 HTML로 내보낸다.
2. 브라우저에서 결과 표가 읽기 좋게 표시된다.
3. Markdown export도 가능하다.

---

## 44. Hermes Agent 금지사항

다음은 피한다.

1. UI만 만들고 분석 엔진을 비워두지 않는다.
2. 분석 엔진을 UI 코드 안에 직접 작성하지 않는다.
3. DataFrame 컬럼과 VariableMeta가 불일치한 상태를 방치하지 않는다.
4. 결측 처리 없이 통계 함수를 직접 호출하지 않는다.
5. p-value만 출력하고 N/효과크기/신뢰구간을 생략하지 않는다.
6. 실패한 분석을 조용히 무시하지 않는다.
7. 사용자의 원본 데이터를 자동으로 덮어쓰지 않는다.
8. 임의 Python 코드 실행을 기본 허용하지 않는다.
9. 테스트 없는 분석 기능을 완료로 표시하지 않는다.
10. 미구현 기능을 구현된 것처럼 메뉴에 활성화하지 않는다.

---

## 45. 바로 구현할 첫 번째 작업 묶음

Hermes Agent가 이 파일을 읽고 바로 시작한다면, 다음 작업부터 수행한다.

### Task 1. 저장소 초기화

- `pyproject.toml`
- `README.md`
- `src/statworkbench`
- `tests`
- `examples`
- `docs`

### Task 2. 핵심 enum/model 구현

- `StorageType`
- `MeasureLevel`
- `VariableRole`
- `MissingRule`
- `VariableMeta`
- `Dataset`

### Task 3. 기본 앱 실행

- `main.py`
- `app.py`
- `MainWindow`
- File/Open placeholder
- Analyze 메뉴 placeholder
- Data/Variable/Output tab placeholder

### Task 4. 테스트

- 변수명 검증
- Dataset 생성
- 변수 추가/삭제/이름 변경
- 결측 규칙
- 앱 import smoke test

### Task 5. 샘플 데이터

- `sample_clinical.csv`
- `sample_survey.csv`
- `sample_regression.csv`

---

## 46. 작업 완료 후 Hermes가 남겨야 하는 보고 형식

Hermes Agent는 큰 작업을 마친 뒤 다음 형식으로 요약한다.

```text
## 완료한 작업
- ...

## 생성/수정한 파일
- ...

## 실행한 검증
- pytest
- app smoke test

## 구현된 기능
- ...

## 아직 남은 제한
- ...

## 다음 추천 작업
- ...
```

---

## 47. 최종 목표 상태

최종적으로 사용자는 다음 방식으로 앱을 사용할 수 있어야 한다.

1. 앱 실행
2. CSV/TXT/Excel 데이터 불러오기
3. Data View에서 데이터 확인
4. Variable View에서 변수 속성 정의
5. Analyze 메뉴에서 분석 선택
6. 변수 선택과 옵션 설정
7. Output View에서 결과 확인
8. 결과 HTML/Markdown/CSV 내보내기
9. 프로젝트 저장
10. 나중에 다시 열어 동일한 분석 맥락 유지

이 목표를 달성하기 전까지, 구현의 우선순위는 항상 “실제 분석 가능한 통계 워크벤치”에 둔다.

---

## 48. 부록 — 분석 메뉴와 구현 상태 표

| 메뉴 | 분석 | MVP | 상태 |
|---|---|---:|---|
| Descriptive Statistics | Frequencies | Yes | 구현 대상 |
| Descriptive Statistics | Descriptives | Yes | 구현 대상 |
| Descriptive Statistics | Explore | Partial | 정규성/기술통계로 시작 |
| Descriptive Statistics | Crosstabs | Yes | 구현 대상 |
| Compare Means | One-Sample t Test | Optional | Phase 2 |
| Compare Means | Independent-Samples t Test | Yes | 구현 대상 |
| Compare Means | Paired-Samples t Test | Yes | 구현 대상 |
| Compare Means | One-Way ANOVA | Yes | 구현 대상 |
| Nonparametric | Mann-Whitney U | Yes | 구현 대상 |
| Nonparametric | Wilcoxon | Yes | 구현 대상 |
| Nonparametric | Kruskal-Wallis | Yes | 구현 대상 |
| Nonparametric | Friedman | Yes | 구현 대상 |
| Correlate | Bivariate | Yes | 구현 대상 |
| Correlate | Partial | No | Phase 2 |
| Regression | Linear | Yes | 구현 대상 |
| Regression | Logistic | Partial | Phase 2 |
| Scale | Reliability | Partial | Phase 2 |
| Diagnostic | ROC | Partial | Phase 2 |
| Survival | Kaplan-Meier | No | Phase 3 |
| Agreement | Kappa/ICC | No | Phase 3 |
| Graphs | Python Visualization | No | Phase 4 |

---

## 49. 부록 — 용어집

| 용어 | 의미 |
|---|---|
| Data View | 실제 데이터를 행과 열로 보는 화면 |
| Variable View | 변수의 속성을 관리하는 화면 |
| VariableMeta | 변수의 통계적 의미와 표시 속성 |
| Storage Type | 실제 저장 타입 |
| Measure Level | 통계 분석을 위한 측정 척도 |
| Value Labels | 숫자/문자 값을 사람이 읽기 쉬운 라벨로 매핑 |
| Missing Rule | 사용자 정의 결측값 규칙 |
| AnalysisSpec | 분석 실행 요청 명세 |
| AnalysisResult | 분석 결과 객체 |
| ResultTable | 결과 표 |
| Syntax Log | GUI 작업을 재현 가능한 명령으로 기록한 로그 |
| Output View | 분석 결과를 누적 표시하는 화면 |

---

## 50. 최종 지시

Hermes Agent는 이 프로젝트를 “통계 계산 함수 모음”이 아니라 “변수 메타데이터 기반 데스크톱 통계 분석 환경”으로 구현한다.

가장 중요한 구현 철학은 다음 세 가지다.

1. **변수의 의미를 먼저 관리한다.**
2. **분석은 변수의 의미에 따라 검증·추천된다.**
3. **모든 결과는 재현 가능하게 기록된다.**

이 원칙을 지키면, 초기 MVP가 작더라도 SPSS/MedCalc형 통계 패키지로 확장 가능한 탄탄한 기반이 된다.

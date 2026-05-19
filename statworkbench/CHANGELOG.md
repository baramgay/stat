# StatWorkbench 변경 이력

모든 주목할 만한 변경 사항은 이 파일에 기록됩니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/) 표준을 따릅니다.

---

## [v2.0.0] - 2026-05-19

### 추가

**고급 분석 5종 신규 구현**

- **로지스틱 회귀** (`logistic_regression.py`)
  - 이항(Binary) 및 다항(Multinomial) 로지스틱 회귀 지원
  - 교차비(Odds Ratio, Exp(B)) 및 95% 신뢰 구간 자동 산출
  - 호스머-레메쇼(Hosmer-Lemeshow) 적합도 검정 포함
  - ROC AUC 및 분류 정확도 보고
- **요인분석** (`factor_analysis.py`)
  - 탐색적 요인분석(EFA) 및 주성분 분석(PCA) 지원
  - Varimax 회전(직교), Oblimin 회전(사각) 선택 가능
  - KMO 표본 적절성 측도, 바틀렛 구형성 검정 포함
  - 스크리 도표 자동 생성
- **군집분석** (`cluster_analysis.py`)
  - K-평균(K-Means): 사용자 정의 군집 수, 반복 상한 설정
  - 계층적 군집(Hierarchical): Ward, Complete, Average, Single 연결 방법
  - 실루엣(Silhouette) 계수 및 엘보우(Elbow) 도표 제공
  - 군집 결과를 새 변수로 데이터셋에 저장
- **생존분석** (`survival_analysis.py`)
  - Kaplan-Meier 생존 함수 추정 및 곡선 시각화
  - Log-rank 검정으로 집단 간 생존 곡선 비교
  - Cox 비례위험 회귀: 위험비(HR), 95% CI, 포레스트 플롯
- **판별분석** (`discriminant_analysis.py`)
  - 선형 판별분석(LDA) 구현
  - 윌크스 람다(Wilks' Lambda), 정준 상관계수 보고
  - 분류 행렬 및 분류 정확도(%) 출력

**SPSS 스프레드시트 대폭 개선**

- Formula Bar 추가: 이름 상자(셀 위치 표시) + 값 입력 바
- 결측값 "." 입력 및 회색 음영 표시 지원
- 소수점 메타데이터 변수 뷰 연동 (소수점 자릿수 데이터 뷰 반영)
- 측정 척도 아이콘 열 머리글 표시 (▪ scale, ● nominal, ◆ ordinal)
- 다중 셀 Ctrl+C / Ctrl+V 복사·붙여넣기
- F2 편집 모드, Delete 셀 삭제, Ctrl+D Fill Down 단축키 추가

**차트 빌더 시스템**

- `chart_builder.py` 완전 재작성: 7종 차트 유형 지원
  - 막대 그래프, 히스토그램, 상자 그림, 산점도, 선 그래프, 파이 차트, 생존 곡선
- `matplotlib` FigureCanvas 기반 실시간 미리보기 패널
- 결과 창 차트 삽입 기능 (`output_view.py` 연동)
- 300 DPI PNG 저장 기능

**새 분석 대화 상자**

- 로지스틱 회귀, 요인분석, K-평균 군집, 계층적 군집, 단일표본 t 검정, 생존분석, 판별분석 대화 상자 신규 구현

**ANOVA 사후 검정**

- Tukey HSD, Bonferroni, Scheffe 사후 검정 지원
- 검정 방법별 다중 비교 결과표 출력

**시스템 통합**

- OUROBOROS v0.39.0 통합

### 개선

- 분석 메뉴 확장: 차원 축소, 군집, 생존분석, 판별분석 서브메뉴 추가
- 분석 레지스트리(`registry.py`)에 로지스틱 회귀 구현 상태 반영
- `Recode`, `Visual Binning` 시그널 연결 버그 수정 (변수 뷰 연동 안정화)
- 단일표본 t 검정 메뉴 항목 분석 엔진과 정상 연결
- CSV 임포트 줄바꿈 문자 호환성 개선 (Windows `\r\n` 자동 처리)
- `output_view.py`: 차트 이미지 블록 삽입 및 우클릭 저장 지원
- 결과 창 트리 목록 항목 아이콘 통일

### 수정

- ANOVA 결과 생성 시 t 검정 대화 상자 참조 오류 수정
- 다크 모드에서 차트 배경색이 흰색으로 고정되던 문제 수정
- Excel 임포트 시 빈 행이 데이터에 포함되던 문제 수정
- 프로젝트 저장(.swb) 후 재열기 시 변수 뷰 소수점 설정이 초기화되던 버그 수정

---

## [v1.0.0] - 2026-05-17

### 추가

**핵심 분석 10종 MVP 구현**

- 빈도분석 (Frequencies): 빈도수, 퍼센트, 유효 퍼센트, 누적 퍼센트
- 기술통계량 (Descriptives): 평균, 표준편차, 최솟값, 최댓값, 왜도, 첨도
- 교차분석 (Crosstabs): 카이제곱 검정, 피셔의 정확 검정
- 독립표본 t 검정: 레빈 등분산 검정, Cohen's d 효과 크기
- 대응표본 t 검정
- 일원분산분석 (One-Way ANOVA): F 통계량, 사후 검정 기반 구조
- 상관분석: Pearson, Spearman, Kendall
- 선형 회귀: R², VIF, 표준화 계수
- 비모수 검정: Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
- 정규성 검정: Shapiro-Wilk, Kolmogorov-Smirnov

**SPSS 스타일 인터페이스**

- Data View: 스프레드시트 형태 데이터 편집, 정렬, 필터
- Variable View: 변수 이름, 유형, 척도, 값 라벨, 결측 설정
- Light / Dark 테마 (Ctrl+Shift+D 전환)

**데이터 관리**

- 프로젝트 저장 / 불러오기: `.swb` 번들 형식 (ZIP + Parquet + JSON)
- CSV, TXT, TSV, Excel (.xlsx) 임포트
- SPSS .sav 임포트 (값 라벨·결측 설정 자동 적용)
- Compute Variable 대화 상자: 수식 기반 파생 변수 생성
- Recode: 다른 변수로 재코딩
- Visual Binning: 히스토그램 기반 범주화

**결과 및 로그**

- 구조화된 결과 창: 트리 목록 + 상세 보기
- HTML 내보내기
- 구문(Syntax) 로그 자동 기록 및 재실행 지원

**테스트**

- pytest 기반 테스트 스위트 550개 항목 100% 통과
- pytest-qt GUI 통합 테스트 포함

---

*이전 버전(pre-release)의 변경 이력은 HERMES.md 명세서를 참조하십시오.*

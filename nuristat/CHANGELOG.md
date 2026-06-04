# NuriStat 변경 이력

모든 주목할 만한 변경 사항은 이 파일에 기록됩니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/) 표준을 따릅니다.

---

## [v3.3.1] - 2026-06-02

### 수정 (Bug Fixes)

- **한글 차트 글리프 누락**: `mixed_anova` · `pca` · `two_way_anova`가 차트를
  `savefig`할 때 한글 폰트가 설정되지 않아 DejaVu Sans로 폴백되어 한글 라벨이
  깨지던 결함 수정. 공용 헬퍼 `analysis/_chart_font.ensure_korean_font()` 도입
- **PCA 대화상자 NameError**: `pca_dialog._setup_ui`가 정의되지 않은 `dataset`을
  참조하여 대화상자 실행 시 `NameError`가 발생하던 버그 수정 (`self._dataset` 사용)

### 변경 (Stabilization & Quality)

- 버전 메타데이터 정합화: `pyproject.toml` · `__init__.py` · README 배지를
  v3.3.1로 동기화 (기존 3.1.0 ↔ CHANGELOG v3.3.0 불일치 해소)
- 린트 정리: `ruff check src/` 무결점 통과 (미사용 임포트·변수 제거, 임포트 정렬,
  E402는 모듈 공통 logger 초기화·`matplotlib.use` 순서 사유로 명시적 예외 처리)
- CI 워크플로 결함 수정: 루트 체크아웃과 패키지 하위 디렉터리(`nuristat/`)
  불일치로 `pip install -e .`가 실패하던 문제를 `working-directory`로 해결
- 추가 검증 테스트 268개(Round 1~4) 보강 — 전체 4420 통과, 커버리지 100% 유지
- 저장소 정리: 임시 산출물·캐시 제거, 개발 추적 문서를 `docs/`·`_archive/`로 재배치

### 품질·배포 (추가)

- **타입 안전성**: mypy 399건 → 0(Success). 핵심 로직 13건 실수정, UI(PySide6 Qt
  스텁 한계)는 검사 제외. `continue-on-error` 제거로 하드 게이트 승격
- **입력 일관성**: normality·frequencies·descriptive·explore가 변수 미지정 시
  무경고 빈 테이블 반환하던 갭 수정 — 전 분석 일관된 경고 반환
- **데이터 입력 최적화**: 대량 붙여넣기 `SPSSGridModel.batch_update()` 도입으로
  셀별 전체 재구축 신호를 1회로 합쳐 7× 가속(O(n²)→O(n))
- **CI 현행화**: 액션 Node 24 최신화(checkout v6·setup-python v6·upload-artifact v7),
  PySide6 헤드리스 시스템 의존성(libegl1 등) 보강, 버전 취약 테스트
  (scheffe·trapezoid·KM 중앙값) 안정화
- **의존성 완결**: tabulate·pyreadstat·wordcloud·python-docx 선언 누락 해소
- **경고 정리**: ml_engine SettingWithCopy, visualization seaborn FutureWarning 제거
- **Windows 실행파일**: PyInstaller spec을 `collect_submodules`로 완결화(139개 모듈),
  빌드·스모크 검증 통과, 용량 최적화 1.1GB → 428MB(61% 감축)
- **최종 검증**: 전 기능 영역 종단 28/28 + 단위 4483 통과, 분석 정확성
  scipy/statsmodels/lifelines 교차검증 일치
- PyPI 메타데이터(keywords·urls·classifier) 보강, release 워크플로 OIDC 전환

---

## [v3.3.0] - 2026-05-29

### 추가

- **MANOVA** (`manova.py`): 다변량 분산분석
  - Pillai's Trace / Wilks' Lambda / Hotelling-Lawley Trace / Roy's Largest Root
  - 단변량 후속 검정 (각 종속변수별 F, 편 η²)
  - Bonferroni / Tukey HSD 사후 검정
  - 대화상자 (`manova_dialog.py`), GLM 메뉴 등록
- **텍스트 마이닝** (`text_mining.py`): 워드클라우드 포함 텍스트 분석
  - 단어 빈도 Top-N (불용어 처리, 최소 길이 필터)
  - 바이그램 / 트라이그램 N-gram 분석
  - TF-IDF 상위 단어 분석
  - 워드클라우드 이미지 (malgun.ttf 자동 탐지)
  - 대화상자 (`text_mining_dialog.py`), 텍스트 마이닝 메뉴 추가
- **프로파일 플롯**: Two-Way ANOVA · Mixed ANOVA에 상호작용 선 그래프 추가
  - `options.profile_plot: True` (기본값)
  - 집단 × 요인 수준별 셀 평균 시각화 (PNG bytes, metadata type="profile_plot")
- **회귀 진단 강화** (`regression.py`)
  - Cook's D + 레버리지 + 표준화 잔차 → 영향력 케이스 진단 표
  - Stepwise / Forward / Backward 변수 선택 (`options.selection_method`)
  - 각 단계별 입력/제거 변수와 p-값 요약 표

### 테스트

- `test_manova.py` 신규 (37개): 구조·다변량 검정·단변량·사후 검정·옵션·입력 검증·결측
- `test_text_mining.py` 신규 (30개): 구조·빈도·N-gram·TF-IDF·옵션·입력 검증·토크나이저
- 전체 테스트: **3,856+ 통과** (예상)

---

## [v3.2.0] - 2026-05-29

### 추가

- **Mixed ANOVA** (`mixed_anova.py`): 집단 간 × 집단 내 혼합 설계 분산분석
  - 집단 간 효과, 집단 내(시점) 효과, 상호작용 효과 검정
  - Mauchly 구형성 검정 + Greenhouse-Geisser / Huynh-Feldt 보정 (재사용)
  - 편 η² (Partial Eta Squared)
  - Bonferroni 사후 검정 (집단 간 ≥ 3, 시점 간 ≥ 3)
  - GLM 메뉴 "혼합 분산분석(Mixed ANOVA)..." 추가
- **Two-Way ANOVA 사후 검정 확장**: Tukey HSD 외 Scheffe / Bonferroni / LSD 선택 가능
  - 대화상자에 사후 검정 방법 콤보박스 추가
- **정규성 검정 메뉴** (`normality_dialog.py`): 기술통계 메뉴에 Shapiro-Wilk 접근 추가

### SPSS 호환성 수정

- **편 η² (Partial Eta Squared)**: Two-Way ANOVA, ANCOVA에서 전체 η² → 편 η²로 수정
  - `편 η² = SS_효과 / (SS_효과 + SS_오차)` — SPSS GLM 기본 출력과 동일
- **`discriminant_analysis.py`**: `dropna()` 후 인덱스 불일치 크래시 수정
- **`logistic_regression.py`**: `"(상수)"` → `"(Constant)"` 일관성 수정

### 테스트

- Scheffe/Bonferroni/LSD 사후 검정 테스트 5개 추가 (`test_two_way_anova.py`)
- 전체 테스트: **3,482+ 통과**

---

## [v3.1.3] - 2026-05-29

### 추가

- **정규성 검정 메뉴** (`normality_dialog.py`): Shapiro-Wilk 검정 대화상자 신규 작성, 기술통계 메뉴에 추가

### 버그 수정 / 호환성

- **`two_way_anova.py`, `ancova.py` η² → 편 η² (Partial Eta Squared)**
  - SPSS GLM은 `편 η² = SS_효과/(SS_효과+SS_오차)`를 기본 출력 — 전체 η²에서 수정
  - 테이블 컬럼명 `η²` → `편 η²`, 각주 추가
- **`discriminant_analysis.py`**: `dropna()` 후 `y_encoded` 인덱스 불일치 크래시 수정 (길이 불일치 boolean indexing 오류 제거)
- **`logistic_regression.py`**: `"(상수)"` → `"(Constant)"` (regression.py 와 일관성)
- 연결 없는 `"평균..."` 메뉴 항목 제거

### 분석 모듈 추가

- **ANCOVA** (`ancova.py`): 공분산분석 — 공변량 조정 후 요인 효과 검정, Type III SS(Sum coding), EMM, Bonferroni 사후 검정, 편 η²
- ANCOVA 대화상자 (`ancova_dialog.py`), GLM 메뉴 등록, 테스트 38개

### 테스트

- `test_ancova.py` 신규 (38개): 구조·통계·옵션·다중 공변량·입력 검증·결측 처리
- 편 η² 변경으로 `test_two_way_anova.py` 테스트 개선 (합계≤1 → 개별 [0,1] 범위 검증)
- 전체 테스트: **3,482 통과**

---

## [v3.1.2] - 2026-05-29

### 추가 / 수정 (v3.1.1 이후)

- `ancova.py`: ANCOVA 분석 모듈
- `ancova_dialog.py`: ANCOVA 대화상자
- `discriminant_analysis.py` 인덱스 버그 수정
- `factor_analysis.py` 설명 분산 % 계산 오류 (`len` → `sum`)
- `registry.py` ANCOVA 등록

---

## [v3.1.1] - 2026-05-29

### 버그 수정

- **`repeated_measures_dialog.py`**: `→` 버튼으로 같은 변수를 여러 번 추가할 수 있던 중복 선택 버그 수정
- **`repeated_measures_anova.py`**: 공분산 행렬이 특이(singular)할 때 `det(S) < 1e-15` 경우 미처리 → W=1.0(구형성 충족)으로 안전 처리
- **`two_way_anova.py`**: `factor_a == factor_b` 동일 변수 지정 시 pandas `AttributeError` 크래시 → 사용자 친화적 경고 반환

### 개선

- **`two_way_anova.py` SPSS 호환성**: `C(var)` treatment coding → `C(var, Sum)` deviation coding 변경
  - Type III SS가 SPSS 출력과 동일해짐 (예: SS_A 62.5 → 187.5, SS 분해 완전 일치)
- **`two_way_anova.py` 빈 셀 경고**: n=0 셀(불균형 설계) 및 n=1 셀 경고 추가

### 테스트

- `test_two_way_anova.py` 신규 (47개): SS 분해, Tukey HSD, 효과크기, 빈 셀, 결측값 등
- `test_repeated_measures_anova.py` 신규 (47개): Mauchly W, GG/HF 보정, Bonferroni 쌍 비교 등
- 전체 테스트: **3,707 통과** (이전 3,613 → +94건)

---

## [v3.1.0] - 2026-05-29

### 추가

- **이원분산분석** (`two_way_anova.py`): 주 효과 및 상호작용 효과 검정, 기술통계, 개체-간 효과 테이블, Tukey HSD 사후 검정 포함
- **반복측정 ANOVA** (`repeated_measures_anova.py`): 구형성 검정(Mauchly's W), Greenhouse-Geisser / Huynh-Feldt 보정, 일원/이원 반복측정 지원

### 개선

**수치 정확성 — R 4.6.0 대비 전항목 검증 완료**
- 20개 분석 모듈, 121개 수치 항목 R 4.6.0 출력과 ±0.5% 이내 일치 확인
- 검증 스크립트 `validation/run_validation.py` 및 R 기준 스크립트 `validation/r_reference.R` 추가

**케이스 처리 요약(CPS) 표준화**
- `get_cps_table_kr()` 전 모듈 통합: `"결측됨"` → `"제외됨"` 컬럼 값 통일
- reliability, cohens_kappa, partial_correlation, icc, chi_square_gof 모듈 임포트 누락 수정

**테스트 suite**
- 3,613 통과 (이전 2,988 → +625건)
- CPS 관련 4개 테스트 파일 `"제외됨"` 표준 반영

---

## [v3.0.0] - 2026-05-27

### 추가

**신규 분석 모듈 8종**

- **편상관 분석** (`partial_correlation.py`)
  - 제3 변수(통제 변수)를 제거한 순수 상관계수 산출
  - Pearson / Spearman 방법 선택 지원
  - 편상관 행렬 및 유의확률 일괄 출력
- **신뢰도 분석** (`reliability.py`)
  - Cronbach's α 계수 및 95% 신뢰 구간
  - 항목 제거 시 α(Alpha-if-deleted), 항목-전체 상관 일괄 출력
  - Split-half 신뢰도, Guttman λ6 지원
- **카이제곱 적합도 검정** (`chi_square_gof.py`)
  - 관찰 빈도 vs. 균등·사용자 정의 기대 빈도 비교
  - χ² 통계량, 자유도, p값, 잔차 출력
- **ROC 분석** (`roc_analysis.py`)
  - AUC와 95% 신뢰 구간(DeLong 방법)
  - 최적 절단점 자동 탐색 (Youden Index)
  - ROC 곡선 시각화 및 민감도·특이도 테이블
- **민감도·특이도 분석** (`sensitivity_specificity.py`)
  - 혼동 행렬(Confusion Matrix) 전체 지표 산출
  - 양성·음성 우도비(Likelihood Ratio) 포함
  - 95% 신뢰 구간(Clopper-Pearson) 자동 산출
- **Cohen's Kappa** (`cohens_kappa.py`)
  - 범주형 평가자 간 일치도(κ) 및 가중 Kappa
  - 교차표 기반 관찰 일치율·기대 일치율 분해 출력
- **급내상관계수 (ICC)** (`icc.py`)
  - ICC(1,1), ICC(2,1), ICC(3,1) 유형 선택
  - 95% 신뢰 구간 및 F 검정 포함
- **블랜드-알트만 도표** (`bland_altman.py`)
  - 평균 차이(Bias) 및 ±1.96 SD 일치 한계 산출
  - 비례 편향(Proportional Bias) 회귀선 옵션

**Engine 래퍼 클래스 6종 추가**

- `TtestEngine`, `DescriptiveEngine`, `AnovaEngine`, `CorrelationEngine`, `RegressionEngine`, `FrequenciesEngine`
- `AnalysisPlugin` 프로토콜을 완전히 준수하는 구조체 클래스
- 분석 레지스트리 통합 완료

### 개선

**코드 품질**

- ruff 정적 분석 위반 0건 달성 (F401, F841, E731, E741, W293 전수 수정)
- `# pragma: no cover` 마킹으로 GUI 진입점 및 도달 불가 코드 명시 처리
- `pyproject.toml` ruff 설정 `[tool.ruff.lint]` 섹션으로 마이그레이션

**테스트 품질**

- pytest 통과 건수: 2,988개 (이전 대비 +200건 이상)
- 코드 커버리지: 98% 달성
- 통합 테스트(`tests/integration/`) 신규 10개 분석 모듈 추가

**문서**

- 사용자 매뉴얼 v3.0.0: 편상관, 신뢰도, ROC, 민감도/특이도, Kappa, ICC, 블랜드-알트만, 기계학습, 탐색, 데이터 품질 진단 섹션 추가
- README.md: MVP 16개 → 37개 분석 모듈 전체 목록으로 업데이트
- 부록 B(FAQ) 신규 추가

### 수정

- `explore.py`: `MissingPolicy` 미사용 임포트 및 `missing_policy` 미사용 변수 블록 제거
- `survival_analysis.py`: `alpha`, `sf`, `ci` 미사용 변수 제거
- `logistic_regression.py`: 미사용 Wald p값 계산 블록 제거
- `factor_analysis.py`: `total_var` 미사용 변수 제거
- `discriminant_analysis.py`: E741 루프 변수명 충돌 수정

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

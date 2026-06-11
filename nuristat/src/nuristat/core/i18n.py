"""Language management for NuriStat.

Two layers:
- Analysis output layer: internal DataFrames stay in English; only the display/export
  step (ResultTable.to_html etc.) translates when lang=="ko".
- UI layer: t(en_text) returns the English string by default; when lang=="ko" it
  returns the Korean equivalent from UI_KO if available.

Default language: "en". The app loads the persisted setting on startup.
"""
from __future__ import annotations

import pandas as pd

_lang: str = "en"


def set_language(lang: str) -> None:
    """Set the active language ("en" | "ko")."""
    global _lang
    _lang = "ko" if str(lang).lower().startswith("ko") else "en"


def get_language() -> str:
    return _lang


# ---------------------------------------------------------------------------
# UI string translation — English is the canonical key.
# t("File") → "File" (en) or "파일" (ko).
# ---------------------------------------------------------------------------

#: Korean translations for UI strings.  Key = English text, value = Korean.
UI_KO: dict[str, str] = {
    # ── File menu ──────────────────────────────────────────────────────────
    "File(&F)": "파일(&F)",
    "New Project": "새 프로젝트",
    "🆕 New Project": "🆕 새 프로젝트",
    "Open Project...": "프로젝트 열기...",
    "📂 Open Project...": "📂 프로젝트 열기...",
    "Save Project": "프로젝트 저장",
    "💾 Save Project": "💾 프로젝트 저장",
    "Save Project As...": "다른 이름으로 저장...",
    "💾 Save Project As...": "💾 다른 이름으로 저장...",
    "Import(&I)": "가져오기(&I)",
    "📥 Import(&I)": "📥 가져오기(&I)",
    "📄 CSV / Text...": "📄 CSV / 텍스트...",
    "📊 Excel File...": "📊 Excel 파일...",
    "📋 SPSS File (.sav)...": "📋 SPSS 파일 (.sav)...",
    "📋 Clipboard...": "📋 클립보드...",
    "Export(&X)": "내보내기(&X)",
    "📤 Export(&X)": "📤 내보내기(&X)",
    "📄 CSV File...": "📄 CSV 파일...",
    "📊 Excel File (export)...": "📊 Excel 파일...",
    "📋 SPSS File (.sav) export...": "📋 SPSS 파일 (.sav)...",
    "🕘 Recent Files(&R)": "🕘 최근 파일(&R)",
    "🚪 Exit": "🚪 끝내기",
    # ── Edit menu ──────────────────────────────────────────────────────────
    "✏️ Edit(&E)": "✏️ 편집(&E)",
    "↩️ Undo": "↩️ 실행 취소",
    "↪️ Redo": "↪️ 다시 실행",
    "✂️ Cut": "✂️ 잘라내기",
    "📋 Copy": "📋 복사",
    "📋 Paste": "📋 붙여넣기",
    "☑️ Select All": "☑️ 모두 선택",
    "Find...": "찾기...",
    # ── View menu ──────────────────────────────────────────────────────────
    "👁️ View(&V)": "👁️ 보기(&V)",
    "🌙 Dark Mode": "🌙 다크 모드",
    "🌐 Output Language": "🌐 분석 결과 언어",
    "🌐 UI Language": "🌐 UI 언어",
    "🔢 Data View": "🔢 데이터 보기",
    "📋 Variable View": "📋 변수 보기",
    "📝 Syntax Editor": "📝 구문 편집기",
    "🏷️ Show Value Labels": "🏷️ 값 라벨 표시",
    "📊 Show Output Window": "📊 결과 창 보기",
    # ── Data menu ──────────────────────────────────────────────────────────
    "Data(&D)": "데이터(&D)",
    "🔍 Select Cases...": "🔍 케이스 선택...",
    "⚖️ Weight Cases...": "⚖️ 케이스 가중치...",
    "🔀 Sort Cases...": "🔀 케이스 정렬...",
    "↔️ Transpose...": "↔️ 행렬 전치...",
    "🔗 Merge Files...": "🔗 파일 병합...",
    "📊 Pivot Table...": "📊 피벗 테이블...",
    # ── Transform menu ─────────────────────────────────────────────────────
    "Transform(&T)": "변환(&T)",
    "🔢 Compute Variable...": "🔢 변수 계산...",
    "🔄 Recode Variable...": "🔄 변수 재코딩...",
    "📊 Visual Binning...": "📊 시각적 구간화...",
    "🏆 Rank Cases...": "🏆 순위 계산...",
    # ── Analyze menu ───────────────────────────────────────────────────────
    "📊 Analyze(&A)": "📊 분석(&A)",
    "🔧 Run Script...": "🔧 스크립트 실행...",
    "📈 Descriptive Statistics(&R)": "📈 기술통계(&R)",
    "📊 Frequencies...": "📊 빈도...",
    "📈 Descriptives...": "📈 기술통계량...",
    "🔍 Explore...": "🔍 탐색...",
    "📊 Crosstabulation...": "📊 교차분석...",
    "📐 Normality Test (Shapiro-Wilk)...": "📐 정규성 검정(Shapiro-Wilk)...",
    "🔄 Compare Means(&M)": "🔄 평균 비교(&M)",
    "1️⃣ One-Sample T Test...": "1️⃣ 단일표본 T 검정...",
    "2️⃣ Independent-Samples T Test...": "2️⃣ 독립표본 T 검정...",
    "🔗 Paired-Samples T Test...": "🔗 대응표본 T 검정...",
    "📊 One-Way ANOVA...": "📊 일원분산분석...",
    "📊 General Linear Model(&G)": "📊 일반선형모형(&G)",
    "📊 Two-Way ANOVA (Univariate)...": "📊 이원분산분석(Univariate)...",
    "🔄 Repeated Measures...": "🔄 반복측정...",
    "📊 ANCOVA...": "📊 ANCOVA(공분산분석)...",
    "🔀 Mixed ANOVA...": "🔀 혼합 분산분석(Mixed ANOVA)...",
    "📊 MANOVA...": "📊 MANOVA(다변량 분산분석)...",
    "🔗 Correlate(&C)": "🔗 상관(&C)",
    "🔗 Bivariate Correlation...": "🔗 상관분석...",
    "🔗 Partial Correlation...": "🔗 편상관...",
    "📈 Regression(&R)": "📈 회귀(&R)",
    "📈 Linear...": "📈 선형...",
    "📊 Logistic...": "📊 로지스틱...",
    "📊 Multinomial Logistic...": "📊 다항 로지스틱...",
    "📉 Dimension Reduction(&D)": "📉 차원 축소(&D)",
    "📉 Factor Analysis...": "📉 요인분석...",
    "📉 Principal Component Analysis (PCA)...": "📉 주성분분석 (PCA)...",
    "🔵 Cluster(&K)": "🔵 군집(&K)",
    "🔵 K-Means Cluster...": "🔵 K-평균 군집...",
    "🔵 Hierarchical Cluster...": "🔵 계층적 군집...",
    "Survival Analysis(&S)": "생존분석(&S)",
    "📉 Cox Proportional Hazards Regression...": "📉 Cox 비례위험 회귀...",
    "🔷 Discriminant Analysis(&I)": "🔷 판별분석(&I)",
    "🔷 Discriminant Analysis...": "🔷 판별분석...",
    "🧪 Nonparametric Tests(&N)": "🧪 비모수 검정(&N)",
    "🧪 Nonparametric Tests...": "🧪 비모수 검정...",
    "🧮 Chi-Square Goodness-of-Fit...": "🧮 카이제곱 적합도...",
    "🔬 Diagnostic Tests(&T)": "🔬 진단 검정(&T)",
    "📈 ROC Analysis...": "📈 ROC 분석...",
    "✅ Agreement Analysis(&G)": "✅ 일치도 분석(&G)",
    "📊 ICC (Intraclass Correlation)...": "📊 급내 상관계수(ICC)...",
    "📐 Scale Analysis(&S)": "📐 척도 분석(&S)",
    "🔁 Reliability Analysis (Cronbach α)...": "🔁 신뢰도 분석(Cronbach α)...",
    "📝 Text Mining(&X)": "📝 텍스트 마이닝(&X)",
    "📝 Text Mining (Word Cloud)...": "📝 텍스트 마이닝(워드클라우드)...",
    "🤖 Machine Learning...": "🤖 기계학습...",
    # ── Graphs menu ────────────────────────────────────────────────────────
    "Graphs(&G)": "차트(&G)",
    "📊 Advanced Visualization...": "📊 고급 시각화...",
    "Chart Builder...": "차트 빌더...",
    "Legacy Dialogs(&L)": "기존 대화상자(&L)",
    "Bar...": "막대...",
    "Line...": "선...",
    "Scatter...": "산점도...",
    "Histogram...": "히스토그램...",
    "Box Plot...": "상자 그림...",
    # ── Utilities menu ─────────────────────────────────────────────────────
    "Utilities(&U)": "유틸리티(&U)",
    "🔍 Data Quality Diagnosis...": "🔍 데이터 품질 진단...",
    "📄 Report Generator...": "📄 보고서 생성...",
    "📋 Variable Information...": "📋 변수 정보...",
    "📁 File Information...": "📁 파일 정보...",
    # ── Window / Help menu ─────────────────────────────────────────────────
    "Window(&W)": "창(&W)",
    "Help(&H)": "도움말(&H)",
    "About NuriStat": "프로그램 정보",
    "User Manual": "사용자 매뉴얼",
    # ── Toolbar ────────────────────────────────────────────────────────────
    "Main Toolbar": "메인 도구 모음",
    "🆕 New": "🆕 새 파일",
    "📂 Open": "📂 열기",
    "💾 Save": "💾 저장",
    "📥 Import": "📥 가져오기",
    "🔢 Data": "🔢 데이터",
    "📋 Variables": "📋 변수",
    "📊 Frequencies": "📊 빈도",
    "📈 Descriptives": "📈 기술통계",
    "📉 T Test": "📉 T 검정",
    "📉 Regression": "📉 회귀",
    # ── Status bar / tabs ──────────────────────────────────────────────────
    "Data View": "데이터 보기",
    "Variable View": "변수 보기",
    "Syntax Editor": "구문 편집기",
    "Variables": "변수",
    "Ready": "준비",
    "No dataset loaded": "데이터 없음",
    # ── Common dialog messages ─────────────────────────────────────────────
    "Warning": "경고",
    "Error": "오류",
    "Information": "정보",
    "OK": "확인",
    "Cancel": "취소",
    "Yes": "예",
    "No": "아니오",
    "Close": "닫기",
    "Apply": "적용",
    "Run": "실행",
    "Please load a dataset first.": "먼저 데이터를 불러오세요.",
    "No data loaded": "데이터가 없습니다",
    # ── Settings dialog ────────────────────────────────────────────────────
    "Settings": "설정",
    "Language": "언어",
    "UI Language:": "UI 언어:",
    "Output Language:": "분석 결과 언어:",
    "Theme": "테마",
    "Dark mode": "다크 모드",
}


def t(en_text: str) -> str:
    """Translate a UI string.

    Returns *en_text* unchanged when the active language is "en".
    Returns the Korean equivalent from UI_KO when lang=="ko", falling back to
    *en_text* if no translation exists yet.
    """
    if _lang != "ko":
        return en_text
    return UI_KO.get(en_text, en_text)


# 분석 테이블 제목 (영어 → 한국어)
TITLES: dict[str, str] = {
    "2×2 Contingency Table": "2×2 분할표",
    "Agreement Statistics": "일치도 통계량",
    "ANCOVA": "공분산분석 (ANCOVA)",
    "ANOVA": "분산분석 (ANOVA)",
    "Area Under the Curve": "곡선하면적 (AUC)",
    "Autocorrelation Test": "자기상관 검정",
    "Bland-Altman Statistics": "Bland-Altman 통계량",
    "Case Processing Summary": "케이스 처리 요약",
    "Chi-Square Goodness-of-Fit": "카이제곱 적합도 검정",
    "Coefficients": "계수",
    "Cohen's Kappa": "Cohen의 카파",
    "Collinearity Diagnostics (VIF)": "공선성 진단 (VIF)",
    "Correlation Analysis": "상관분석",
    "Crosstabulation": "교차표",
    "Descriptive Statistics": "기술통계량",
    "Descriptives": "기술통계",
    "Diagnostic Accuracy Measures": "진단 정확도 지표",
    "Dummy Coding": "더미 코딩",
    "Estimated Marginal Means": "추정 주변 평균",
    "Extreme Values": "극단값",
    "Frequencies": "빈도분석",
    "Frequency Table": "빈도표",
    "Group Statistics": "집단 통계량",
    "ICC": "급내상관계수 (ICC)",
    "Independent Samples t-Test": "독립표본 t 검정",
    "Individual Differences": "개별 차이",
    "Interpretation": "해석",
    "Intraclass Correlation Coefficient": "급내상관계수",
    "Item Statistics": "문항 통계량",
    "Item-Total Statistics": "문항-전체 통계량",
    "Levene's Test of Equality of Error Variances": "오차분산 동질성에 대한 Levene 검정",
    "Likelihood Ratios": "우도비",
    "Limits of Agreement": "일치 한계",
    "Linear Regression": "선형 회귀",
    "Logistic Regression": "로지스틱 회귀",
    "Mauchly's Test of Sphericity": "Mauchly의 구형성 검정",
    "Model Summary": "모형 요약",
    "N Matrix": "N 행렬",
    "Nonparametric Test": "비모수 검정",
    "Normality Test (Shapiro-Wilk)": "정규성 검정 (Shapiro-Wilk)",
    "One-Sample Statistics": "일표본 통계량",
    "One-Sample T Test": "일표본 t 검정",
    "One-Sample t-Test": "일표본 t 검정",
    "One-Way ANOVA": "일원배치 분산분석",
    "Optimal Cutoff": "최적 절단점",
    "Paired Samples Statistics": "대응표본 통계량",
    "Paired Samples t-Test": "대응표본 t 검정",
    "Pairwise Comparisons (Bonferroni)": "대응별 비교 (Bonferroni)",
    "Pairwise Correlations": "대응별 상관",
    "Partial Correlation": "편상관",
    "Percentiles": "백분위수",
    "Post-Hoc: Bonferroni": "사후검정: Bonferroni",
    "Post-Hoc: Scheffe": "사후검정: Scheffé",
    "Post-Hoc: Tukey HSD": "사후검정: Tukey HSD",
    "p-value Matrix": "p값 행렬",
    "Ranks": "순위",
    "Reliability Analysis": "신뢰도 분석",
    "Reliability Statistics": "신뢰도 통계량",
    "Repeated Measures ANOVA": "반복측정 분산분석",
    "Residual Summary": "잔차 요약",
    "Residuals": "잔차",
    "ROC Coordinates": "ROC 좌표",
    "ROC Curve Analysis": "ROC 곡선 분석",
    "Scale Statistics": "척도 통계량",
    "Sensitivity / Specificity Analysis": "민감도 / 특이도 분석",
    "Significance (2-tailed)": "유의확률 (양측)",
    "Symmetric Measures": "대칭적 측도",
    "Test for Equality of Variances": "분산 동질성 검정",
    "Test of Homogeneity of Regression Slopes": "회귀기울기 동질성 검정",
    "Test of Homogeneity of Variances": "분산 동질성 검정",
    "Test Statistics": "검정 통계량",
    "Tests of Between-Subjects Effects": "개체간 효과 검정",
    "Tests of Normality": "정규성 검정",
    "Tests of Within-Subjects Effects": "개체내 효과 검정",
    "t-Test": "t 검정",
    "Two-Way ANOVA": "이원배치 분산분석",
    "Welch ANOVA (for unequal variances)": "Welch 분산분석 (이분산)",
    "Zero-order Correlations": "0차 상관",
}

# 컬럼 헤더·행 라벨 (영어 → 한국어). 통계 표기(F, t, B, N, R 등)는 보편적이라 유지.
TERMS: dict[str, str] = {
    "Variable": "변수", "Variable Pair": "변수쌍", "Variant": "구분",
    "Statistic": "통계량", "Value": "값", "Source": "소스",
    "Mean": "평균", "Median": "중앙값", "Mode": "최빈값",
    "SD": "표준편차", "Std": "표준편차", "Std.": "표준편차",
    "Std. Deviation": "표준편차", "Std Error": "표준오차", "SE": "표준오차",
    "Min": "최솟값", "Max": "최댓값", "Minimum": "최솟값", "Maximum": "최댓값",
    "Range": "범위", "Sum": "합계", "Count": "개수",
    "IQR": "사분위범위", "Skewness": "왜도", "Kurtosis": "첨도",
    "Variance": "분산", "df": "자유도", "df1": "자유도1", "df2": "자유도2",
    "p-value": "유의확률", "Sig.": "유의확률", "Sig": "유의확률",
    "p-adj": "수정 유의확률", "p-value (2-tailed)": "유의확률(양측)",
    "SS": "제곱합", "MS": "평균제곱", "Sum of Squares": "제곱합", "Mean Square": "평균제곱",
    "Coefficient": "계수", "Beta": "베타", "Std. Beta": "표준화 베타",
    "R-squared": "R 제곱", "Adjusted R-squared": "수정된 R 제곱",
    "Eta-squared": "에타제곱", "Partial Eta-squared": "부분 에타제곱",
    "Omega-squared": "오메가제곱", "Epsilon-squared": "엡실론제곱",
    "Cohen's d": "Cohen의 d", "Effect Size": "효과크기",
    "95% CI": "95% 신뢰구간", "CI": "신뢰구간", "Lower": "하한", "Upper": "상한",
    "Mean Difference": "평균 차이", "SE Difference": "차이 표준오차",
    "Frequency": "빈도", "Percent": "퍼센트", "Valid Percent": "유효 퍼센트",
    "Cumulative Percent": "누적 퍼센트", "Group": "집단", "Group1": "집단1", "Group2": "집단2",
    "Test": "검정", "Total Cases": "전체 케이스", "Valid Cases": "유효 케이스",
    "Excluded Cases": "제외 케이스", "Excluded %": "제외 %", "Missing": "결측",
    "N": "N", "Interpretation": "해석", "Decision": "판정",
    "Sensitivity": "민감도", "Specificity": "특이도", "Accuracy": "정확도",
    "Precision": "정밀도", "Recall": "재현율", "AUC": "AUC",
    "Cutoff": "절단점", "Threshold": "임계값",
    "Eigenvalue": "고유값", "Communality": "공통성", "Loading": "적재량",
    "Cronbach's Alpha": "Cronbach 알파", "Corrected Item-Total Correlation": "교정 문항-전체 상관",
    "reject": "기각", "meandiff": "평균차이",
    "Observed": "관측빈도", "Expected": "기대빈도", "Residual": "잔차",
    "Rank": "순위", "Mean Rank": "평균순위", "Sum of Ranks": "순위합",
}


def tr_title(title: str) -> str:
    """테이블 제목 번역 (현재 언어가 ko이고 사전에 있으면)."""
    if _lang == "ko":
        return TITLES.get(title, title)
    return title


def tr_term(term: str) -> str:
    """컬럼명·라벨 번역."""
    if _lang == "ko":
        return TERMS.get(str(term), str(term))
    return str(term)


def tr_frame(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 컬럼·인덱스·라벨성 셀 값을 번역한 사본 반환 (en이면 원본 그대로).

    데이터 무결성을 위해 원본은 변경하지 않으며, 표시/내보내기용 사본만 만든다.
    셀 값은 TERMS에 정확히 일치하는 라벨(예: 'Mean','t','p-value')만 치환한다.
    """
    if _lang != "ko" or df is None or df.empty:
        return df
    out = df.copy()
    out.columns = [tr_term(c) for c in out.columns]
    if out.index.dtype == object:
        out.index = [tr_term(i) for i in out.index]
    # 라벨성 첫 열(object) 셀 값 치환 — 정확 일치하는 통계 용어만
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: TERMS.get(v, v) if isinstance(v, str) else v)
    return out

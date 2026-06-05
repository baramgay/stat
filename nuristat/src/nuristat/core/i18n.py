"""분석 결과 출력 언어 전환 (한국어/영어).

비파괴 방식: 분석 결과의 내부 DataFrame(컬럼명·값)은 영어로 유지하고,
표시·내보내기 단계(ResultTable.to_html 등)에서만 번역을 적용한다.
기본 언어는 "en"이며 앱은 설정에 따라 "ko"를 사용한다.
"""
from __future__ import annotations

import pandas as pd

_lang: str = "en"


def set_language(lang: str) -> None:
    """출력 언어 설정 ("ko" | "en")."""
    global _lang
    _lang = "ko" if str(lang).lower().startswith("ko") else "en"


def get_language() -> str:
    return _lang


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

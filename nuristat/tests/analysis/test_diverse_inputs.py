"""다양한 입력 형태에 대한 견고성 테스트.

문자열, 불리언, 정수, 부동소수, 혼합 타입, 유니코드, 극단값 등
실제 사용자가 입력할 수 있는 모든 형태의 값에 대해 각 분석 모듈이
정상 동작하거나 graceful AnalysisResult를 반환하는지 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
src_path = str(_PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset

import nuristat.analysis.descriptive as mod_descriptive
import nuristat.analysis.frequencies as mod_frequencies
import nuristat.analysis.ttests as mod_ttests
import nuristat.analysis.anova as mod_anova
import nuristat.analysis.correlation as mod_correlation
import nuristat.analysis.regression as mod_regression
import nuristat.analysis.normality as mod_normality
import nuristat.analysis.crosstab as mod_crosstab


def ds(df: pd.DataFrame) -> Dataset:
    return Dataset(data=df, name="test")


# ──────────────────────────────────────────────────────────────────────────────
# 1. 정수형 입력
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegerInputs:
    def test_descriptive_int_column(self):
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["score"]}})
        assert isinstance(result, AnalysisResult)

    def test_descriptive_zero_and_negative_int(self):
        df = pd.DataFrame({"x": [-100, -50, 0, 50, 100]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_regression_int_columns(self):
        df = pd.DataFrame({
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        })
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 부동소수점 극단값
# ──────────────────────────────────────────────────────────────────────────────

class TestFloatEdgeCases:
    @pytest.mark.parametrize("values", [
        [1e-300, 2e-300, 3e-300],
        [1e300, 2e300, 3e300],
        [1e-10, 1e10, -1e10],
        [0.1 + 0.2, 0.3, 0.4],  # 부동소수 정밀도 이슈
    ])
    def test_extreme_floats_descriptive(self, values):
        df = pd.DataFrame({"x": values + [0.0] * (10 - len(values))})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_nan_inf_mixed(self):
        df = pd.DataFrame({"x": [1.0, float("inf"), float("-inf"), float("nan"), 2.0]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_very_small_variance(self):
        """분산이 거의 0인 경우."""
        df = pd.DataFrame({"x": [1.0000001 * i for i in range(1, 21)]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 문자열 입력
# ──────────────────────────────────────────────────────────────────────────────

class TestStringInputs:
    def test_frequencies_text_column(self):
        df = pd.DataFrame({"city": ["서울", "부산", "대구", "서울", "부산", "서울"]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["city"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_frequencies_mixed_case_strings(self):
        df = pd.DataFrame({"cat": ["A", "a", "A", "B", "b", "B", "C"]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["cat"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_frequencies_special_chars(self):
        df = pd.DataFrame({"val": ["a&b", "c>d", "e<f", "<null>", "N/A", ""]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["val"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_frequencies_numeric_strings(self):
        """숫자처럼 보이는 문자열."""
        df = pd.DataFrame({"x": ["1.5", "2.3", "3.7", "1.5", "2.3"]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_crosstab_string_categories(self):
        df = pd.DataFrame({
            "row": ["남자", "여자", "남자", "여자", "남자", "여자"] * 5,
            "col": ["찬성", "반대", "중립", "찬성", "반대", "중립"] * 5,
        })
        spec = {"variables": {"row": "row", "column": "col"}}
        result = mod_crosstab.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_descriptive_numeric_strings_coerced(self):
        """문자열을 숫자로 변환하여 분석 시도."""
        df = pd.DataFrame({"x": ["1", "2", "3", "4", "5"]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 불리언 입력
# ──────────────────────────────────────────────────────────────────────────────

class TestBooleanInputs:
    def test_frequencies_boolean(self):
        df = pd.DataFrame({"flag": [True, False, True, True, False]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["flag"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_descriptive_boolean(self):
        df = pd.DataFrame({"flag": [True, False, True, True, False, True]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["flag"]}})
        assert isinstance(result, AnalysisResult)

    def test_crosstab_boolean_vs_string(self):
        df = pd.DataFrame({
            "passed": [True, False, True, False, True, False] * 5,
            "group": ["A", "B", "A", "B", "A", "B"] * 5,
        })
        spec = {"variables": {"row": "passed", "column": "group"}}
        result = mod_crosstab.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 5. 혼합 타입 (object dtype)
# ──────────────────────────────────────────────────────────────────────────────

class TestMixedTypes:
    def test_frequencies_mixed_num_str(self):
        """숫자와 문자가 섞인 컬럼."""
        df = pd.DataFrame({"x": [1, "a", 2.5, "b", None, True]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_descriptive_mixed_graceful(self):
        df = pd.DataFrame({"x": [1, "hello", 3.5, None, True, "world"]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_anova_mixed_group_var(self):
        """집단변수가 숫자/문자 혼합."""
        df = pd.DataFrame({
            "y": np.random.default_rng(0).normal(size=20),
            "g": [1, "A", 2, "B", 1, "A", 2, "B"] * 2 + [1, "A", 2, "B"],
        })
        spec = {"variables": {"dependent": "y", "factor": "g"}}
        result = mod_anova.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 6. 유니코드 / 특수 문자 컬럼명
# ──────────────────────────────────────────────────────────────────────────────

class TestUnicodeAndSpecialChars:
    def test_korean_variable_names_descriptive(self):
        df = pd.DataFrame({
            "나이": np.random.default_rng(0).integers(20, 80, 30).astype(float),
            "소득": np.random.default_rng(1).integers(1000, 10000, 30).astype(float),
        })
        dataset = ds(df)
        col0, col1 = dataset.data.columns[0], dataset.data.columns[1]
        result = mod_descriptive.run_analysis(
            dataset, {"variables": {"scale": [col0, col1]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_emoji_in_values_frequencies(self):
        df = pd.DataFrame({"cat": ["😀", "😢", "😀", "😐", "😢", "😀"]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["cat"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_newline_tab_in_values_frequencies(self):
        df = pd.DataFrame({"x": ["a\nb", "c\td", "e\nf", "a\nb", "c\td"]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 7. 결측값 다양한 패턴
# ──────────────────────────────────────────────────────────────────────────────

class TestMissingValuePatterns:
    def test_alternating_nan_descriptive(self):
        """홀수 행만 결측."""
        vals = [float("nan") if i % 2 == 0 else float(i) for i in range(20)]
        df = pd.DataFrame({"x": vals})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_all_nan_normality(self):
        df = pd.DataFrame({"x": [float("nan")] * 10})
        result = mod_normality.run_analysis(ds(df), {"variables": {"target": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_single_valid_value(self):
        vals = [float("nan")] * 9 + [42.0]
        df = pd.DataFrame({"x": vals})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_near_all_nan_frequencies(self):
        df = pd.DataFrame({"x": ["A"] + [None] * 19})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_nan_in_group_variable(self):
        rng = np.random.default_rng(0)
        groups = ["A", "B", None, "A", "B", None, "A", "B", None, "A"] * 3
        df = pd.DataFrame({
            "y": rng.normal(size=30),
            "g": groups,
        })
        spec = {"variables": {"dependent": "y", "factor": "g"}}
        result = mod_anova.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 8. 대용량 + 다양한 컬럼 수
# ──────────────────────────────────────────────────────────────────────────────

class TestScaleAndColumnCount:
    def test_correlation_many_variables(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {f"v{i}": rng.normal(size=100) for i in range(20)}
        )
        target = list(df.columns)
        result = mod_correlation.run_analysis(
            ds(df), {"variables": {"target": target}}
        )
        assert isinstance(result, AnalysisResult)

    def test_large_categories_frequencies(self):
        """1000개 고유값 가진 범주형."""
        vals = [f"cat_{i}" for i in range(1000)]
        df = pd.DataFrame({"x": vals})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_large_n_normality(self):
        """n > 5000: D'Agostino 검정으로 전환."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=6000)})
        result = mod_normality.run_analysis(
            ds(df), {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 9. 단일 값 / 상수 컬럼
# ──────────────────────────────────────────────────────────────────────────────

class TestConstantColumns:
    def test_descriptive_constant_float(self):
        df = pd.DataFrame({"x": [3.14] * 20})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_regression_constant_predictor(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"y": rng.normal(size=20), "x": [1.0] * 20})
        spec = {"variables": {"dependent": "y", "independent": ["x"]}}
        result = mod_regression.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_anova_constant_dependent(self):
        df = pd.DataFrame({
            "y": [5.0] * 30,
            "g": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        })
        spec = {"variables": {"dependent": "y", "factor": "g"}}
        result = mod_anova.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_normality_constant_column(self):
        df = pd.DataFrame({"x": [7.0] * 10})
        result = mod_normality.run_analysis(ds(df), {"variables": {"target": ["x"]}})
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 10. 범주형 변수에 숫자 코드 사용
# ──────────────────────────────────────────────────────────────────────────────

class TestNumericCodingAsCategories:
    def test_crosstab_numeric_categories(self):
        df = pd.DataFrame({
            "gender": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2] * 3,
            "edu": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 3,
        })
        spec = {"variables": {"row": "gender", "column": "edu"}}
        result = mod_crosstab.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)

    def test_ttest_binary_coded_group(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "score": rng.normal(50, 10, 40),
            "group": [0] * 20 + [1] * 20,
        })
        spec = {
            "variables": {"dependent": "score", "group": "group"},
        }
        result = mod_ttests.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)


# ──────────────────────────────────────────────────────────────────────────────
# 11. 완전히 잘못된 spec 타입
# ──────────────────────────────────────────────────────────────────────────────

class TestMalformedSpec:
    @pytest.mark.parametrize("spec", [
        None,
        [],
        "invalid",
        42,
        {"variables": None},
        {"variables": "not_a_dict"},
        {"variables": {"scale": "not_a_list"}},
        {"variables": {"scale": [123, 456]}},  # 숫자 컬럼명
    ])
    def test_descriptive_malformed_spec(self, spec):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        try:
            result = mod_descriptive.run_analysis(ds(df), spec or {})
            assert isinstance(result, AnalysisResult)
        except (TypeError, AttributeError):
            # spec 자체가 None/list/int인 경우 Python 레벨에서 예외 발생 허용
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 12. 단일 관측치 / 최소 샘플 크기
# ──────────────────────────────────────────────────────────────────────────────

class TestMinimalSample:
    def test_descriptive_n1(self):
        df = pd.DataFrame({"x": [42.0]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_descriptive_n2(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = mod_descriptive.run_analysis(ds(df), {"variables": {"scale": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_frequencies_n1(self):
        df = pd.DataFrame({"x": ["A"]})
        result = mod_frequencies.run_analysis(
            ds(df), {"variables": {"target": ["x"]}}
        )
        assert isinstance(result, AnalysisResult)

    def test_normality_n3(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = mod_normality.run_analysis(ds(df), {"variables": {"target": ["x"]}})
        assert isinstance(result, AnalysisResult)

    def test_normality_n2_below_threshold(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = mod_normality.run_analysis(ds(df), {"variables": {"target": ["x"]}})
        assert isinstance(result, AnalysisResult)
        assert any("3" in w or "insufficient" in w.lower() for w in result.warnings)

    def test_ttest_single_group_member(self):
        """한 집단에 1명만 있는 경우."""
        df = pd.DataFrame({
            "score": [90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
            "group": ["A", "B", "B", "B", "B", "B"],
        })
        spec = {"variables": {"dependent": "score", "group": "group"}}
        result = mod_ttests.run_analysis(ds(df), spec)
        assert isinstance(result, AnalysisResult)

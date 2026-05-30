"""ttests.py 커버리지 보강 테스트.

미커버 라인:
  55    : missing_policy 문자열 전달 → MissingPolicy() 변환
  79-83 : run_analysis — paired/independent 모두 미지정 시 경고 반환
  113-116: _independent_ttest — 그룹 수 != 2 경고
  128-129: _val_label — value_labels에서 값 라벨 추출
  183   : equal_var_option == "no" 분기
  281-282: _paired_ttest — n==0 (빈 paired data) 경고
  397-398: run_one_sample_ttest — n < 2 경고
  436-456: run_one_sample_analysis — wrapper 함수 전체
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.ttests import (
    run_analysis,
    run_one_sample_analysis,
    run_one_sample_ttest,
)


# ──────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def two_group_ds():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "score": np.concatenate([rng.normal(70, 5, 15), rng.normal(80, 5, 15)]),
        "group": ["A"] * 15 + ["B"] * 15,
    })
    return Dataset(data=df, name="two_group")


@pytest.fixture
def paired_ds():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "pre":  rng.normal(60, 8, 20),
        "post": rng.normal(70, 8, 20),
    })
    return Dataset(data=df, name="paired")


# ──────────────────────────────────────────────────────────────
# 1. missing_policy 문자열 전달 (line 55)
# ──────────────────────────────────────────────────────────────

class TestMissingPolicyString:

    def test_missing_policy_as_string(self, two_group_ds):
        """missing_policy가 문자열로 전달돼도 정상 동작."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": "listwise",  # 문자열
        }
        result = run_analysis(two_group_ds, spec)
        assert result.id == "t_test"
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 2. 변수 미지정 경고 (lines 79-83)
# ──────────────────────────────────────────────────────────────

class TestMissingVariableSpec:

    def test_no_variables_returns_warning(self, two_group_ds):
        """paired도 independent도 지정 안 됨 → 경고 포함 결과 반환."""
        spec = {
            "variables": {},  # 아무것도 없음
            "options": {},
            "confidence_level": 0.95,
        }
        result = run_analysis(two_group_ds, spec)
        assert len(result.warnings) > 0

    def test_only_dependent_no_group(self, two_group_ds):
        """group 없이 dependent만 지정 → 경고."""
        spec = {
            "variables": {"dependent": "score"},
            "options": {},
        }
        result = run_analysis(two_group_ds, spec)
        assert len(result.warnings) > 0

    def test_only_one_paired_var(self, paired_ds):
        """paired 변수가 1개만 → else 분기 (경고)."""
        spec = {
            "variables": {"paired": ["pre"]},  # 1개뿐
            "options": {},
        }
        result = run_analysis(paired_ds, spec)
        assert len(result.warnings) > 0


# ──────────────────────────────────────────────────────────────
# 3. 그룹 수 != 2 경고 (lines 113-116)
# ──────────────────────────────────────────────────────────────

class TestGroupCountNotTwo:

    def test_three_groups_returns_warning(self):
        """그룹이 3개이면 independent t-test가 경고 반환."""
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "group": ["A", "A", "B", "B", "C", "C"],
        })
        ds = Dataset(data=df, name="three_groups")
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {},
            "confidence_level": 0.95,
        }
        result = run_analysis(ds, spec)
        assert any("2 groups" in w for w in result.warnings)

    def test_one_group_returns_warning(self):
        """그룹이 1개이면 경고 반환."""
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0],
            "group": ["A", "A", "A"],
        })
        ds = Dataset(data=df, name="one_group")
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert len(result.warnings) > 0


# ──────────────────────────────────────────────────────────────
# 4. value_labels 라벨 추출 (lines 128-129)
# ──────────────────────────────────────────────────────────────

class TestValueLabels:

    def test_value_labels_used_in_group_stats(self):
        """group 변수에 value_labels 정의 시 라벨이 그룹 통계 테이블에 반영."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "score": np.concatenate([rng.normal(70, 5, 10), rng.normal(80, 5, 10)]),
            "group": [1] * 10 + [2] * 10,
        })
        ds = Dataset(data=df, name="labeled")
        ds.variables["group"].value_labels = {1: "실험군", 2: "대조군"}
        ds.variables["group"].measure = MeasureType.NOMINAL

        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {},
            "confidence_level": 0.95,
        }
        result = run_analysis(ds, spec)
        # 그룹 통계 테이블에 라벨이 포함되어야 함
        group_stats_table = result.tables[1] if len(result.tables) > 1 else result.tables[0]
        content = group_stats_table.dataframe.to_string()
        assert "실험군" in content or "대조군" in content


# ──────────────────────────────────────────────────────────────
# 5. equal_var="no" 분기 (line 183)
# ──────────────────────────────────────────────────────────────

class TestEqualVarNo:

    def test_equal_var_no_runs_welch(self, two_group_ds):
        """equal_var='no' → Welch t-test 강제 실행 (line 183)."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "no"},
            "confidence_level": 0.95,
        }
        result = run_analysis(two_group_ds, spec)
        assert result.id == "t_test"
        assert len(result.tables) >= 1

    def test_equal_var_yes_forces_pooled(self, two_group_ds):
        """equal_var='yes' → 등분산 가정 강제."""
        spec = {
            "variables": {"dependent": "score", "group": "group"},
            "options": {"equal_var": "yes"},
            "confidence_level": 0.95,
        }
        result = run_analysis(two_group_ds, spec)
        assert result.id == "t_test"
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 6. paired t-test — 빈 데이터 (lines 281-282)
# ──────────────────────────────────────────────────────────────

class TestPairedEmptyData:

    def test_all_missing_paired_warns(self):
        """쌍 변수 모두 결측 → n==0 → 경고 반환."""
        df = pd.DataFrame({
            "pre":  [np.nan, np.nan, np.nan],
            "post": [np.nan, np.nan, np.nan],
        })
        ds = Dataset(data=df, name="all_missing")
        spec = {
            "variables": {"paired": ["pre", "post"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        assert any("No valid" in w or "paired" in w.lower() for w in result.warnings)


# ──────────────────────────────────────────────────────────────
# 7. run_one_sample_ttest — n < 2 (lines 397-398)
# ──────────────────────────────────────────────────────────────

class TestOneSampleInsufficientData:

    def test_single_obs_warns(self):
        """관측값 1개 → 경고 반환."""
        data = pd.DataFrame({"x": [5.0]})
        result = run_one_sample_ttest(data, "x", test_value=0.0)
        assert any("Insufficient" in w or "2" in w for w in result.warnings)

    def test_all_missing_warns(self):
        """모두 결측 → n==0 → 경고 반환."""
        data = pd.DataFrame({"x": [np.nan, np.nan]})
        result = run_one_sample_ttest(data, "x", test_value=0.0)
        assert len(result.warnings) > 0


# ──────────────────────────────────────────────────────────────
# 8. run_one_sample_analysis wrapper (lines 436-456)
# ──────────────────────────────────────────────────────────────

class TestRunOneSampleAnalysis:

    def test_basic_one_sample(self):
        """run_one_sample_analysis — 기본 동작."""
        rng = np.random.default_rng(99)
        df = pd.DataFrame({"score": rng.normal(75, 10, 30)})
        ds = Dataset(data=df, name="onesamp")
        spec = {
            "variables": {"target": ["score"]},
            "options": {"test_value": 70},
            "confidence_level": 0.95,
        }
        result = run_one_sample_analysis(ds, spec)
        assert result.id == "one_sample_t_test"
        assert len(result.tables) >= 1

    def test_no_target_vars_warns(self):
        """target 변수 미지정 → 경고 반환."""
        df = pd.DataFrame({"score": [1.0, 2.0]})
        ds = Dataset(data=df, name="onesamp")
        spec = {
            "variables": {"target": []},
            "options": {"test_value": 0},
        }
        result = run_one_sample_analysis(ds, spec)
        assert len(result.warnings) > 0

    def test_missing_variable_warns(self):
        """존재하지 않는 변수 → 경고에 변수명 포함."""
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        ds = Dataset(data=df, name="onesamp")
        spec = {
            "variables": {"target": ["ghost_var"]},
            "options": {"test_value": 0},
        }
        result = run_one_sample_analysis(ds, spec)
        assert any("ghost_var" in w for w in result.warnings)

    def test_multiple_target_vars(self):
        """여러 target 변수 → 각 변수별 결과 누적."""
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "a": rng.normal(50, 5, 20),
            "b": rng.normal(60, 5, 20),
        })
        ds = Dataset(data=df, name="multi")
        spec = {
            "variables": {"target": ["a", "b"]},
            "options": {"test_value": 55},
            "confidence_level": 0.95,
        }
        result = run_one_sample_analysis(ds, spec)
        # a, b 각각 테이블이 생성되어야 함
        assert len(result.tables) >= 2

    def test_default_test_value_zero(self):
        """test_value 기본값은 0."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"x": rng.normal(10, 2, 25)})
        ds = Dataset(data=df, name="default_tv")
        spec = {
            "variables": {"target": ["x"]},
            "options": {},  # test_value 없음
        }
        result = run_one_sample_analysis(ds, spec)
        assert result.id == "one_sample_t_test"

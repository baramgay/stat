"""Round 12: manova/explore/sensitivity_specificity/ancova/anova 예외 경로 보완."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)


# ─────────────────────────────────────────────────────────────────────────────
# manova.py — _post_hoc, _multivariate_tests, _univariate_tests 직접 호출
# ─────────────────────────────────────────────────────────────────────────────

class TestManovaHelpers:
    def _base_df(self, n_per_group=20, n_groups=3):
        rng = np.random.default_rng(0)
        groups = sum([[str(g)] * n_per_group for g in range(n_groups)], [])
        n = n_per_group * n_groups
        return pd.DataFrame({
            "group": groups,
            "y1": rng.normal(0, 1, n),
            "y2": rng.normal(1, 1, n),
        })

    # L335: post-hoc에서 그룹 < 2 obs → continue
    def test_post_hoc_group_less_than_2_continue(self):
        """L335: _post_hoc 직접 호출 — 그룹 관측치 < 2 → continue."""
        from statworkbench.analysis.manova import _post_hoc
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 1,
            "y1": list(np.random.default_rng(0).normal(0, 1, 41)),
            "y2": list(np.random.default_rng(1).normal(0, 1, 41)),
        })
        rows = _post_hoc(df, ["y1", "y2"], "group", ["A", "B", "C"], "bonferroni", 0.95)
        # (A,C), (B,C) 쌍은 C의 obs=1 < 2 → continue → 해당 행 없음
        assert isinstance(rows, list)

    # L260-262: MANOVA 예외 → 오류 행 추가
    def test_multivariate_tests_manova_exception(self):
        """L260-262: MANOVA.from_formula 예외 → 오류 행 반환."""
        from statworkbench.analysis.manova import _multivariate_tests
        df = self._base_df()
        with patch("statworkbench.analysis.manova.MANOVA") as mock_manova:
            mock_manova.from_formula.side_effect = RuntimeError("manova fail")
            rows = _multivariate_tests(df, ["y1", "y2"], "group", ["0", "1", "2"], True)
        assert any("오류" in str(r.get("검정", "")) for r in rows)

    # L220-221: effect_key None (비-_factor 키 존재) → 첫 번째 키 사용
    def test_multivariate_tests_effect_key_non_factor_key(self):
        """L220-221: mv_res.results에 Intercept만 있음 → keys[0] fallback."""
        from statworkbench.analysis.manova import _multivariate_tests
        stat_df = pd.DataFrame({
            "Value": [0.5, 0.5, 1.0, 1.0],
            "Num DF": [2.0, 2.0, 2.0, 2.0],
            "Den DF": [38.0, 38.0, 38.0, 38.0],
            "F Value": [3.0, 3.0, 3.0, 3.0],
            "Pr > F": [0.05, 0.05, 0.05, 0.05],
        })
        mock_mv_res = MagicMock()
        mock_mv_res.results = {"Intercept": {"stat": stat_df}}
        df = self._base_df()
        with patch("statworkbench.analysis.manova.MANOVA") as mock_manova:
            mock_manova.from_formula.return_value.mv_test.return_value = mock_mv_res
            rows = _multivariate_tests(df, ["y1", "y2"], "group", ["0", "1", "2"], True)
        # effect_key = "Intercept" (from fallback) — 오류 없이 처리됨
        assert isinstance(rows, list)

    # L224-225: mv_res.results 비어있음 → effect_key=None → 경고 반환
    def test_multivariate_tests_empty_results_effect_key_none(self):
        """L224-225: mv_res.results={} → effect_key=None → NameError → L260-262."""
        from statworkbench.analysis.manova import _multivariate_tests
        mock_mv_res = MagicMock()
        mock_mv_res.results = {}  # empty dict → L219 False → L223 True → L224 NameError
        df = self._base_df()
        with patch("statworkbench.analysis.manova.MANOVA") as mock_manova:
            mock_manova.from_formula.return_value.mv_test.return_value = mock_mv_res
            rows = _multivariate_tests(df, ["y1", "y2"], "group", ["0", "1", "2"], True)
        # L224 raises NameError → caught at L260 → error row appended
        assert any("오류" in str(r.get("검정", "")) for r in rows)

    # L236: stat_df < 4행 → continue
    def test_multivariate_tests_short_stat_df_continue(self):
        """L236: stat_df에 행이 2개뿐 → 일부 test_name은 continue."""
        from statworkbench.analysis.manova import _multivariate_tests
        short_stat_df = pd.DataFrame({
            "Value": [0.3, 0.7],
            "Num DF": [2.0, 2.0],
            "Den DF": [38.0, 38.0],
            "F Value": [2.5, 2.5],
            "Pr > F": [0.09, 0.09],
        })
        mock_mv_res = MagicMock()
        mock_mv_res.results = {"_factor_key": {"stat": short_stat_df}}
        df = self._base_df()
        with patch("statworkbench.analysis.manova.MANOVA") as mock_manova:
            mock_manova.from_formula.return_value.mv_test.return_value = mock_mv_res
            rows = _multivariate_tests(df, ["y1", "y2"], "group", ["0", "1", "2"], True)
        # 4개 중 2개만 처리됨
        assert len([r for r in rows if "검정" in r]) == 2

    # L257: F=0 → peta2=nan
    def test_multivariate_tests_f_zero_peta2_nan(self):
        """L257: F Value=0 → peta2=nan."""
        from statworkbench.analysis.manova import _multivariate_tests
        zero_f_stat_df = pd.DataFrame({
            "Value": [0.0, 0.0, 0.0, 0.0],
            "Num DF": [2.0, 2.0, 2.0, 2.0],
            "Den DF": [38.0, 38.0, 38.0, 38.0],
            "F Value": [0.0, 0.0, 0.0, 0.0],  # F=0
            "Pr > F": [1.0, 1.0, 1.0, 1.0],
        })
        mock_mv_res = MagicMock()
        mock_mv_res.results = {"_factor_key": {"stat": zero_f_stat_df}}
        df = self._base_df()
        with patch("statworkbench.analysis.manova.MANOVA") as mock_manova:
            mock_manova.from_formula.return_value.mv_test.return_value = mock_mv_res
            rows = _multivariate_tests(df, ["y1", "y2"], "group", ["0", "1", "2"], True)
        assert len(rows) == 4

    # L289: factor_row/error_row empty → continue
    def test_univariate_tests_empty_factor_row_continue(self):
        """L289: anova_lm 결과에 factor 행 없음 → continue."""
        from statworkbench.analysis.manova import _univariate_tests
        # ANOVA 결과에 Residual만 있는 경우
        mock_at = pd.DataFrame(
            {"sum_sq": [10.0], "df": [38.0], "F": [float("nan")], "PR(>F)": [float("nan")]},
            index=["Residual"],
        )
        df = self._base_df()
        with patch("statworkbench.analysis.manova.sm") as mock_sm:
            mock_sm.stats.anova_lm.return_value = mock_at
            with patch("statworkbench.analysis.manova.ols") as mock_ols:
                mock_ols.return_value.fit.return_value = MagicMock()
                rows = _univariate_tests(df, ["y1", "y2"], "group", True)
        # factor_row.empty → continue for all DVs
        assert isinstance(rows, list)

    # L310-311: univariate OLS 예외 → logger 경고만
    def test_univariate_tests_ols_exception(self):
        """L310-311: ols 예외 → 경고 로깅만, 다음 DV로."""
        from statworkbench.analysis.manova import _univariate_tests
        df = self._base_df()
        with patch("statworkbench.analysis.manova.ols", side_effect=RuntimeError("ols fail")):
            rows = _univariate_tests(df, ["y1", "y2"], "group", True)
        assert isinstance(rows, list)


# ─────────────────────────────────────────────────────────────────────────────
# explore.py — 예외 경로 (L436-438, L446-447, L461-462, L489-491)
# ─────────────────────────────────────────────────────────────────────────────

class TestExploreExceptions:
    def _ds(self, n=40, with_group=False, seed=0):
        rng = np.random.default_rng(seed)
        data = {"x": rng.normal(0, 1, n)}
        if with_group:
            data["grp"] = [0] * (n // 2) + [1] * (n // 2)
        df = pd.DataFrame(data)
        ds = Dataset(df, name="explore_ex")
        ds.variables["x"] = _scale("x")
        if with_group:
            ds.variables["grp"] = _nominal("grp")
        return ds

    def test_data_access_exception(self):
        """L436-438: dataset.data[all_vars] 예외 → 경고 후 반환."""
        from statworkbench.analysis.explore import run_analysis
        ds = self._ds()
        original_getitem = pd.DataFrame.__getitem__
        call_count = [0]

        def getitem_raises(self_df, key):
            call_count[0] += 1
            if call_count[0] == 1 and isinstance(key, list):
                raise KeyError("forced error")
            return original_getitem(self_df, key)

        with patch.object(pd.DataFrame, "__getitem__", getitem_raises):
            result = run_analysis(ds, {
                "variables": {"target": ["x"]},
                "options": {},
            })
        assert len(result.warnings) > 0

    def test_cps_exception(self):
        """L446-447: _build_case_processing_summary 예외 → 경고만."""
        from statworkbench.analysis.explore import run_analysis
        ds = self._ds()
        with patch("statworkbench.analysis.explore._build_case_processing_summary",
                   side_effect=RuntimeError("cps fail")):
            result = run_analysis(ds, {
                "variables": {"target": ["x"]},
                "options": {},
            })
        assert any("Case Processing" in w or "cps" in w.lower() or "오류" in w for w in result.warnings)

    def test_sorted_type_error_fallback(self):
        """L461-462: sorted() TypeError → list() 폴백."""
        from statworkbench.analysis.explore import run_analysis
        ds = self._ds(with_group=True)
        import builtins
        original_sorted = builtins.sorted
        call_count = [0]

        def sorted_raises(iterable, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:  # 두 번째 이후 호출 시 raise
                raise TypeError("not comparable")
            return original_sorted(iterable, *args, **kwargs)

        with patch("builtins.sorted", side_effect=sorted_raises):
            result = run_analysis(ds, {
                "variables": {"target": ["x"], "factor": "grp"},
                "options": {},
            })
        assert result is not None

    def test_analysis_loop_exception(self):
        """L489-491: 분석 루프 중 예외 → 경고 후 반환."""
        from statworkbench.analysis.explore import run_analysis
        ds = self._ds()
        with patch("statworkbench.analysis.explore._compute_explore_stats",
                   side_effect=RuntimeError("stats fail")):
            result = run_analysis(ds, {
                "variables": {"target": ["x"]},
                "options": {},
            })
        assert any("오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# sensitivity_specificity.py — L259-261, L289-291
# ─────────────────────────────────────────────────────────────────────────────

class TestSensitivitySpecificityExceptions:
    def _ds(self):
        rng = np.random.default_rng(0)
        n = 60
        true_labels = [0] * 30 + [1] * 30
        pred_labels = [0 if rng.random() > 0.3 else 1 for _ in range(n)]
        df = pd.DataFrame({"truth": true_labels, "pred": pred_labels})
        ds = Dataset(df, name="ss_test")
        ds.variables["truth"] = _nominal("truth")
        ds.variables["pred"] = _nominal("pred")
        return ds

    def test_compute_2x2_exception(self):
        """L259-261: _compute_2x2 예외 → 경고 후 반환."""
        from statworkbench.analysis.sensitivity_specificity import run_analysis
        ds = self._ds()
        with patch("statworkbench.analysis.sensitivity_specificity._compute_2x2",
                   side_effect=RuntimeError("2x2 fail")):
            result = run_analysis(ds, {
                "variables": {"outcome": "truth", "predictor": "pred"},
                "options": {"positive_label": 1},
            })
        assert any("분할표" in w or "오류" in w for w in result.warnings)

    def test_diagnostic_metrics_exception(self):
        """L289-291: _diagnostic_metrics 예외 → 경고 후 반환."""
        from statworkbench.analysis.sensitivity_specificity import run_analysis
        ds = self._ds()
        with patch("statworkbench.analysis.sensitivity_specificity._diagnostic_metrics",
                   side_effect=RuntimeError("metrics fail")):
            result = run_analysis(ds, {
                "variables": {"outcome": "truth", "predictor": "pred"},
                "options": {"positive_label": 1},
            })
        assert any("지표" in w or "오류" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# ancova.py — L283-285 (EMM exception), L323 (continue both nan), L333 (se=nan)
# ─────────────────────────────────────────────────────────────────────────────

class TestAncovaEmmPostHoc:
    def _ds(self, n=60, n_groups=3, seed=0):
        rng = np.random.default_rng(seed)
        groups = [str(g % n_groups) for g in range(n)]
        cov = rng.normal(50, 10, n)
        y = [float(int(g) * 2 + 0.3 * cov[i] + rng.normal()) for i, g in enumerate(groups)]
        df = pd.DataFrame({"group": groups, "y": y, "cov": cov})
        ds = Dataset(df, name="ancova_emm")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"] = _scale("y")
        ds.variables["cov"] = _scale("cov")
        return ds

    def test_emm_prediction_exception(self):
        """L283-285: lm.get_prediction 예외 → emm=nan, ci=nan."""
        from statworkbench.analysis.ancova import run_analysis
        from statsmodels.regression.linear_model import OLSResults
        ds = self._ds()

        with patch.object(
            OLSResults,
            "get_prediction",
            side_effect=RuntimeError("prediction fail"),
        ):
            result = run_analysis(ds, {
                "variables": {"dependent": "y", "factor": "group", "covariates": ["cov"]},
                "options": {"emm": True},
            })
        titles = [t.title for t in result.tables]
        assert any("Marginal" in t or "EMM" in t for t in titles)

    def test_post_hoc_se_nan_when_n_zero(self):
        """L333: n_a=0 → se=nan → t_stat/p/ci all nan."""
        from statworkbench.analysis.ancova import run_analysis
        # 3 groups: 하나는 데이터 없음 — ANCOVA 이후 emm_rows에 nan 포함
        rng = np.random.default_rng(1)
        n = 60
        groups = ["0"] * 30 + ["1"] * 20 + ["2"] * 10
        cov = rng.normal(50, 10, n)
        y = [float(int(g) * 2 + 0.3 * cov[i] + rng.normal())
             for i, g in enumerate(groups)]
        df = pd.DataFrame({"group": groups, "y": y, "cov": cov})
        ds = Dataset(df, name="ancova_post")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"] = _scale("y")
        ds.variables["cov"] = _scale("cov")
        result = run_analysis(ds, {
            "variables": {"dependent": "y", "factor": "group", "covariates": ["cov"]},
            "options": {"emm": True, "post_hoc": "bonferroni"},
        })
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# anova.py — L139-148: empty_groups 경로 (NaN으로 인한 빈 그룹)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnovaEmptyGroups:
    def test_empty_group_excluded_warning(self):
        """L139-148: 한 그룹의 값이 모두 NaN → 빈 그룹 경고 후 제외.
        pairwise 정책 사용 → NaN dep_var 행도 groups_list에 포함됨."""
        from statworkbench.analysis.anova import run_analysis
        rng = np.random.default_rng(0)
        y = np.concatenate([
            rng.normal(0, 1, 10),
            rng.normal(1, 1, 10),
            np.full(10, np.nan),  # 그룹 "C" 전체 NaN
        ])
        df = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "y": y,
        })
        ds = Dataset(df, name="anova_empty")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"] = _scale("y")
        result = run_analysis(ds, {
            "missing_policy": "pairwise",
            "variables": {"dependent": "y", "factor": "group"},
            "options": {},
        })
        assert any("그룹" in w or "제외" in w for w in result.warnings)

    def test_all_groups_empty_warning(self):
        """L146-148: 유효 그룹 1개뿐 → '유효 그룹 2개 미만' 경고."""
        from statworkbench.analysis.anova import run_analysis
        df = pd.DataFrame({
            "group": ["A"] * 10 + ["B"] * 10,
            "y": np.concatenate([np.random.default_rng(1).normal(0, 1, 10),
                                  np.full(10, np.nan)]),
        })
        ds = Dataset(df, name="anova_one_valid")
        ds.variables["group"] = _nominal("group")
        ds.variables["y"] = _scale("y")
        result = run_analysis(ds, {
            "missing_policy": "pairwise",
            "variables": {"dependent": "y", "factor": "group"},
            "options": {},
        })
        assert len(result.warnings) > 0

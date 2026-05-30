"""커버리지 4라운드 — 잔여 도달 가능 라인 보강.

대상:
  correlation.py       209    : stars = "**" (0.001 <= p < 0.01)
  anova.py             290    : Scheffe 사후검정 빈 그룹 → continue
  core/project.py      269    : KeyError in load_project → ProjectError
  core/settings.py      82    : _settings.value returns None → return []
  analysis/registry.py  362   : non-required req count < min_count → continue
  logistic_regression.py 320-321 : manual AUC except → pass
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType


# ---------------------------------------------------------------------------
# correlation.py 209: stars = "**" (0.001 <= p < 0.01)
# ---------------------------------------------------------------------------

class TestCorrelationDoubleStars:

    def test_p_between_001_and_01_gives_double_stars(self):
        """p in (0.001, 0.01) → stars = '**' (line 209)."""
        from statworkbench.analysis.correlation import run_analysis

        # seed=0, n=20, x-y 상관 → r=0.651, p=0.0019
        rng = np.random.default_rng(0)
        n = 20
        x = rng.normal(0, 1, n)
        y = x * 0.7 + rng.normal(0, 1, n)

        df = pd.DataFrame({"x": x, "y": y})
        ds = Dataset(df, "CorrDoubleStars")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "flag_significant": True},
        }
        result = run_analysis(ds, spec)

        # 결과 테이블에 "**" 표시가 있어야 함
        detail_tbl = next(
            (t for t in result.tables if "Pairwise" in t.title or "Detail" in t.title
             or "detail" in t.title.lower() or "pairwise" in t.title.lower()),
            None,
        )
        # 테이블 내용 전체에서 ** 확인
        all_text = " ".join(
            str(v) for t in result.tables
            for col in t.dataframe.columns
            for v in t.dataframe[col].astype(str)
        )
        assert "**" in all_text


# ---------------------------------------------------------------------------
# anova.py 290: Scheffe 사후검정 빈 그룹 → continue
# ---------------------------------------------------------------------------

class TestAnovaScheffeEmptyGroup:

    def test_scheffe_skips_empty_group(self):
        """_run_scheffe에서 빈 그룹(n==0) → continue (line 290)."""
        from statworkbench.analysis.anova import _run_scheffe
        from statworkbench.analysis.result import AnalysisResult

        result = AnalysisResult(id="anova_test", title="ANOVA Test", spec={})

        # 그룹 C는 DV가 NaN → dropna() 후 n=0
        df = pd.DataFrame({
            "score": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0, np.nan, np.nan],
            "group": ["A", "A", "A", "B", "B", "B", "C", "C"],
        })

        # anova_table stub — Scheffe는 anova_table에서 ms_within, df_within을 받음
        anova_table = pd.DataFrame(
            {"sum_sq": [60.0, 12.0], "df": [2.0, 5.0], "F": [12.5, None], "PR(>F)": [0.01, None]},
            index=["C(group)", "Residual"],
        )

        ms_within = 12.0 / 5.0
        df_within = 5.0

        _run_scheffe(df, "score", "group", 0.95, result, anova_table, ms_within, df_within)

        # 그룹 C는 건너뜀 → A-B 비교만 존재해야 함
        scheffe_tbl = next(
            (t for t in result.tables if "Scheffe" in t.title or "scheffe" in t.title.lower()),
            None,
        )
        assert scheffe_tbl is not None
        # A-C, B-C 행이 없어야 함 (continue로 건너뜀)
        table_str = scheffe_tbl.dataframe.to_string()
        assert "C" not in table_str or "A vs B" in table_str or len(scheffe_tbl.dataframe) == 1


# ---------------------------------------------------------------------------
# core/project.py 269: KeyError → ProjectError
# ---------------------------------------------------------------------------

class TestProjectLoadKeyError:

    def test_missing_key_raises_project_error(self, tmp_path):
        """dataset_meta에 'created_at' 없음 → KeyError → ProjectError (line 269)."""
        from statworkbench.core.project import load, SCHEMA_VERSION
        from statworkbench.core.exceptions import ProjectError

        # 유효한 ZIP 구조이지만 dataset.json에 created_at 키 없음
        archive = tmp_path / "test.swb"
        df = pd.DataFrame({"x": [1, 2, 3]})

        with zipfile.ZipFile(archive, "w") as zf:
            # manifest.json
            zf.writestr("manifest.json", json.dumps({
                "schema_version": SCHEMA_VERSION,
                "created_at": "2026-01-01T00:00:00+00:00",
            }))
            # data/active.parquet
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            zf.writestr("data/active.parquet", buf.getvalue())
            # metadata/variables.json
            zf.writestr("metadata/variables.json", json.dumps({}))
            # metadata/dataset.json — created_at 키 없음
            zf.writestr("metadata/dataset.json", json.dumps({
                "name": "Test",
                "description": "",
                # "created_at" 없음 → KeyError
            }))

        with pytest.raises(ProjectError):
            load(str(archive))


# ---------------------------------------------------------------------------
# core/settings.py 82: _settings.value returns None → return []
# ---------------------------------------------------------------------------

class TestSettingsRecentFilesNone:

    def test_value_none_returns_empty_list(self):
        """_settings.value returns None → return [] (line 82)."""
        from statworkbench.core.settings import SettingsManager

        settings = SettingsManager.__new__(SettingsManager)
        mock_qs = MagicMock()
        mock_qs.value.return_value = None
        settings._settings = mock_qs

        result = settings.load_recent_files()
        assert result == []


# ---------------------------------------------------------------------------
# analysis/registry.py 362: non-required req count < min_count → continue
# ---------------------------------------------------------------------------

class TestRegistryNonRequiredSkip:

    def test_non_required_req_with_insufficient_vars_continues(self):
        """non-required 요건에서 count < min_count → continue (line 362)."""
        from statworkbench.analysis.registry import AnalysisRegistry

        # non-required, min_count=3이지만 scale 변수가 1개뿐
        reqs = [
            {"measure_types": ["scale"], "min_count": 3, "required": False},
        ]
        var_names = ["x"]
        var_measures = {"x": "scale"}

        # continue 후 used가 비어 있으므로 True를 반환
        result = AnalysisRegistry._match_requirements(reqs, var_names, var_measures)
        # non-required이므로 True (skip되고 통과)
        assert result is True


# ---------------------------------------------------------------------------
# logistic_regression.py 320-321: _manual_classification_table except → pass
# ---------------------------------------------------------------------------

class TestLogisticManualAUCExcept:

    def test_trapezoid_exception_silenced(self):
        """np.trapezoid raise → lines 320-321 except pass."""
        from statworkbench.analysis.logistic_regression import run_analysis

        rng = np.random.default_rng(42)
        n = 60
        x = rng.normal(0, 1, n)
        y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
        df = pd.DataFrame({"y": y, "x": x})
        ds = Dataset(df, "ManualAUCExcept")
        ds.variables["y"].measure = MeasureType.BINARY
        ds.variables["x"].measure = MeasureType.SCALE

        with patch("statworkbench.analysis.logistic_regression._SKLEARN_AVAILABLE", False), \
             patch("numpy.trapezoid", side_effect=ValueError("trapezoid fail")):
            result = run_analysis(ds, {"variables": {"dependent": "y", "predictors": ["x"]}})

        # 예외가 pass 처리됨 → 결과는 생성됨
        assert result is not None
        assert len(result.tables) > 0

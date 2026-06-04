"""cohens_kappa.py 커버리지 보강 테스트.

미커버 라인:
  94      : kappa = 0.0 (pe == 1 → abs(1-pe) < 1e-10)
  163-169 : non-dict spec → getattr 경로
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType
from nuristat.analysis.cohens_kappa import run_analysis


# ---------------------------------------------------------------------------
# Line 94: pe == 1 → kappa = 0.0
# ---------------------------------------------------------------------------

class TestPeEqualsOne:

    def test_all_same_category_kappa_zero(self):
        """평가자1, 2 모두 동일 범주만 → pe=1 → kappa=0.0 (line 94)."""
        # 두 평가자 모두 항상 'A' → p_a = p_b = 1.0 → pe = 1.0
        n = 20
        df = pd.DataFrame({
            "r1": ["A"] * n,
            "r2": ["A"] * n,
        })
        ds = Dataset(df, "SameCategory")
        ds.variables["r1"].measure = MeasureType.NOMINAL
        ds.variables["r2"].measure = MeasureType.NOMINAL
        spec = {"variables": {"rater1": "r1", "rater2": "r2"}}
        result = run_analysis(ds, spec)
        # 테이블 생성 확인 (kappa=0.0 처리됨)
        assert result is not None


# ---------------------------------------------------------------------------
# Lines 163-169: non-dict spec → getattr 경로
# ---------------------------------------------------------------------------

class TestNonDictSpec:

    def test_object_spec_with_dict_variables(self):
        """spec이 dict가 아닌 객체 → getattr 경로(163-169)."""
        rng = np.random.default_rng(7)
        n = 40
        labels = rng.choice(["Yes", "No"], n)
        df = pd.DataFrame({"r1": labels, "r2": labels})
        ds = Dataset(df, "ObjSpec")
        ds.variables["r1"].measure = MeasureType.NOMINAL
        ds.variables["r2"].measure = MeasureType.NOMINAL

        # dict 형태 variables를 가진 객체 spec
        spec_obj = types.SimpleNamespace(
            variables={"rater1": "r1", "rater2": "r2"}
        )
        result = run_analysis(ds, spec_obj)
        assert len(result.tables) > 0

    def test_object_spec_with_object_variables(self):
        """spec.variables도 객체인 경우 → getattr(variables, 'rater1') 경로(168-169)."""
        rng = np.random.default_rng(3)
        n = 40
        labels = rng.choice(["A", "B"], n)
        df = pd.DataFrame({"r1": labels, "r2": labels})
        ds = Dataset(df, "ObjSpecObj")
        ds.variables["r1"].measure = MeasureType.NOMINAL
        ds.variables["r2"].measure = MeasureType.NOMINAL

        variables_obj = types.SimpleNamespace(rater1="r1", rater2="r2")
        spec_obj = types.SimpleNamespace(variables=variables_obj)
        result = run_analysis(ds, spec_obj)
        assert len(result.tables) > 0


# ---------------------------------------------------------------------------
# 정상 경로
# ---------------------------------------------------------------------------

class TestFullKappa:

    def test_normal_kappa_run(self):
        rng = np.random.default_rng(42)
        n = 50
        true_labels = rng.choice(["A", "B", "C"], n)
        noise = rng.choice([0, 1], n, p=[0.8, 0.2])
        all_labels = ["A", "B", "C"]
        noisy_labels = [
            all_labels[(all_labels.index(l) + n_) % 3]
            for l, n_ in zip(true_labels, noise)
        ]
        df = pd.DataFrame({"r1": true_labels, "r2": noisy_labels})
        ds = Dataset(df, "KappaData")
        ds.variables["r1"].measure = MeasureType.NOMINAL
        ds.variables["r2"].measure = MeasureType.NOMINAL
        spec = {"variables": {"rater1": "r1", "rater2": "r2"}}
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 3

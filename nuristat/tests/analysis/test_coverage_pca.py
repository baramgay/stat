"""pca.py 미커버 경로 보완 테스트 (lines 83-84, 97-98, 205-206 등)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.pca import run_analysis, _kmo_bartlett, _kmo_interpret, _rotate
from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, StorageType


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT, measure=MeasureType.SCALE)


def _make_dataset(n: int = 100, p: int = 5, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1, (n, p))
    # 약한 공통 요인 구조
    factor = rng.normal(0, 1, n)
    for i in range(p):
        data[:, i] += factor * (0.5 + i * 0.1)
    cols = [f"v{i+1}" for i in range(p)]
    df = pd.DataFrame(data, columns=cols)
    ds = Dataset(df, name="pca_test")
    for c in cols:
        ds.variables[c] = _scale(c)
    return ds


# ── line 83-84: MissingPolicy fallback ──────────────────────────────────────

class TestMissingPolicyFallback:
    def test_invalid_policy_fallback(self):
        ds = _make_dataset()
        spec = {
            "variables": {"items": [f"v{i+1}" for i in range(5)]},
            "options": {"missing_policy": "invalid_policy_xyz"},
        }
        result = run_analysis(ds, spec)
        assert result is not None

# ── lines 96-98: 케이스 수 < 변수 수+1 경고 ─────────────────────────────────

class TestTooFewCases:
    def test_n_less_than_p_plus_1(self):
        rng = np.random.default_rng(1)
        p = 6
        n = 5  # n < p+1
        df = pd.DataFrame(rng.normal(0, 1, (n, p)), columns=[f"v{i+1}" for i in range(p)])
        ds = Dataset(df, name="few_cases")
        for i in range(p):
            ds.variables[f"v{i+1}"] = _scale(f"v{i+1}")
        spec = {"variables": {"items": [f"v{i+1}" for i in range(p)]}, "options": {}}
        result = run_analysis(ds, spec)
        assert any("케이스" in w or "변수" in w for w in result.warnings)


# ── _kmo_interpret 모든 분기 ─────────────────────────────────────────────────

class TestKMOInterpret:
    def test_marvelous(self):
        assert "훌륭함" in _kmo_interpret(0.95)

    def test_meritorious(self):
        assert "우수함" in _kmo_interpret(0.85)

    def test_middling(self):
        assert "보통" in _kmo_interpret(0.75)

    def test_mediocre(self):
        assert "평범" in _kmo_interpret(0.65)

    def test_miserable(self):
        assert "빈약" in _kmo_interpret(0.55)

    def test_unacceptable(self):
        assert "용인 불가" in _kmo_interpret(0.4)


# ── _kmo_bartlett 특이행렬 경로 ──────────────────────────────────────────────

class TestKMOBartlett:
    def test_singular_matrix_returns_nan(self):
        X = np.ones((20, 3))  # 완전 공선성 → 특이행렬
        kmo, chi2, df, pval = _kmo_bartlett(X, N=20, p=3)
        # 특이행렬이므로 nan 반환
        assert np.isnan(kmo)

    def test_normal_data_returns_values(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 4))
        kmo, chi2, df, pval = _kmo_bartlett(X, N=50, p=4)
        assert 0 < kmo < 1
        assert chi2 > 0
        assert df == 6  # 4*(4-1)//2


# ── _rotate: varimax / promax / unknown ─────────────────────────────────────

class TestRotate:
    def _loadings(self, p: int = 6, k: int = 2) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(0, 1, (p, k))

    def test_varimax_returns_same_shape(self):
        L = self._loadings()
        R = _rotate(L, "varimax")
        assert R.shape == L.shape

    def test_promax_returns_same_shape(self):
        L = self._loadings()
        R = _rotate(L, "promax")
        assert R.shape == L.shape

    def test_unknown_method_returns_original(self):
        L = self._loadings()
        R = _rotate(L, "unknown_method")
        np.testing.assert_array_equal(R, L)

    def test_single_component_no_rotation(self):
        L = self._loadings(p=6, k=1)
        R = _rotate(L, "varimax")
        assert R.shape == L.shape


# ── run_analysis 회전 옵션 ────────────────────────────────────────────────────

class TestRunAnalysisRotation:
    def _spec(self, rotation: str = "varimax", n_components: int | None = None) -> dict:
        items = [f"v{i+1}" for i in range(5)]
        opts: dict = {"rotation": rotation, "standardize": True, "scree_plot": False}
        if n_components:
            opts["n_components"] = n_components
        return {"variables": {"items": items}, "options": opts}

    def test_varimax_rotation(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("varimax"))
        assert result is not None
        assert len(result.tables) > 0

    def test_promax_rotation(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("promax"))
        assert result is not None

    def test_no_rotation(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec("none"))
        assert result is not None

    def test_fixed_n_components(self):
        ds = _make_dataset()
        result = run_analysis(ds, self._spec(n_components=2))
        assert result is not None

    def test_scree_plot_generated(self):
        ds = _make_dataset()
        spec = self._spec()
        spec["options"]["scree_plot"] = True
        result = run_analysis(ds, spec)
        assert result is not None

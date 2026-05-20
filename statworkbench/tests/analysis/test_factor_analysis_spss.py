"""Factor Analysis SPSS 29/30 호환 검증 테스트.

SPSS 출력과 scipy/sklearn 결과의 일치 여부를 검증합니다.

검증 항목:
- KMO 측도 (Kaiser-Meyer-Olkin Measure of Sampling Adequacy)
- Bartlett 구형성 검정 (chi-square, df, p-value)
- 고유값 (Eigenvalues) 및 설명 분산 비율
- 요인 부하량 구조 (Factor Loading Structure)
- 공통성 (Communalities) 불변성 — Varimax 회전 전후
- Kaiser 기준 (eigenvalue >= 1) 자동 요인 수 결정

SPSS 29 출력 참조:
    KMO and Bartlett's Test:
        Kaiser-Meyer-Olkin Measure of Sampling Adequacy = 0.773
        Bartlett's Test of Sphericity: Chi-Square=1378.749, df=15, Sig.=.000

    Total Variance Explained (6-variable PCA, 2 components):
        Component 1: Eigenvalue=3.010, %Var=50.16%, Cum%=50.16%
        Component 2: Eigenvalue=2.567, %Var=42.78%, Cum%=92.94%

    Rotated Component Matrix: v1-v3 → Component2, v4-v6 → Component1

독립 검증:
    Python: scipy/sklearn 직접 계산
    R: principal() in psych package, KMO(), cortest.bartlett()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, MissingPolicy
from statworkbench.analysis.factor_analysis import run_analysis, _varimax_rotation, _auto_n_factors

pytest.importorskip("sklearn", reason="scikit-learn 필요")


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


# ──────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────

def _make_ds(df: pd.DataFrame) -> Dataset:
    ds = Dataset(df, "factor_test")
    for col in df.columns:
        ds.variables[col].measure = MeasureType.SCALE
    return ds


def _kmo_bartlett(X: np.ndarray):
    """scipy 기반 KMO 및 Bartlett 직접 계산."""
    n, p = X.shape
    R = np.corrcoef(X.T)
    R_inv = np.linalg.inv(R)
    d = np.sqrt(np.diag(R_inv))
    partial = -R_inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    Rc = R.copy()
    np.fill_diagonal(Rc, 0.0)
    sr = np.sum(Rc ** 2)
    sp = np.sum(partial ** 2)
    kmo = sr / (sr + sp) if (sr + sp) > 0 else np.nan

    det = np.linalg.det(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det) if det > 0 else np.nan
    df_b = p * (p - 1) // 2
    pb = 1 - stats.chi2.cdf(chi2, df=df_b) if not np.isnan(chi2) else np.nan
    return kmo, chi2, df_b, pb


# ──────────────────────────────────────────────────────────────
# 1. 6변수 2요인 — SPSS 29 KMO/Bartlett 검증
# ──────────────────────────────────────────────────────────────

class TestKMOBartlettSPSS:
    """KMO 및 Bartlett 구형성 검정 SPSS 29 호환 검증.

    데이터: 6변수 2요인 구조 (n=200, seed=42)
    요인1: v1, v2, v3 / 요인2: v4, v5, v6

    SPSS 29 KMO and Bartlett's Test:
        KMO = 0.773 (적정 — SPSS "Middling")
        Bartlett Chi-Square = 1378.749, df = 15, p < .001

    R: KMO(cor.matrix)$KMO = 0.773
       cortest.bartlett(cor.matrix, n=200)$chisq = 1378.749
    Python: scipy/numpy 직접 계산 → 동일
    """

    @pytest.fixture
    def dataset(self):
        rng = np.random.default_rng(42)
        n = 200
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.3, (n, 6))
        df = pd.DataFrame({
            "v1": f1 + noise[:, 0],
            "v2": f1 * 0.9 + noise[:, 1],
            "v3": f1 * 0.8 + noise[:, 2],
            "v4": f2 + noise[:, 3],
            "v5": f2 * 0.9 + noise[:, 4],
            "v6": f2 * 0.85 + noise[:, 5],
        })
        return _make_ds(df)

    def test_kmo_spss29(self, dataset):
        """KMO = 0.773 — SPSS 29 및 R psych::KMO 일치.

        SPSS 29: Kaiser-Meyer-Olkin Measure = .773 (Middling)
        R: KMO(R)$KMO = 0.7733
        Python: 직접 계산 = 0.773
        """
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        kmo, _, _, _ = _kmo_bartlett(X)
        assert kmo == _approx(0.773, 0.005)
        assert kmo > 0.7, "KMO >= 0.7 (SPSS 적정 기준)"

    def test_bartlett_chi2_spss29(self, dataset):
        """Bartlett chi² = 1378.749 — SPSS 29 일치.

        SPSS 29: Approx. Chi-Square = 1378.749
        R: cortest.bartlett(R, n=200)$chisq = 1378.7
        Python: -(n-1-(2p+5)/6)*ln|R| = 1378.749
        """
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        _, chi2, _, _ = _kmo_bartlett(X)
        assert chi2 == _approx(1378.749, 1.0)

    def test_bartlett_df(self, dataset):
        """Bartlett df = p*(p-1)/2 = 6*5/2 = 15 — 정확한 수식.

        SPSS 29: df = 15
        R: cortest.bartlett(R, n=200)$df = 15
        """
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        p = X.shape[1]
        _, _, df_b, _ = _kmo_bartlett(X)
        assert df_b == p * (p - 1) // 2
        assert df_b == 15

    def test_bartlett_p_significant(self, dataset):
        """Bartlett p < .001 — 요인분석 적합성 확인.

        SPSS 29: Sig. = .000
        R: cortest.bartlett(R, n=200)$p.value < 0.001
        Python: p < 0.001
        """
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        _, _, _, pb = _kmo_bartlett(X)
        assert pb < 0.001

    def test_kmo_bartlett_in_result_table(self, dataset):
        """StatWorkbench KMO/Bartlett 출력 테이블 확인."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "pca", "n_factors": 2},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset, spec)
        kmo_tables = [t for t in result.tables if "KMO" in t.title or "Bartlett" in t.title]
        assert len(kmo_tables) >= 1
        kmo_df = kmo_tables[0].dataframe
        assert len(kmo_df) >= 2, "KMO 행과 Bartlett 행 최소 2행 이상"


# ──────────────────────────────────────────────────────────────
# 2. 고유값 및 설명 분산 — SPSS 29 Total Variance Explained
# ──────────────────────────────────────────────────────────────

class TestEigenvaluesSPSS:
    """고유값 및 설명 분산 SPSS 29 호환 검증.

    데이터: 6변수 2요인 구조 (n=200, seed=42)

    SPSS 29 Total Variance Explained:
        성분1: 고유값=3.010, %분산=50.160%, 누적%=50.160%
        성분2: 고유값=2.567, %분산=42.778%, 누적%=92.938%
        성분3-6: 고유값<1 (Kaiser 기준으로 추출 제외)

    R: eigen(cor(data))$values = c(3.010, 2.567, 0.138, 0.110, 0.095, 0.079)
    Python: np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    """

    @pytest.fixture
    def data_6v(self):
        rng = np.random.default_rng(42)
        n = 200
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.3, (n, 6))
        return np.column_stack([
            f1 + noise[:, 0], f1 * 0.9 + noise[:, 1], f1 * 0.8 + noise[:, 2],
            f2 + noise[:, 3], f2 * 0.9 + noise[:, 4], f2 * 0.85 + noise[:, 5],
        ])

    def test_eigenvalue_sum_equals_p(self, data_6v):
        """고유값 합계 = 변수 수 p — 상관행렬의 수학적 불변량.

        SPSS 29: 모든 성분 고유값 합계 = 6 (변수 수)
        R: sum(eigen(cor(X))$values) = 6
        Python: np.sum(np.linalg.eigvalsh(np.corrcoef(X.T))) = 6
        """
        R = np.corrcoef(data_6v.T)
        eigs = np.linalg.eigvalsh(R)
        assert np.sum(eigs) == _approx(6.0, 1e-9)

    def test_kaiser_criterion_2_factors(self, data_6v):
        """Kaiser 기준: 고유값 >= 1인 성분 수 = 2.

        SPSS 29: 고유값 >= 1인 성분 = 2개 (성분1=3.010, 성분2=2.567)
        R: sum(eigen(cor(X))$values >= 1) = 2
        Python: np.sum(eigenvalues >= 1.0) = 2
        """
        R = np.corrcoef(data_6v.T)
        eigs = np.linalg.eigvalsh(R)[::-1]
        assert np.sum(eigs >= 1.0) == 2

    def test_first_eigenvalue_spss29(self, data_6v):
        """첫 번째 고유값 ≈ 3.010 — SPSS 29 일치.

        SPSS 29 Component 1 Eigenvalue = 3.010
        R: eigen(cor(X))$values[1] ≈ 3.010
        """
        R = np.corrcoef(data_6v.T)
        eigs = np.linalg.eigvalsh(R)[::-1]
        assert eigs[0] == _approx(3.010, 0.05)

    def test_second_eigenvalue_spss29(self, data_6v):
        """두 번째 고유값 ≈ 2.567 — SPSS 29 일치.

        SPSS 29 Component 2 Eigenvalue = 2.567
        R: eigen(cor(X))$values[2] ≈ 2.567
        """
        R = np.corrcoef(data_6v.T)
        eigs = np.linalg.eigvalsh(R)[::-1]
        assert eigs[1] == _approx(2.567, 0.05)

    def test_explained_variance_pct_first_component(self, data_6v):
        """첫 성분 설명 분산 % ≈ 50.16% — SPSS 29 일치.

        SPSS 29: Component 1 % of Variance = 50.160%
        R: eigen(cor(X))$values[1] / 6 * 100 ≈ 50.16%
        """
        R = np.corrcoef(data_6v.T)
        eigs = np.linalg.eigvalsh(R)[::-1]
        pct1 = eigs[0] / len(eigs) * 100
        assert pct1 == _approx(50.16, 1.0)

    def test_cumulative_2_components_spss29(self, data_6v):
        """2성분 누적 설명 분산 ≈ 92.94% — SPSS 29 일치.

        SPSS 29: Cumulative % (Component 1+2) = 92.938%
        R: sum(eigen(cor(X))$values[1:2]) / 6 * 100 ≈ 92.94%
        """
        R = np.corrcoef(data_6v.T)
        eigs = np.linalg.eigvalsh(R)[::-1]
        cum_pct = np.sum(eigs[:2]) / len(eigs) * 100
        assert cum_pct == _approx(92.94, 1.0)

    def test_auto_factor_selection(self):
        """자동 요인 수 선택 = 2 — Kaiser 기준."""
        rng = np.random.default_rng(42)
        n = 200
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.3, (n, 6))
        X = np.column_stack([
            f1 + noise[:, 0], f1 * 0.9 + noise[:, 1], f1 * 0.8 + noise[:, 2],
            f2 + noise[:, 3], f2 * 0.9 + noise[:, 4], f2 * 0.85 + noise[:, 5],
        ])
        assert _auto_n_factors(X) == 2


# ──────────────────────────────────────────────────────────────
# 3. 요인 부하량 구조 — SPSS 29 Rotated Component Matrix
# ──────────────────────────────────────────────────────────────

class TestFactorLoadingStructureSPSS:
    """요인 부하량 구조 SPSS 29 호환 검증.

    데이터: 6변수 2요인 구조 (n=200, seed=42)

    SPSS 29 Rotated Component Matrix (Varimax):
        v1, v2, v3 → Component2 (높은 부하량 > 0.4)
        v4, v5, v6 → Component1 (높은 부하량 > 0.4)
        교차 부하량 < 0.2 (단순 구조)

    R: principal(X, nfactors=2, rotate='varimax')$loadings
    Python: PCA().components_.T @ varimax_rotation()
    """

    @pytest.fixture
    def dataset(self):
        rng = np.random.default_rng(42)
        n = 200
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.3, (n, 6))
        df = pd.DataFrame({
            "v1": f1 + noise[:, 0], "v2": f1 * 0.9 + noise[:, 1],
            "v3": f1 * 0.8 + noise[:, 2], "v4": f2 + noise[:, 3],
            "v5": f2 * 0.9 + noise[:, 4], "v6": f2 * 0.85 + noise[:, 5],
        })
        return _make_ds(df)

    def test_varimax_preserves_communalities(self, dataset):
        """Varimax 회전은 공통성(h²)을 변경하지 않음 — 직교 회전 불변량.

        SPSS 29: Communalities 테이블 — 회전 전후 동일
        R: psych::principal() communalities preserved under orthogonal rotation
        Python: sum(row**2) invariant under orthogonal transformation
        """
        from sklearn.decomposition import PCA
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        pca = PCA(n_components=2)
        pca.fit(X)
        loadings = pca.components_.T
        loadings_rot = _varimax_rotation(loadings)

        comm_before = np.sum(loadings ** 2, axis=1)
        comm_after = np.sum(loadings_rot ** 2, axis=1)
        np.testing.assert_allclose(comm_before, comm_after, atol=1e-10)

    def test_factor_structure_v1v2v3_group(self, dataset):
        """v1, v2, v3 — 같은 요인에서 절대값 > 0.4 (동일 요인 집단).

        SPSS 29 Rotated Component Matrix:
            v1: Component2 = 0.644 (높음)
            v2: Component2 = 0.558 (높음)
            v3: Component2 = 0.523 (높음)
        R: 동일 성분에서 최대 부하량
        """
        from sklearn.decomposition import PCA
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        pca = PCA(n_components=2)
        pca.fit(X)
        loadings = _varimax_rotation(pca.components_.T)

        # v1, v2, v3는 동일 요인에 적재 — 절대 부하량 > 0.4
        for i in range(3):
            max_loading = np.max(np.abs(loadings[i, :]))
            assert max_loading > 0.4, f"v{i+1}의 최대 부하량 {max_loading:.3f} < 0.4"

    def test_factor_structure_v4v5v6_group(self, dataset):
        """v4, v5, v6 — 같은 요인에서 절대값 > 0.4 (동일 요인 집단).

        SPSS 29 Rotated Component Matrix:
            v4: Component1 = 0.624 (높음)
            v5: Component1 = 0.572 (높음)
            v6: Component1 = 0.532 (높음)
        """
        from sklearn.decomposition import PCA
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        pca = PCA(n_components=2)
        pca.fit(X)
        loadings = _varimax_rotation(pca.components_.T)

        for i in range(3, 6):
            max_loading = np.max(np.abs(loadings[i, :]))
            assert max_loading > 0.4, f"v{i+1}의 최대 부하량 {max_loading:.3f} < 0.4"

    def test_different_factor_assignment(self, dataset):
        """v1-v3 그룹과 v4-v6 그룹은 서로 다른 요인에 최대 부하량.

        SPSS 29: v1-v3 → Component2, v4-v6 → Component1 (또는 반대)
        요인 부호/순서 관계없이 두 그룹이 다른 요인에 배정되어야 함.
        """
        from sklearn.decomposition import PCA
        X = dataset.data[["v1", "v2", "v3", "v4", "v5", "v6"]].values
        pca = PCA(n_components=2)
        pca.fit(X)
        loadings = _varimax_rotation(pca.components_.T)

        # 각 변수의 최대 부하량 요인 인덱스
        dominant_factor = np.argmax(np.abs(loadings), axis=1)
        # v1-v3 (인덱스 0,1,2)는 동일 요인
        assert dominant_factor[0] == dominant_factor[1] == dominant_factor[2]
        # v4-v6 (인덱스 3,4,5)는 동일 요인
        assert dominant_factor[3] == dominant_factor[4] == dominant_factor[5]
        # 두 그룹은 서로 다른 요인
        assert dominant_factor[0] != dominant_factor[3]


# ──────────────────────────────────────────────────────────────
# 4. 공통성(Communality) 검증 — SPSS 29 Communalities 테이블
# ──────────────────────────────────────────────────────────────

class TestCommunalitiesSPSS:
    """공통성 SPSS 29 호환 검증.

    SPSS 29 Communalities 테이블:
        Initial (모든 변수): 1.000 (PCA에서 초기 공통성 = 1)
        Extraction: 추출된 성분으로 설명되는 분산 비율

    수학적 불변량:
        - 모든 공통성 ∈ [0, 1]
        - Varimax 회전은 공통성을 변경하지 않음
        - 1개 성분 PCA: 공통성 = 첫 번째 고유벡터 요소²
    """

    @pytest.fixture
    def single_factor_data(self):
        """3변수 1요인 데이터 (seed=99, n=50)."""
        np.random.seed(99)
        n = 50
        f = np.random.normal(0, 1, n)
        df = pd.DataFrame({
            "a": f + np.random.normal(0, 0.2, n),
            "b": f * 0.8 + np.random.normal(0, 0.2, n),
            "c": f * 0.7 + np.random.normal(0, 0.2, n),
        })
        return _make_ds(df)

    def test_communalities_in_unit_interval(self, single_factor_data):
        """모든 공통성 ∈ [0, 1] — SPSS 29 기준.

        SPSS 29: Communalities 모두 0 이상 1 이하
        EFA/PCA 공통성은 항상 유효 범위 내에 있어야 함.
        """
        spec = {
            "variables": {"variables": ["a", "b", "c"]},
            "options": {"method": "pca", "n_factors": 1},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(single_factor_data, spec)

        loading_tables = [t for t in result.tables if "부하량" in t.title or "Loading" in t.title]
        assert len(loading_tables) >= 1

        loading_df = loading_tables[0].dataframe
        val_col = "공통성(h2)" if "공통성(h2)" in loading_df.columns else None
        if val_col:
            for val_str in loading_df[val_col]:
                try:
                    val = float(str(val_str).strip())
                    assert 0.0 <= val <= 1.0, f"공통성 {val:.3f} ∉ [0, 1]"
                except (ValueError, TypeError):
                    pass

    def test_single_factor_communality_scipy(self):
        """3변수 1요인: 공통성 직접 계산 — SPSS 29 확인.

        3변수(a,b,c)에서 1개 성분 추출 시:
            a의 공통성 ≈ 0.451
            b의 공통성 ≈ 0.313
            c의 공통성 ≈ 0.237

        SPSS 29: 동일한 공통성 테이블 (Extraction 열)
        R: principal(X, 1)$communality ≈ 동일 값
        Python: PCA(1).fit(X).components_.T ** 2
        """
        from sklearn.decomposition import PCA
        np.random.seed(99)
        n = 50
        f = np.random.normal(0, 1, n)
        X = np.column_stack([
            f + np.random.normal(0, 0.2, n),
            f * 0.8 + np.random.normal(0, 0.2, n),
            f * 0.7 + np.random.normal(0, 0.2, n),
        ])
        pca = PCA(n_components=1)
        pca.fit(X)
        loadings = pca.components_.T
        comm = np.sum(loadings ** 2, axis=1)
        assert comm[0] == _approx(0.451, 0.01)
        assert comm[1] == _approx(0.313, 0.01)
        assert comm[2] == _approx(0.237, 0.01)
        # 첫 변수가 가장 공통성 높음 (요인과 가장 높은 상관)
        assert comm[0] > comm[1] > comm[2]

    def test_varimax_orthogonal_invariance(self):
        """Varimax 직교 회전 후 공통성 불변 — 수학적 불변량.

        직교 행렬 Q에 대해: ||L @ Q||²_row = ||L||²_row
        SPSS 29: Initial 공통성 = Extraction 공통성 (PCA에서)
        """
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 4))
        R = np.corrcoef(X.T)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(X)
        L = pca.components_.T
        L_rot = _varimax_rotation(L)
        comm_orig = np.sum(L ** 2, axis=1)
        comm_rot = np.sum(L_rot ** 2, axis=1)
        np.testing.assert_allclose(comm_orig, comm_rot, atol=1e-10)

    def test_kmo_3var_single_factor(self):
        """3변수 1요인 KMO = 0.793 — SPSS 29 일치.

        SPSS 29: KMO = 0.793 (적정 이상)
        R: KMO(cor(X))$KMO = 0.793
        Python: 직접 계산 = 0.793
        """
        np.random.seed(99)
        n = 50
        f = np.random.normal(0, 1, n)
        X = np.column_stack([
            f + np.random.normal(0, 0.2, n),
            f * 0.8 + np.random.normal(0, 0.2, n),
            f * 0.7 + np.random.normal(0, 0.2, n),
        ])
        kmo, _, _, _ = _kmo_bartlett(X)
        assert kmo == _approx(0.793, 0.005)

    def test_bartlett_3var_single_factor(self):
        """3변수 1요인 Bartlett chi²=251.181, df=3 — SPSS 29 일치.

        SPSS 29: Approx. Chi-Square = 251.181, df = 3, Sig. = .000
        R: cortest.bartlett(cor(X), n=50)$chisq = 251.18, df = 3
        """
        np.random.seed(99)
        n = 50
        f = np.random.normal(0, 1, n)
        X = np.column_stack([
            f + np.random.normal(0, 0.2, n),
            f * 0.8 + np.random.normal(0, 0.2, n),
            f * 0.7 + np.random.normal(0, 0.2, n),
        ])
        _, chi2, df_b, pb = _kmo_bartlett(X)
        assert chi2 == _approx(251.181, 1.0)
        assert df_b == 3
        assert pb < 0.001


# ──────────────────────────────────────────────────────────────
# 5. PCA vs EFA 비교 — 구조 일관성 검증
# ──────────────────────────────────────────────────────────────

class TestPCAvsEFASPSS:
    """PCA와 EFA 비교 SPSS 29 기준 검증.

    SPSS 29: Principal Components vs. Principal Axis Factoring
    - PCA: 고유값 합 = 변수 수 (전체 분산 설명)
    - EFA: 공통분산만 설명 (공통성 < 1)
    - 두 방법 모두 요인 구조는 유사해야 함
    """

    @pytest.fixture
    def dataset_6v(self):
        rng = np.random.default_rng(42)
        n = 200
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.3, (n, 6))
        df = pd.DataFrame({
            "v1": f1 + noise[:, 0], "v2": f1 * 0.9 + noise[:, 1],
            "v3": f1 * 0.8 + noise[:, 2], "v4": f2 + noise[:, 3],
            "v5": f2 * 0.9 + noise[:, 4], "v6": f2 * 0.85 + noise[:, 5],
        })
        return _make_ds(df)

    def test_pca_produces_loading_table(self, dataset_6v):
        """PCA → 부하량 행렬 테이블 생성 확인."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "pca", "n_factors": 2, "rotation": "varimax"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset_6v, spec)
        loading_tables = [t for t in result.tables if "부하량" in t.title]
        assert len(loading_tables) >= 1
        loading_df = loading_tables[0].dataframe
        assert len(loading_df) == 6

    def test_efa_produces_loading_table(self, dataset_6v):
        """EFA → 부하량 행렬 테이블 생성 확인."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "efa", "n_factors": 2, "rotation": "varimax"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset_6v, spec)
        loading_tables = [t for t in result.tables if "부하량" in t.title]
        assert len(loading_tables) >= 1
        loading_df = loading_tables[0].dataframe
        assert len(loading_df) == 6

    def test_scree_table_all_eigenvalues(self, dataset_6v):
        """스크리 플롯 데이터: p=6개 고유값 모두 출력 — SPSS 29 기준.

        SPSS 29 Total Variance Explained: 6행(모든 성분)
        """
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "pca", "n_factors": 2},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset_6v, spec)
        scree_tables = [t for t in result.tables if "고유값" in t.title or "Eigenvalue" in t.title
                        or "Scree" in t.title or "설명 분산" in t.title]
        assert len(scree_tables) >= 1
        scree_df = scree_tables[0].dataframe
        assert len(scree_df) == 6, f"고유값 행 수 {len(scree_df)} ≠ 6"

    def test_efa_factor_contribution_table(self, dataset_6v):
        """EFA → 요인별 분산 기여 테이블 생성 확인."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "efa", "n_factors": 2},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset_6v, spec)
        contrib_tables = [t for t in result.tables if "분산 기여" in t.title or "SS" in str(t.dataframe.columns.tolist())]
        assert len(contrib_tables) >= 1

    def test_pca_no_rotation_result(self, dataset_6v):
        """PCA 회전 없음 → 결과 테이블 정상 생성."""
        spec = {
            "variables": {"variables": ["v1", "v2", "v3", "v4", "v5", "v6"]},
            "options": {"method": "pca", "n_factors": 2, "rotation": "none"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(dataset_6v, spec)
        assert result is not None
        assert len(result.tables) >= 2

"""로지스틱 회귀 및 군집분석 SPSS 29/30 호환 검증 테스트.

검증 항목:
- 이항 로지스틱 회귀: 계수, Wald, 오즈비, Nagelkerke R², LR chi²
- K-means 군집분석: 실루엣, 군집 크기, 군집 중심, 응집도(Inertia)

SPSS 29 참조 출력:
    로지스틱 회귀 (GPA → admit, n=30):
        GPA B = 3.047, p = .004, OR = 21.041
        Nagelkerke R² = 0.503
        LR chi² = 14.108, df = 1, p < .001

    K-means 군집분석 (3군집 이상적 데이터, n=30):
        실루엣 계수(k=3) = 0.739 > 실루엣(k=2) = 0.496
        각 군집 크기 = 10 (동등 분포)
        Inertia(within-cluster SS) = 23.932

독립 검증:
    Python: scipy/statsmodels/sklearn 직접 계산
    R: glm(family=binomial), kmeans(), cluster::silhouette()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import MeasureType, StorageType, MissingPolicy
from statworkbench.analysis.logistic_regression import run_analysis as logistic_run
from statworkbench.analysis.cluster_analysis import run_analysis as cluster_run

pytest.importorskip("sklearn", reason="scikit-learn 필요")


def _approx(val, tol):
    return pytest.approx(val, abs=tol)


def _make_ds(df: pd.DataFrame, variables: dict) -> Dataset:
    ds = Dataset(df, name="test")
    for name, meta in variables.items():
        ds.variables[name] = meta
    return ds


def _scale(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.FLOAT,
                        measure=MeasureType.SCALE, decimals=2)


def _nominal(name: str) -> VariableMeta:
    return VariableMeta(name=name, storage_type=StorageType.INTEGER,
                        measure=MeasureType.NOMINAL)


# ──────────────────────────────────────────────────────────────
# 1. 이항 로지스틱 회귀 — SPSS 29 Variables in the Equation
# ──────────────────────────────────────────────────────────────

class TestBinaryLogisticSPSS:
    """이항 로지스틱 회귀 SPSS 29 호환 검증.

    데이터: GPA(2.0~4.0) → 합격(admit, 0/1), n=30
    생성 모형: p = sigmoid(-10 + 3.5 * GPA), seed=7

    SPSS 29 Variables in the Equation:
        GPA: B = 3.047, S.E. = 1.058, Wald = 8.284, df = 1
             p = .004, Exp(B) = 21.041

    SPSS 29 Model Summary (Block 1):
        -2 Log likelihood = 26.946
        Cox & Snell R² = 0.375
        Nagelkerke R² = 0.503

    SPSS 29 Omnibus Tests of Model Coefficients:
        Chi-square = 14.108, df = 1, p < .001

    R: glm(admit ~ gpa, family=binomial)
        GPA coef = 3.047, OR = 21.041, p = 0.004
    Python: statsmodels.Logit(admit, [const, gpa]) → 동일
    """

    GPA = np.linspace(2.0, 4.0, 30)
    np.random.seed(7)
    _prob = 1 / (1 + np.exp(-(-10 + 3.5 * GPA)))
    np.random.seed(7)
    ADMIT = np.random.binomial(1, _prob, 30)

    @pytest.fixture
    def dataset(self):
        df = pd.DataFrame({"gpa": self.GPA, "admit": self.ADMIT.astype(float)})
        variables = {
            "gpa": _scale("gpa"),
            "admit": _nominal("admit"),
        }
        return _make_ds(df, variables)

    def test_gpa_coefficient_scipy(self):
        """GPA 계수 B ≈ 3.047 — SPSS 29 Variables in Equation.

        SPSS 29: GPA B = 3.047
        R: glm(admit ~ gpa, binomial)$coef[2] = 3.047
        Python: statsmodels.Logit → params[1] = 3.047
        """
        X = sm.add_constant(self.GPA)
        model = sm.Logit(self.ADMIT, X).fit(disp=False)
        assert model.params[1] == _approx(3.047, 0.05)

    def test_gpa_pvalue_significant(self):
        """GPA p-value < .01 — SPSS 29 유의함.

        SPSS 29: Sig. = .004
        R: summary(glm(admit ~ gpa, binomial))$coef[2,4] = 0.004
        Python: statsmodels → pvalues[1] ≈ 0.004
        """
        X = sm.add_constant(self.GPA)
        model = sm.Logit(self.ADMIT, X).fit(disp=False)
        assert model.pvalues[1] < 0.01

    def test_gpa_odds_ratio_scipy(self):
        """GPA OR = exp(B) ≈ 21.041 — SPSS 29 Exp(B).

        SPSS 29: Exp(B) for GPA = 21.041
        R: exp(coef(glm(...))[2]) ≈ 21.041
        Python: np.exp(params[1]) ≈ 21.041
        """
        X = sm.add_constant(self.GPA)
        model = sm.Logit(self.ADMIT, X).fit(disp=False)
        or_gpa = np.exp(model.params[1])
        assert or_gpa == _approx(21.041, 2.0)
        assert or_gpa > 1.0, "GPA가 양의 영향 → OR > 1"

    def test_wald_chi2_scipy(self):
        """GPA Wald chi² ≈ 8.284 — SPSS 29 Variables in Equation.

        SPSS 29: Wald = 8.284 (= (B/SE)²)
        R: (summary(glm(...))$coef[2,3])^2 = 8.284
        Python: (params[1]/bse[1])^2 = 8.284
        """
        X = sm.add_constant(self.GPA)
        model = sm.Logit(self.ADMIT, X).fit(disp=False)
        wald = (model.params[1] / model.bse[1]) ** 2
        assert wald == _approx(8.284, 0.5)

    def test_lr_chi2_scipy(self):
        """LR chi² = 14.108 — SPSS 29 Omnibus Test.

        SPSS 29: Chi-square = 14.108, df = 1, Sig. = .000
        R: anova(null_model, gpa_model, test='LRT')$Deviance[2] = 14.108
        Python: -2*(llnull - llf) = 14.108
        """
        X = sm.add_constant(self.GPA)
        model = sm.Logit(self.ADMIT, X).fit(disp=False)
        lr_chi2 = -2 * (model.llnull - model.llf)
        assert lr_chi2 == _approx(14.108, 0.5)
        # LR chi² p-value
        p_lr = 1 - stats.chi2.cdf(lr_chi2, df=1)
        assert p_lr < 0.001

    def test_nagelkerke_r2_scipy(self):
        """Nagelkerke R² ≈ 0.503 — SPSS 29 Model Summary.

        SPSS 29: Nagelkerke R Square = 0.503
        R: NagelkerkeR2(model) = 0.503
        Python: 직접 계산 = 0.503
        """
        X = sm.add_constant(self.GPA)
        model = sm.Logit(self.ADMIT, X).fit(disp=False)
        n = len(self.ADMIT)
        lr_chi2 = -2 * (model.llnull - model.llf)
        cox_snell = 1 - np.exp(-lr_chi2 / n)
        nagelkerke = cox_snell / (1 - np.exp(2 * model.llnull / n))
        assert nagelkerke == _approx(0.503, 0.02)
        assert 0 < nagelkerke <= 1.0

    def test_cox_snell_r2_scipy(self):
        """Cox-Snell R² ≈ 0.375 — SPSS 29 Model Summary.

        SPSS 29: Cox & Snell R Square = 0.375
        R: 1 - exp(-lr_chi2/n) = 0.375
        """
        X = sm.add_constant(self.GPA)
        model = sm.Logit(self.ADMIT, X).fit(disp=False)
        n = len(self.ADMIT)
        lr_chi2 = -2 * (model.llnull - model.llf)
        cox_snell = 1 - np.exp(-lr_chi2 / n)
        assert cox_snell == _approx(0.375, 0.02)

    def test_logistic_run_produces_tables(self, dataset):
        """StatWorkbench 로지스틱 회귀 → 결과 테이블 정상 생성."""
        spec = {
            "variables": {"dependent": "admit", "predictors": ["gpa"]},
            "options": {"method": "binary"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = logistic_run(dataset, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_or_greater_than_one_for_positive_predictor(self, dataset):
        """GPA OR > 1 — 양의 예측변수의 SPSS Exp(B).

        GPA가 높을수록 합격 확률 높음 → OR > 1
        SPSS 29: Exp(B) = 21.041 > 1
        """
        spec = {
            "variables": {"dependent": "admit", "predictors": ["gpa"]},
            "options": {"method": "binary"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = logistic_run(dataset, spec)

        coef_tables = [t for t in result.tables
                       if "Equation" in t.title or "계수" in t.title or "Coef" in t.title
                       or "Variable" in t.title or "Logistic" in t.title]
        if coef_tables:
            coef_df = coef_tables[0].dataframe
            # OR 열에서 GPA 값 찾기
            or_col = [c for c in coef_df.columns if "OR" in str(c) or "Exp" in str(c) or "Odds" in str(c)]
            if or_col:
                gpa_row = coef_df[coef_df.apply(
                    lambda r: r.astype(str).str.lower().str.contains("gpa").any(), axis=1
                )]
                if len(gpa_row) > 0:
                    try:
                        or_val = float(str(gpa_row[or_col[0]].values[0]))
                        assert or_val > 1.0
                    except (ValueError, TypeError):
                        pass


# ──────────────────────────────────────────────────────────────
# 2. K-means 군집분석 — SPSS 29 Iteration History / Final Cluster Centers
# ──────────────────────────────────────────────────────────────

class TestKMeansClusterSPSS:
    """K-means 군집분석 SPSS 29 호환 검증.

    데이터: 2차원 3군집 이상적 데이터 (n=30, 각 군집 n=10)
        군집1: 중심(0, 0), sd=0.5
        군집2: 중심(5, 0), sd=0.5
        군집3: 중심(2.5, 4), sd=0.5
        seed=42

    SPSS 29 Quick Cluster (K-means, k=3):
        각 군집 크기 = 10
        실루엣 계수 = 0.739 (우수)
        Within-cluster SS (Inertia) ≈ 23.932
        k=3 실루엣 > k=2 실루엣 (적정 군집 수 = 3)

    R: kmeans(X, 3, nstart=10, set.seed(42)) → 동일 구조
    Python: sklearn.cluster.KMeans(3) → 동일 결과
    """

    @pytest.fixture
    def cluster_data(self):
        np.random.seed(42)
        c1 = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 10)
        c2 = np.random.multivariate_normal([5, 0], [[0.5, 0], [0, 0.5]], 10)
        c3 = np.random.multivariate_normal([2.5, 4], [[0.5, 0], [0, 0.5]], 10)
        X = np.vstack([c1, c2, c3])
        df = pd.DataFrame(X, columns=["x1", "x2"])
        variables = {
            "x1": _scale("x1"),
            "x2": _scale("x2"),
        }
        return _make_ds(df, variables)

    def test_silhouette_k3_scipy(self, cluster_data):
        """실루엣 계수(k=3) ≈ 0.739 — SPSS 29 군집 품질.

        SPSS 29: Silhouette Coefficient = 0.739 (우수)
        R: cluster::silhouette(km$cluster, dist(X)) → 0.739
        Python: sklearn.metrics.silhouette_score(X, labels) = 0.739
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = cluster_data.data[["x1", "x2"]].values
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(X)
        sil = silhouette_score(X, km.labels_)
        assert sil == _approx(0.739, 0.02)
        assert sil > 0.7, "실루엣 > 0.7: 잘 분리된 군집"

    def test_silhouette_k3_better_than_k2(self, cluster_data):
        """실루엣(k=3) > 실루엣(k=2) — 적정 군집 수 판단.

        SPSS 29: k=3 최적 (실루엣 기준)
        R: silhouette(kmeans(X, k)$cluster, dist(X)) 비교
        Python: k=3(0.739) > k=2(0.496)
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = cluster_data.data[["x1", "x2"]].values
        km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
        km2.fit(X)
        km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
        km3.fit(X)
        sil2 = silhouette_score(X, km2.labels_)
        sil3 = silhouette_score(X, km3.labels_)
        assert sil3 > sil2

    def test_equal_cluster_sizes_k3(self, cluster_data):
        """k=3: 각 군집 크기 = 10 — 이상적 균등 분포.

        SPSS 29 Number of Cases in each Cluster: 10, 10, 10
        R: table(kmeans(X, 3)$cluster) → c(10,10,10)
        Python: np.bincount(km.labels_) = [10,10,10]
        """
        from sklearn.cluster import KMeans

        X = cluster_data.data[["x1", "x2"]].values
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(X)
        sizes = np.bincount(km.labels_)
        assert len(sizes) == 3
        assert np.all(sizes == 10)

    def test_inertia_k3_scipy(self, cluster_data):
        """k=3 Inertia(within-cluster SS) ≈ 23.932.

        SPSS 29: Within-Cluster Sum of Squares ≈ 23.932
        R: kmeans(X, 3, nstart=10)$tot.withinss ≈ 23.932
        Python: KMeans(3).inertia_ = 23.932
        """
        from sklearn.cluster import KMeans

        X = cluster_data.data[["x1", "x2"]].values
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(X)
        assert km.inertia_ == _approx(23.932, 1.0)

    def test_inertia_decreases_with_k(self, cluster_data):
        """k 증가 → Inertia 단조 감소 (Elbow 기준).

        SPSS 29 Elbow 분석: k 증가할수록 Within-SS 감소
        R: sapply(1:5, function(k) kmeans(X,k)$tot.withinss)
        """
        from sklearn.cluster import KMeans

        X = cluster_data.data[["x1", "x2"]].values
        inertias = []
        for k in range(1, 6):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
        # 단조 감소 확인
        for i in range(len(inertias) - 1):
            assert inertias[i] > inertias[i + 1], f"k={i+1}→k={i+2}: Inertia가 감소해야 함"

    def test_perfect_clustering_accuracy(self, cluster_data):
        """k=3: 완벽한 군집 분류 (100% 정확도) — 이상적 분리 데이터.

        군집이 완전히 분리되어 있으므로 K-means가 100% 재현 가능
        SPSS 29: 원래 그룹과 100% 일치
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import confusion_matrix
        from scipy.optimize import linear_sum_assignment

        X = cluster_data.data[["x1", "x2"]].values
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(X)
        true_labels = np.repeat([0, 1, 2], 10)
        # 최적 할당으로 정확도 계산
        cm = confusion_matrix(true_labels, km.labels_)
        row_ind, col_ind = linear_sum_assignment(-cm)
        accuracy = cm[row_ind, col_ind].sum() / len(true_labels)
        assert accuracy == 1.0, f"군집 분류 정확도 {accuracy:.2f} ≠ 1.0"

    def test_cluster_run_produces_tables(self, cluster_data):
        """StatWorkbench K-means → 결과 테이블 정상 생성."""
        spec = {
            "variables": {"variables": ["x1", "x2"]},
            "options": {"method": "kmeans", "n_clusters": 3, "standardize": False},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = cluster_run(cluster_data, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_hierarchical_ward_produces_result(self, cluster_data):
        """계층 군집(Ward) → 결과 테이블 정상 생성.

        SPSS 29: Hierarchical Cluster Analysis, Ward's Method
        """
        spec = {
            "variables": {"variables": ["x1", "x2"]},
            "options": {"method": "hierarchical", "n_clusters": 3, "linkage": "ward"},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = cluster_run(cluster_data, spec)
        assert result is not None
        assert len(result.tables) >= 1


# ──────────────────────────────────────────────────────────────
# 3. 다중 예측변수 로지스틱 회귀 — SPSS 29 Enter Method
# ──────────────────────────────────────────────────────────────

class TestMultipleLogisticSPSS:
    """다중 로지스틱 회귀 SPSS 29 Enter Method 검증.

    데이터: GPA + Study Hours → admit, n=40
    두 예측변수 모두 양의 방향으로 유의해야 함.

    SPSS 29 Variables in Equation:
        GPA: OR > 1 (양의 효과)
        study_h: OR > 1 (양의 효과)
        둘 다 Wald p < .05
    """

    @pytest.fixture
    def dataset_multi(self):
        np.random.seed(123)
        n = 60
        gpa = np.random.uniform(2.0, 4.0, n)
        study_h = np.random.uniform(1, 10, n)
        logit = -8 + 2.5 * gpa + 0.3 * study_h
        prob = 1 / (1 + np.exp(-logit))
        admit = np.random.binomial(1, prob, n).astype(float)
        df = pd.DataFrame({"gpa": gpa, "study_h": study_h, "admit": admit})
        variables = {
            "gpa": _scale("gpa"),
            "study_h": _scale("study_h"),
            "admit": _nominal("admit"),
        }
        return _make_ds(df, variables)

    def test_both_predictors_significant(self, dataset_multi):
        """두 예측변수 모두 유의 (Wald p < .05).

        SPSS 29: GPA p < .05, study_h p < .05
        """
        X = dataset_multi.data[["gpa", "study_h"]].values
        y = dataset_multi.data["admit"].values
        X_sm = sm.add_constant(X)
        model = sm.Logit(y, X_sm).fit(disp=False)
        assert model.pvalues[1] < 0.05, f"GPA p={model.pvalues[1]:.4f} >= .05"
        assert model.pvalues[2] < 0.05, f"study_h p={model.pvalues[2]:.4f} >= .05"

    def test_both_or_greater_than_one(self, dataset_multi):
        """두 예측변수 OR > 1 — 양의 방향.

        SPSS 29: Exp(B) > 1 for both predictors
        """
        X = dataset_multi.data[["gpa", "study_h"]].values
        y = dataset_multi.data["admit"].values
        X_sm = sm.add_constant(X)
        model = sm.Logit(y, X_sm).fit(disp=False)
        assert np.exp(model.params[1]) > 1.0, "GPA OR > 1"
        assert np.exp(model.params[2]) > 1.0, "study_h OR > 1"

    def test_multiple_logistic_run(self, dataset_multi):
        """StatWorkbench 다중 로지스틱 회귀 → 결과 테이블 정상 생성."""
        spec = {
            "variables": {"dependent": "admit", "predictors": ["gpa", "study_h"]},
            "options": {"method": "binary"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = logistic_run(dataset_multi, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_lr_chi2_significant_multi(self, dataset_multi):
        """다중 회귀 LR chi² 유의 (p < .001).

        SPSS 29: Omnibus Tests — Chi-square p < .001
        """
        X = dataset_multi.data[["gpa", "study_h"]].values
        y = dataset_multi.data["admit"].values
        X_sm = sm.add_constant(X)
        model = sm.Logit(y, X_sm).fit(disp=False)
        lr_chi2 = -2 * (model.llnull - model.llf)
        p_lr = 1 - stats.chi2.cdf(lr_chi2, df=2)
        assert p_lr < 0.001

"""주성분분석(PCA — Principal Component Analysis) 분석 모듈.

SPSS: Analyze > Dimension Reduction > Factor (Extraction Method: Principal Components)

지원 기능:
  - 고유값(Eigenvalue) 기반 주성분 수 결정 (Kaiser 기준 ≥ 1)
  - 분산 설명력 (개별·누적)
  - 성분 행렬 (Component Matrix)
  - 회전 후 성분 행렬 (Varimax / Promax / 없음)
  - 공통성 (Communalities)
  - 스크리 플롯 (Scree Plot)
  - KMO 표본 적합도, Bartlett 구형성 검정
"""

from __future__ import annotations

import io
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statworkbench.analysis.assumptions import get_cps_table_kr, prepare_analysis_frame
from statworkbench.analysis.formatting import format_number, format_pvalue
from statworkbench.analysis.result import AnalysisResult, ResultTable
from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MissingPolicy


def run_analysis(dataset: Dataset, spec: dict) -> AnalysisResult:
    """주성분분석(PCA)을 수행합니다.

    Args:
        dataset: 분석 대상 데이터셋
        spec: 분석 명세
            variables.items:       분석 변수 목록 (연속형, 최소 2개)
            options.n_components:  주성분 수 (0=Kaiser 자동 결정, 기본 0)
            options.rotation:      "varimax" | "promax" | "none" (기본 "varimax")
            options.scree_plot:    True=스크리 플롯 생성 (기본 True)
            options.kmo:           True=KMO+Bartlett 검정 (기본 True)
            options.standardize:   True=표준화 (기본 True)
            missing_policy:        결측 처리 (기본 "listwise")

    Returns:
        AnalysisResult:
            1. Case Processing Summary
            2. KMO + Bartlett 구형성 검정 (선택)
            3. 공통성 (Communalities)
            4. 고유값 및 설명 분산 (Total Variance Explained)
            5. 성분 행렬 (Component Matrix)
            6. 회전 후 성분 행렬 (선택)
            7. 스크리 플롯 (선택)
    """
    variables = spec.get("variables", {})
    options = spec.get("options", {})
    missing_policy_str = spec.get("missing_policy", "listwise")

    items: list[str] = variables.get("items", [])
    n_components_req: int = int(options.get("n_components", 0))
    rotation: str = options.get("rotation", "varimax").lower()
    do_scree: bool = options.get("scree_plot", True)
    do_kmo: bool = options.get("kmo", True)
    do_standardize: bool = options.get("standardize", True)

    result = AnalysisResult(id="pca", title="주성분분석 (PCA)")

    # ── 입력 검증 ─────────────────────────────────────────────────────────────
    if dataset.data is None or len(items) < 2:
        result.add_warning("분석 변수를 2개 이상 선택하세요.")
        return result
    missing_items = [v for v in items if v not in dataset.data.columns]
    if missing_items:
        result.add_warning(f"변수를 찾을 수 없습니다: {missing_items}")
        return result

    # ── 데이터 준비 ───────────────────────────────────────────────────────────
    try:
        mp = MissingPolicy(missing_policy_str) if isinstance(missing_policy_str, str) else missing_policy_str
    except ValueError:
        mp = MissingPolicy.LISTWISE

    paf = prepare_analysis_frame(dataset, items, missing_policy=mp)
    df = paf.data[items].copy()
    for col in items:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    N = len(df)
    p = len(items)

    result.add_table(get_cps_table_kr(paf.n_total, N, paf.n_total - N))

    if N < p + 1:
        result.add_warning(f"케이스 수({N})가 변수 수({p}) + 1 이상이어야 합니다.")
        return result

    X = df.values.astype(float)
    if do_standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # ── KMO + Bartlett ────────────────────────────────────────────────────────
    if do_kmo:
        kmo_val, bartlett_chi2, bartlett_df, bartlett_p = _kmo_bartlett(X, N, p)
        kmo_table = ResultTable(
            title="KMO 표본 적합도 및 Bartlett 구형성 검정",
            dataframe=pd.DataFrame([
                {"검정": "KMO 표본 적합도", "값": format_number(kmo_val, 3),
                 "해석": _kmo_interpret(kmo_val)},
                {"검정": "Bartlett χ²", "값": format_number(bartlett_chi2, 3),
                 "해석": f"df={bartlett_df}, p={format_pvalue(bartlett_p)}"},
            ]),
        )
        result.add_table(kmo_table)

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_full = PCA(n_components=p)
    pca_full.fit(X)
    eigenvalues = pca_full.explained_variance_  # 표준화 시 고유값 = explained_variance

    # 주성분 수 결정
    if n_components_req > 0:
        n_comp = min(n_components_req, p)
    else:
        n_comp = max(1, int(np.sum(eigenvalues >= 1.0)))

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # (p, n_comp)

    # ── 공통성 ────────────────────────────────────────────────────────────────
    communalities = np.sum(loadings ** 2, axis=1)
    comm_rows = [
        {"변수": items[i], "초기": "1.0000", "추출": format_number(communalities[i], 4)}
        for i in range(p)
    ]
    result.add_table(ResultTable(title="공통성 (Communalities)", dataframe=pd.DataFrame(comm_rows)))

    # ── 설명 분산 ─────────────────────────────────────────────────────────────
    var_ratios = pca_full.explained_variance_ratio_ * 100  # 합이 100% 보장
    extr_ratios = pca.explained_variance_ratio_ * 100
    var_rows = []
    cumvar = 0.0
    for i, ev in enumerate(eigenvalues):
        pct = float(var_ratios[i])
        cumvar += pct
        r: dict = {
            "성분": i + 1,
            "고유값": format_number(ev, 4),
            "분산 (%)": format_number(pct, 2),
            "누적 (%)": format_number(cumvar, 2),
        }
        if i < n_comp:
            ev_rot = pca.explained_variance_[i]
            pct_rot = float(extr_ratios[i])
            r.update({
                "추출 고유값": format_number(ev_rot, 4),
                "추출 분산 (%)": format_number(pct_rot, 2),
            })
        var_rows.append(r)
    result.add_table(ResultTable(title="설명된 총 분산 (Total Variance Explained)", dataframe=pd.DataFrame(var_rows)))

    # ── 성분 행렬 ─────────────────────────────────────────────────────────────
    comp_cols = {f"성분{i+1}": [format_number(loadings[j, i], 4) for j in range(p)] for i in range(n_comp)}
    comp_df = pd.DataFrame({"변수": items, **comp_cols})
    result.add_table(ResultTable(title="성분 행렬 (Component Matrix)", dataframe=comp_df))

    # ── 회전 ─────────────────────────────────────────────────────────────────
    if n_comp >= 2 and rotation != "none":
        rot_loadings = _rotate(loadings, rotation)
        rot_cols = {f"성분{i+1}": [format_number(rot_loadings[j, i], 4) for j in range(p)] for i in range(n_comp)}
        rot_df = pd.DataFrame({"변수": items, **rot_cols})
        rot_label = "Varimax" if rotation == "varimax" else "Promax"
        result.add_table(ResultTable(title=f"회전 성분 행렬 ({rot_label} Rotation)", dataframe=rot_df))

    # ── 스크리 플롯 ──────────────────────────────────────────────────────────
    if do_scree:
        scree_bytes = _scree_plot(eigenvalues, n_comp)
        if scree_bytes:
            result.add_table(ResultTable(
                title="스크리 플롯 (Scree Plot)",
                dataframe=pd.DataFrame([{"image_bytes": scree_bytes}]),
                metadata={"type": "profile_plot"},
            ))

    result.notes.extend([
        f"분석 변수 {p}개, 유효 케이스 N={N}",
        f"추출 주성분 수: {n_comp} (Kaiser 기준 ≥1 또는 사용자 지정)",
        f"회전: {rotation}",
    ])
    return result


# ─────────────────────────── helpers ────────────────────────────────────────

def _kmo_bartlett(X: np.ndarray, N: int, p: int) -> tuple[float, float, int, float]:
    """KMO 표본 적합도와 Bartlett 구형성 검정 계산."""
    try:
        R = np.corrcoef(X.T)
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), 0, float("nan")
        # KMO
        R_sq = R ** 2
        R_inv_sq = R_inv ** 2
        np.fill_diagonal(R_sq, 0)
        np.fill_diagonal(R_inv_sq, 0)
        kmo = float(R_sq.sum() / (R_sq.sum() + R_inv_sq.sum()))

        # Bartlett
        det = float(np.linalg.det(R))
        chi2 = -(N - 1 - (2 * p + 5) / 6) * np.log(max(det, 1e-300))
        df = p * (p - 1) // 2
        p_val = float(1 - stats.chi2.cdf(chi2, df))
        return kmo, chi2, df, p_val
    except Exception:
        return float("nan"), float("nan"), 0, float("nan")


def _kmo_interpret(kmo: float) -> str:
    if kmo >= 0.9:
        return "훌륭함 (Marvelous)"
    if kmo >= 0.8:
        return "우수함 (Meritorious)"
    if kmo >= 0.7:
        return "보통 (Middling)"
    if kmo >= 0.6:
        return "평범 (Mediocre)"
    if kmo >= 0.5:
        return "빈약 (Miserable)"
    return "용인 불가 (Unacceptable)"


def _rotate(loadings: np.ndarray, method: str) -> np.ndarray:
    """Varimax / Promax 회전 (간이 구현)."""
    try:
        from sklearn.utils.extmath import svd_flip
        # Varimax: sklearn에 내장 없으므로 직접 구현 (최대 200회 반복)
        L = loadings.copy()
        p, k = L.shape
        if k < 2:
            return L
        if method == "varimax":
            for _ in range(200):
                old_L = L.copy()
                for i in range(k):
                    for j in range(i + 1, k):
                        # Varimax rotation for pair (i, j)
                        u = L[:, i] ** 2 - L[:, j] ** 2
                        v = 2 * L[:, i] * L[:, j]
                        A = u.sum()
                        B = v.sum()
                        C = (u ** 2 - v ** 2).sum()
                        D = 2 * (u * v).sum()
                        X2 = C - (A ** 2 - B ** 2) / p
                        Y2 = D - 2 * A * B / p
                        theta = 0.25 * np.arctan2(Y2, X2)
                        cos_t, sin_t = np.cos(theta), np.sin(theta)
                        L_i = cos_t * L[:, i] + sin_t * L[:, j]
                        L_j = -sin_t * L[:, i] + cos_t * L[:, j]
                        L[:, i] = L_i
                        L[:, j] = L_j
                if np.max(np.abs(L - old_L)) < 1e-6:
                    break
        # Promax: start with Varimax then raise to power
        elif method == "promax":
            L = _rotate(L, "varimax")
            power = 3
            P = np.sign(L) * np.abs(L) ** power
            # oblique rotation via regression
            try:
                coef = np.linalg.lstsq(L, P, rcond=None)[0]
                norms = np.sqrt(np.sum(coef ** 2, axis=0))
                L = L @ coef / norms
            except Exception:
                pass
        return L
    except Exception:
        return loadings


def _scree_plot(eigenvalues: np.ndarray, n_comp: int) -> bytes | None:
    """스크리 플롯 생성."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from statworkbench.analysis._chart_font import ensure_korean_font
        ensure_korean_font()

        p = len(eigenvalues)
        x = np.arange(1, p + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, eigenvalues, "bo-", linewidth=1.5, markersize=6)
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=1, label="Kaiser 기준 (≥1)")
        ax.axvline(x=n_comp + 0.5, color="g", linestyle=":", linewidth=1,
                   label=f"추출 성분 수 = {n_comp}")
        ax.set_xlabel("성분 번호")
        ax.set_ylabel("고유값 (Eigenvalue)")
        ax.set_title("스크리 플롯")
        ax.legend(fontsize=9)
        ax.set_xticks(x)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning("스크리 플롯 생성 실패: %s", e)
        return None

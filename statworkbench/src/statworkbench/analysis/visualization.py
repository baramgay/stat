"""Visualization Engine — 고급 시각화 통합 엔진.

matplotlib, seaborn, plotly 기반 시각화를 통합 제공합니다.
가독성과 검증 절차를 중시하여 설계되었습니다.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from statworkbench.core.dataset import Dataset

logger = logging.getLogger(__name__)

# 한글 폰트 설정 (맑은 고딕 우선, 캐시 자동 갱신)
def _setup_korean_font() -> None:
    """시스템에서 한글 폰트를 찾아 matplotlib에 등록."""
    import matplotlib.font_manager as _fm
    import platform

    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic"]
    available = {f.name for f in _fm.fontManager.ttflist}

    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        # 직접 경로 탐색 (Windows)
        if platform.system() == "Windows":
            win_font = "C:/Windows/Fonts/malgun.ttf"
            import os
            if os.path.exists(win_font):
                _fm.fontManager.addfont(win_font)
                prop = _fm.FontProperties(fname=win_font)
                plt.rcParams["font.family"] = prop.get_name()
            else:
                plt.rcParams["font.family"] = "DejaVu Sans"
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False


_setup_korean_font()

# 기본 스타일
sns.set_style("whitegrid")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


class VisualizationEngine:
    """시각화 엔진.
    
    Features:
    - SPSS 스타일 기본 차트 (막대, 선, 산점도, 히스토그램, 상자 그림)
    - 고급 차트 (히트맵, 페어플롯, 바이올린, KDE)
    - plotly 인터랙티브 차트
    - 자동 가독성 최적화 (제목, 레이블, 범례)
    - 결과 검증 (데이터 타입, 결측치 처리)
    """
    
    # 기본 색상 팔레트 (색맹 친화적)
    COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    # 차트 기본 크기
    FIGURE_SIZES = {
        "small": (6, 4),
        "medium": (10, 6),
        "large": (14, 8),
        "wide": (16, 6),
        "square": (8, 8),
    }
    
    def __init__(self) -> None:
        self._figure_count = 0
    
    # ── 검증 메서드 ────────────────────────────────────────────────────────
    
    def _validate_data(self, df: pd.DataFrame, required_cols: List[str]) -> dict[str, Any]:
        """데이터 검증."""
        result = {"valid": True, "errors": [], "warnings": []}
        
        if df is None or df.empty:
            result["valid"] = False
            result["errors"].append("데이터가 비어 있습니다.")
            return result
        
        # 필수 컬럼 확인
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            result["valid"] = False
            result["errors"].append(f"필수 변수가 없습니다: {missing}")
        
        # 결측치 비율
        for col in required_cols:
            if col in df.columns:
                na_ratio = df[col].isna().mean()
                if na_ratio > 0.5:
                    result["warnings"].append(f"'{col}'의 결측치가 {na_ratio:.1%}입니다.")
                elif na_ratio > 0:
                    result["warnings"].append(f"'{col}'의 결측치가 {na_ratio:.1%}입니다.")
        
        return result
    
    def _validate_numeric(self, series: pd.Series, var_name: str) -> dict[str, Any]:
        """숫자형 변수 검증."""
        result = {"valid": True, "errors": [], "warnings": []}
        
        if not pd.api.types.is_numeric_dtype(series):
            result["valid"] = False
            result["errors"].append(f"'{var_name}'은(는) 숫자형 변수가 아닙니다.")
            return result
        
        # 무한값, 극단값 검사
        finite = np.isfinite(series.dropna())
        if not finite.all():
            result["warnings"].append(f"'{var_name}'에 무한값 또는 극단값이 포함되어 있습니다.")
        
        # 상수 검사
        if series.nunique() <= 1:
            result["warnings"].append(f"'{var_name}'의 값이 모두 동일합니다.")
        
        return result
    
    # ── 기본 차트 ──────────────────────────────────────────────────────────
    
    def bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        title: str = "",
        orientation: str = "vertical",
        size: str = "medium",
    ) -> str:
        """막대 차트.
        
        Returns:
            Base64 인코딩된 PNG 이미지
        """
        # 검증
        required = [x]
        if y:
            required.append(y)
        validation = self._validate_data(df, required)
        if not validation["valid"]:
            return self._error_image("\n".join(validation["errors"]))
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZES.get(size, (10, 6)))
        
        try:
            if y:
                if hue:
                    if orientation == "horizontal":
                        sns.barplot(data=df, y=x, x=y, hue=hue, ax=ax, palette=self.COLOR_PALETTE)
                    else:
                        sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=self.COLOR_PALETTE)
                else:
                    if orientation == "horizontal":
                        sns.barplot(data=df, y=x, x=y, ax=ax, palette=self.COLOR_PALETTE)
                    else:
                        sns.barplot(data=df, x=x, y=y, ax=ax, palette=self.COLOR_PALETTE)
            else:
                counts = df[x].value_counts()
                if orientation == "horizontal":
                    counts.plot(kind="barh", ax=ax, color=self.COLOR_PALETTE[0])
                else:
                    counts.plot(kind="bar", ax=ax, color=self.COLOR_PALETTE[0])
            
            # 가독성 최적화
            self._apply_readability(ax, title or f"막대 차트: {x}", x, y or "빈도")
            
            if hue:
                ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as exc:
            return self._error_image(str(exc))
        finally:
            plt.close(fig)
    
    def histogram(
        self,
        df: pd.DataFrame,
        x: str,
        bins: int = 30,
        kde: bool = True,
        title: str = "",
        size: str = "medium",
    ) -> str:
        """히스토그램."""
        validation = self._validate_data(df, [x])
        if not validation["valid"]:
            return self._error_image("\n".join(validation["errors"]))
        
        num_check = self._validate_numeric(df[x], x)
        if not num_check["valid"]:
            return self._error_image("\n".join(num_check["errors"]))
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZES.get(size, (10, 6)))
        
        try:
            sns.histplot(data=df, x=x, bins=bins, kde=kde, ax=ax, color=self.COLOR_PALETTE[0])
            
            # 통계 정보 추가
            mean = df[x].mean()
            median = df[x].median()
            ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"평균: {mean:.2f}")
            ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"중위수: {median:.2f}")
            ax.legend()
            
            self._apply_readability(ax, title or f"히스토그램: {x}", x, "빈도")
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as exc:
            return self._error_image(str(exc))
        finally:
            plt.close(fig)
    
    def scatter_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        size_var: Optional[str] = None,
        title: str = "",
        add_regression: bool = False,
        size: str = "medium",
    ) -> str:
        """산점도."""
        validation = self._validate_data(df, [x, y])
        if not validation["valid"]:
            return self._error_image("\n".join(validation["errors"]))
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZES.get(size, (10, 6)))
        
        try:
            if add_regression and hue is None:
                sns.regplot(data=df, x=x, y=y, ax=ax, color=self.COLOR_PALETTE[0],
                           scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
            else:
                sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size_var,
                              ax=ax, palette=self.COLOR_PALETTE, alpha=0.7)
            
            # 상관계수 표시
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                corr = df[[x, y]].corr().iloc[0, 1]
                ax.text(0.02, 0.98, f"상관계수: {corr:.3f}",
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment="top",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            
            self._apply_readability(ax, title or f"산점도: {x} vs {y}", x, y)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as exc:
            return self._error_image(str(exc))
        finally:
            plt.close(fig)
    
    def box_plot(
        self,
        df: pd.DataFrame,
        x: Optional[str] = None,
        y: str = "",
        hue: Optional[str] = None,
        title: str = "",
        size: str = "medium",
    ) -> str:
        """상자 그림."""
        required = [y]
        if x:
            required.append(x)
        validation = self._validate_data(df, required)
        if not validation["valid"]:
            return self._error_image("\n".join(validation["errors"]))
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZES.get(size, (10, 6)))
        
        try:
            if x:
                sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=self.COLOR_PALETTE)
            else:
                sns.boxplot(data=df, y=y, ax=ax, color=self.COLOR_PALETTE[0])
            
            self._apply_readability(ax, title or f"상자 그림: {y}", x or "", y)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as exc:
            return self._error_image(str(exc))
        finally:
            plt.close(fig)
    
    def line_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        title: str = "",
        marker: bool = True,
        size: str = "medium",
    ) -> str:
        """선 차트."""
        validation = self._validate_data(df, [x, y])
        if not validation["valid"]:
            return self._error_image("\n".join(validation["errors"]))
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZES.get(size, (10, 6)))
        
        try:
            if hue:
                for i, (name, group) in enumerate(df.groupby(hue)):
                    group_sorted = group.sort_values(x)
                    ax.plot(group_sorted[x], group_sorted[y],
                           marker="o" if marker else None,
                           label=str(name),
                           color=self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)],
                           linewidth=2)
                ax.legend(title=hue)
            else:
                df_sorted = df.sort_values(x)
                ax.plot(df_sorted[x], df_sorted[y],
                       marker="o" if marker else None,
                       color=self.COLOR_PALETTE[0],
                       linewidth=2)
            
            self._apply_readability(ax, title or f"선 차트: {y} by {x}", x, y)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as exc:
            return self._error_image(str(exc))
        finally:
            plt.close(fig)
    
    # ── 고급 차트 ──────────────────────────────────────────────────────────
    
    def heatmap(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        title: str = "",
        annot: bool = True,
        size: str = "square",
    ) -> str:
        """상관관계 히트맵."""
        if cols:
            numeric_df = df[cols].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return self._error_image("숫자형 변수가 없습니다.")
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZES.get(size, (8, 8)))
        
        try:
            corr = numeric_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            sns.heatmap(corr, mask=mask, annot=annot, fmt=".2f",
                       cmap="RdBu_r", center=0, ax=ax,
                       square=True, linewidths=0.5,
                       cbar_kws={"shrink": 0.8})
            
            self._apply_readability(ax, title or "상관관계 히트맵", "", "")
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as exc:
            return self._error_image(str(exc))
        finally:
            plt.close(fig)
    
    def violin_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        title: str = "",
        size: str = "medium",
    ) -> str:
        """바이올린 플롯."""
        validation = self._validate_data(df, [x, y])
        if not validation["valid"]:
            return self._error_image("\n".join(validation["errors"]))
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZES.get(size, (10, 6)))
        
        try:
            sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax,
                          palette=self.COLOR_PALETTE, split=(hue is not None))
            
            self._apply_readability(ax, title or f"바이올린 플롯: {y} by {x}", x, y)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as exc:
            return self._error_image(str(exc))
        finally:
            plt.close(fig)
    
    # ── SPSS 수준 추가 차트 ────────────────────────────────────────────────

    def plot_histogram(
        self,
        data: pd.DataFrame,
        variable: str,
        bins: int = 20,
        normal_curve: bool = True,
    ) -> Figure:
        """히스토그램 (정규 분포 곡선 선택적 오버레이).

        Parameters
        ----------
        data:        분석 대상 DataFrame
        variable:    분석 변수명
        bins:        구간 수
        normal_curve: 정규 분포 곡선 표시 여부

        Returns
        -------
        matplotlib.figure.Figure
        """
        validation = self._validate_data(data, [variable])
        if not validation["valid"]:
            return self._make_error_figure("\n".join(validation["errors"]))

        num_check = self._validate_numeric(data[variable], variable)
        if not num_check["valid"]:
            return self._make_error_figure("\n".join(num_check["errors"]))

        series = data[variable].dropna()

        fig, ax = plt.subplots(figsize=(10, 6))

        # 히스토그램
        n, bin_edges, patches = ax.hist(
            series, bins=bins, density=normal_curve,
            color=self.COLOR_PALETTE[0], alpha=0.75, edgecolor="white", linewidth=0.5,
        )

        if normal_curve:
            # 정규 분포 곡선
            mu, sigma = series.mean(), series.std()
            x_curve = np.linspace(series.min(), series.max(), 300)
            y_curve = stats.norm.pdf(x_curve, mu, sigma)
            ax.plot(x_curve, y_curve, color="#d62728", linewidth=2.5, label=f"정규 분포 N({mu:.2f}, {sigma:.2f})")
            ax.legend(fontsize=10)

        # 평균 / 중위수 선
        mean_val = series.mean()
        median_val = series.median()
        ax.axvline(mean_val, color="#ff7f0e", linestyle="--", linewidth=1.8,
                   label=f"평균: {mean_val:.3f}")
        ax.axvline(median_val, color="#2ca02c", linestyle=":", linewidth=1.8,
                   label=f"중위수: {median_val:.3f}")
        ax.legend(fontsize=10)

        # 통계 정보 박스
        skew_val = series.skew()
        kurt_val = series.kurtosis()
        stats_text = (
            f"N = {len(series):,}\n"
            f"평균 = {mean_val:.3f}\n"
            f"표준편차 = {series.std():.3f}\n"
            f"왜도 = {skew_val:.3f}\n"
            f"첨도 = {kurt_val:.3f}"
        )
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", alpha=0.9))

        self._apply_readability(ax, f"히스토그램: {variable}", variable,
                                "밀도" if normal_curve else "빈도")
        fig.tight_layout()
        return fig

    def plot_boxplot(
        self,
        data: pd.DataFrame,
        x_var: str,
        y_var: Optional[str] = None,
        by_group: bool = True,
    ) -> Figure:
        """상자 그림 (그룹별 선택).

        Parameters
        ----------
        data:     DataFrame
        x_var:    X 변수 (그룹 변수 또는 단일 수치형 변수)
        y_var:    Y 수치형 변수 (그룹화 시 필수)
        by_group: 그룹별 표시 여부
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            if by_group and y_var and y_var in data.columns:
                sns.boxplot(
                    data=data, x=x_var, y=y_var, ax=ax,
                    palette=self.COLOR_PALETTE,
                    width=0.5, linewidth=1.5,
                    flierprops=dict(marker="o", markersize=4, alpha=0.6),
                )
                # 개별 데이터 점 오버레이
                sns.stripplot(
                    data=data, x=x_var, y=y_var, ax=ax,
                    color="black", alpha=0.25, size=3, jitter=True,
                )
                title = f"상자 그림: {y_var} (by {x_var})"
                xlabel, ylabel = x_var, y_var
            else:
                sns.boxplot(
                    data=data, y=x_var, ax=ax,
                    color=self.COLOR_PALETTE[0], width=0.4, linewidth=1.5,
                    flierprops=dict(marker="o", markersize=5, alpha=0.6),
                )
                title = f"상자 그림: {x_var}"
                xlabel, ylabel = "", x_var

            self._apply_readability(ax, title, xlabel, ylabel)
            fig.tight_layout()
        except Exception as exc:
            logger.exception("plot_boxplot 오류")
            return self._make_error_figure(str(exc))

        return fig

    def plot_scatter(
        self,
        data: pd.DataFrame,
        x_var: str,
        y_var: str,
        color_var: Optional[str] = None,
        fit_line: bool = True,
    ) -> Figure:
        """산점도 (회귀선 + 상관계수 선택적 표시).

        Parameters
        ----------
        data:      DataFrame
        x_var:     X 변수 (수치형)
        y_var:     Y 변수 (수치형)
        color_var: 색상 그룹 변수 (범주형)
        fit_line:  최소제곱 회귀선 표시 여부
        """
        validation = self._validate_data(data, [x_var, y_var])
        if not validation["valid"]:
            return self._make_error_figure("\n".join(validation["errors"]))

        fig, ax = plt.subplots(figsize=(10, 7))

        try:
            if color_var and color_var in data.columns:
                groups = data[color_var].unique()
                for i, grp in enumerate(groups):
                    subset = data[data[color_var] == grp]
                    ax.scatter(
                        subset[x_var], subset[y_var],
                        color=self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)],
                        label=str(grp), alpha=0.7, s=50, edgecolors="white", linewidths=0.5,
                    )
                    if fit_line:
                        _x = subset[x_var].dropna()
                        _y = subset[y_var].dropna()
                        idx = _x.index.intersection(_y.index)
                        if len(idx) >= 2:
                            m, b = np.polyfit(_x[idx], _y[idx], 1)
                            xr = np.linspace(_x.min(), _x.max(), 100)
                            ax.plot(xr, m * xr + b,
                                    color=self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)],
                                    linewidth=1.5, alpha=0.8)
                ax.legend(title=color_var, fontsize=10)
            else:
                ax.scatter(
                    data[x_var], data[y_var],
                    color=self.COLOR_PALETTE[0], alpha=0.6, s=50,
                    edgecolors="white", linewidths=0.5,
                )
                if fit_line:
                    valid = data[[x_var, y_var]].dropna()
                    if len(valid) >= 2:
                        m, b = np.polyfit(valid[x_var], valid[y_var], 1)
                        xr = np.linspace(valid[x_var].min(), valid[x_var].max(), 100)
                        ax.plot(xr, m * xr + b, color="#d62728", linewidth=2.0,
                                label=f"회귀선: y={m:.3f}x+{b:.3f}")
                        ax.legend(fontsize=10)

            # 상관계수
            valid2 = data[[x_var, y_var]].dropna()
            if (pd.api.types.is_numeric_dtype(data[x_var])
                    and pd.api.types.is_numeric_dtype(data[y_var])
                    and len(valid2) >= 2):
                r, p = stats.pearsonr(valid2[x_var], valid2[y_var])
                p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
                ax.text(0.02, 0.98, f"r = {r:.3f}, {p_str}\nN = {len(valid2):,}",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeeba", alpha=0.9))

            self._apply_readability(ax, f"산점도: {x_var} vs {y_var}", x_var, y_var)
            fig.tight_layout()
        except Exception as exc:
            logger.exception("plot_scatter 오류")
            return self._make_error_figure(str(exc))

        return fig

    def plot_bar(
        self,
        data: pd.DataFrame,
        x_var: str,
        y_var: Optional[str] = None,
        error_bars: bool = True,
    ) -> Figure:
        """막대 그래프 (오차 막대 선택적 표시).

        Parameters
        ----------
        data:       DataFrame
        x_var:      범주형 X 변수
        y_var:      수치형 Y 변수 (None 이면 빈도 표시)
        error_bars: 95% 신뢰구간 오차 막대 표시 여부
        """
        validation = self._validate_data(data, [x_var] + ([y_var] if y_var else []))
        if not validation["valid"]:
            return self._make_error_figure("\n".join(validation["errors"]))

        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            if y_var and y_var in data.columns:
                ci = 95 if error_bars else None
                sns.barplot(
                    data=data, x=x_var, y=y_var, ax=ax,
                    palette=self.COLOR_PALETTE, errorbar=("ci", ci) if ci else None,
                    capsize=0.08,
                )
                ylabel = y_var
                title = f"막대 그래프: {y_var} by {x_var}"
            else:
                counts = data[x_var].value_counts()
                bars = ax.bar(
                    range(len(counts)), counts.values,
                    color=self.COLOR_PALETTE[:len(counts)], edgecolor="white",
                )
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(counts.index, rotation=45, ha="right")
                # 빈도값 레이블
                for bar, val in zip(bars, counts.values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                        str(val), ha="center", va="bottom", fontsize=9,
                    )
                ylabel = "빈도"
                title = f"막대 그래프: {x_var}"

            self._apply_readability(ax, title, x_var, ylabel)
            fig.tight_layout()
        except Exception as exc:
            logger.exception("plot_bar 오류")
            return self._make_error_figure(str(exc))

        return fig

    def plot_line(
        self,
        data: pd.DataFrame,
        x_var: str,
        y_var: str,
        by_group: Optional[str] = None,
    ) -> Figure:
        """선 그래프 (그룹별 선택).

        Parameters
        ----------
        data:     DataFrame
        x_var:    X 변수 (순서형 또는 수치형)
        y_var:    Y 수치형 변수
        by_group: 그룹 구분 변수
        """
        validation = self._validate_data(data, [x_var, y_var])
        if not validation["valid"]:
            return self._make_error_figure("\n".join(validation["errors"]))

        fig, ax = plt.subplots(figsize=(11, 6))

        try:
            if by_group and by_group in data.columns:
                for i, (name, grp) in enumerate(data.groupby(by_group)):
                    grp_sorted = grp.sort_values(x_var)
                    ax.plot(
                        grp_sorted[x_var], grp_sorted[y_var],
                        marker="o", markersize=5,
                        color=self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)],
                        linewidth=2, label=str(name),
                    )
                ax.legend(title=by_group, fontsize=10)
            else:
                df_sorted = data.sort_values(x_var)
                ax.plot(
                    df_sorted[x_var], df_sorted[y_var],
                    marker="o", markersize=5,
                    color=self.COLOR_PALETTE[0], linewidth=2,
                )

            self._apply_readability(
                ax, f"선 그래프: {y_var} by {x_var}", x_var, y_var
            )
            fig.tight_layout()
        except Exception as exc:
            logger.exception("plot_line 오류")
            return self._make_error_figure(str(exc))

        return fig

    def plot_qq(self, data: pd.DataFrame, variable: str) -> Figure:
        """Q-Q 정규성 플롯 (Quantile-Quantile Plot).

        Parameters
        ----------
        data:     DataFrame
        variable: 검증할 수치형 변수
        """
        validation = self._validate_data(data, [variable])
        if not validation["valid"]:
            return self._make_error_figure("\n".join(validation["errors"]))

        num_check = self._validate_numeric(data[variable], variable)
        if not num_check["valid"]:
            return self._make_error_figure("\n".join(num_check["errors"]))

        series = data[variable].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Q-Q 플롯
        ax_qq = axes[0]
        osm, osr = stats.probplot(series, dist="norm")
        ax_qq.scatter(osm[0], osm[1], color=self.COLOR_PALETTE[0],
                      alpha=0.7, s=30, edgecolors="white", linewidths=0.4)
        # 기준선
        min_x, max_x = osm[0].min(), osm[0].max()
        slope, intercept = osr[0], osr[1]
        ax_qq.plot([min_x, max_x],
                   [slope * min_x + intercept, slope * max_x + intercept],
                   color="#d62728", linewidth=2.0, label="이론적 정규선")
        ax_qq.legend(fontsize=10)
        self._apply_readability(ax_qq, f"Q-Q 플롯: {variable}", "이론적 분위수", "관측 분위수")

        # 정규성 검정 결과 표시
        stat_sw, p_sw = stats.shapiro(series[:5000])  # Shapiro 최대 5000
        stat_ks, p_ks = stats.kstest(series, "norm",
                                      args=(series.mean(), series.std()))
        info_text = (
            f"Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f}\n"
            f"Kolmogorov-Smirnov: D={stat_ks:.4f}, p={p_ks:.4f}\n"
            f"왜도: {series.skew():.4f}\n"
            f"첨도: {series.kurtosis():.4f}"
        )
        ax_qq.text(0.03, 0.97, info_text, transform=ax_qq.transAxes,
                   fontsize=9, verticalalignment="top",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", alpha=0.9))

        # 히스토그램 + 정규 곡선 (보조)
        ax_hist = axes[1]
        ax_hist.hist(series, bins=25, density=True, color=self.COLOR_PALETTE[0],
                     alpha=0.7, edgecolor="white")
        x_curve = np.linspace(series.min(), series.max(), 300)
        ax_hist.plot(x_curve, stats.norm.pdf(x_curve, series.mean(), series.std()),
                     color="#d62728", linewidth=2.0, label="정규 분포 곡선")
        ax_hist.legend(fontsize=10)
        self._apply_readability(ax_hist, f"분포: {variable}", variable, "밀도")

        fig.suptitle(f"정규성 검정: {variable}", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig

    def plot_correlation_heatmap(
        self,
        data: pd.DataFrame,
        variables: List[str],
    ) -> Figure:
        """상관관계 히트맵.

        Parameters
        ----------
        data:      DataFrame
        variables: 포함할 변수 목록 (수치형만 유효)
        """
        numeric_cols = [c for c in variables if c in data.columns
                        and pd.api.types.is_numeric_dtype(data[c])]
        if len(numeric_cols) < 2:
            return self._make_error_figure("히트맵에는 숫자형 변수가 2개 이상 필요합니다.")

        corr = data[numeric_cols].corr()
        p_values = pd.DataFrame(np.ones_like(corr.values),
                                 index=corr.index, columns=corr.columns)
        for i, c1 in enumerate(numeric_cols):
            for j, c2 in enumerate(numeric_cols):
                if i != j:
                    valid = data[[c1, c2]].dropna()
                    if len(valid) >= 3:
                        _, p = stats.pearsonr(valid[c1], valid[c2])
                        p_values.loc[c1, c2] = p

        n = len(numeric_cols)
        fig_size = max(8, n * 0.9)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        mask = np.triu(np.ones_like(corr, dtype=bool))
        hm = sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax,
            square=True, linewidths=0.6,
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            annot_kws={"size": 9},
        )

        # 유의미한 상관에 별표 추가
        for i in range(n):
            for j in range(i):
                p = p_values.iloc[i, j]
                star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                if star:
                    ax.text(j + 0.5, i + 0.75, star,
                            ha="center", va="center", color="black", fontsize=8,
                            fontweight="bold")

        ax.set_title("상관관계 히트맵\n* p<0.05  ** p<0.01  *** p<0.001",
                     fontsize=13, fontweight="bold", pad=15)
        fig.tight_layout()
        return fig

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
    ) -> Figure:
        """ROC 곡선.

        Parameters
        ----------
        fpr:       False Positive Rate 배열
        tpr:       True Positive Rate 배열
        auc_score: AUC 값
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        # ROC 곡선
        ax.plot(fpr, tpr, color=self.COLOR_PALETTE[0], linewidth=2.5,
                label=f"ROC 곡선 (AUC = {auc_score:.4f})")
        # 대각 기준선 (무작위 분류기)
        ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--",
                linewidth=1.5, label="무작위 분류기 (AUC = 0.5)")
        # 이상적 지점 표시
        ax.scatter([0], [1], color="#d62728", zorder=5, s=80, label="이상적 지점")

        ax.fill_between(fpr, tpr, alpha=0.15, color=self.COLOR_PALETTE[0])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", fontsize=11)
        self._apply_readability(ax, "ROC 곡선 (Receiver Operating Characteristic)",
                                "위양성률 (1 - 특이도)", "민감도 (참양성률)")
        fig.tight_layout()
        return fig

    def plot_survival_curve(
        self,
        time: np.ndarray,
        survival_prob: np.ndarray,
        groups: Optional[dict] = None,
    ) -> Figure:
        """생존 분석 곡선 (Kaplan-Meier 스타일).

        Parameters
        ----------
        time:          생존 시간 배열
        survival_prob: 생존 확률 배열
        groups:        그룹별 데이터 {'그룹명': (time_array, prob_array)} 형태
        """
        fig, ax = plt.subplots(figsize=(11, 7))

        if groups:
            for i, (grp_name, (t, s)) in enumerate(groups.items()):
                sort_idx = np.argsort(t)
                ax.step(t[sort_idx], s[sort_idx],
                        where="post",
                        color=self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)],
                        linewidth=2.0, label=str(grp_name))
        else:
            sort_idx = np.argsort(time)
            ax.step(time[sort_idx], survival_prob[sort_idx],
                    where="post", color=self.COLOR_PALETTE[0],
                    linewidth=2.5, label="생존 곡선")

        ax.axhline(0.5, color="#7f7f7f", linestyle=":", linewidth=1.5,
                   label="중위 생존 (50%)")
        ax.set_ylim([0.0, 1.05])
        ax.legend(fontsize=11)
        self._apply_readability(ax, "생존 분석 곡선 (Kaplan-Meier)",
                                "시간", "생존 확률")
        fig.tight_layout()
        return fig

    def plot_residuals(
        self,
        fitted: np.ndarray,
        residuals: np.ndarray,
    ) -> Figure:
        """회귀 진단 플롯 (잔차 분석).

        Parameters
        ----------
        fitted:    예측값 배열
        residuals: 잔차 배열
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # 1. 잔차 vs 적합값
        ax1 = axes[0, 0]
        ax1.scatter(fitted, residuals, color=self.COLOR_PALETTE[0],
                    alpha=0.6, s=30, edgecolors="white", linewidths=0.4)
        ax1.axhline(0, color="#d62728", linewidth=1.5, linestyle="--")
        # Lowess 평활선
        from scipy.ndimage import uniform_filter1d
        sort_idx = np.argsort(fitted)
        smoothed = uniform_filter1d(residuals[sort_idx], size=max(3, len(residuals) // 20))
        ax1.plot(fitted[sort_idx], smoothed, color="#ff7f0e", linewidth=2.0, label="평활선")
        ax1.legend(fontsize=9)
        self._apply_readability(ax1, "잔차 vs 적합값", "적합값", "잔차")

        # 2. Q-Q 플롯 (잔차 정규성)
        ax2 = axes[0, 1]
        osm, osr = stats.probplot(residuals, dist="norm")
        ax2.scatter(osm[0], osm[1], color=self.COLOR_PALETTE[0],
                    alpha=0.7, s=30, edgecolors="white", linewidths=0.4)
        min_x, max_x = osm[0].min(), osm[0].max()
        ax2.plot([min_x, max_x],
                 [osr[0] * min_x + osr[1], osr[0] * max_x + osr[1]],
                 color="#d62728", linewidth=2.0)
        self._apply_readability(ax2, "정규 Q-Q 플롯 (잔차)", "이론적 분위수", "표준화 잔차")

        # 3. 척도-위치 플롯 (Scale-Location)
        ax3 = axes[1, 0]
        sqrt_abs_resid = np.sqrt(np.abs(residuals))
        ax3.scatter(fitted, sqrt_abs_resid,
                    color=self.COLOR_PALETTE[0], alpha=0.6, s=30,
                    edgecolors="white", linewidths=0.4)
        sort_idx2 = np.argsort(fitted)
        smoothed2 = uniform_filter1d(sqrt_abs_resid[sort_idx2],
                                      size=max(3, len(residuals) // 20))
        ax3.plot(fitted[sort_idx2], smoothed2, color="#ff7f0e", linewidth=2.0)
        self._apply_readability(ax3, "척도-위치 플롯", "적합값", "표준화 잔차의 제곱근")

        # 4. 잔차 히스토그램
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=25, density=True,
                 color=self.COLOR_PALETTE[0], alpha=0.75, edgecolor="white")
        x_curve = np.linspace(residuals.min(), residuals.max(), 300)
        ax4.plot(x_curve,
                 stats.norm.pdf(x_curve, residuals.mean(), residuals.std()),
                 color="#d62728", linewidth=2.0, label="정규 분포 곡선")
        ax4.legend(fontsize=9)
        self._apply_readability(ax4, "잔차 분포", "잔차", "밀도")

        fig.suptitle("회귀 진단 플롯", fontsize=15, fontweight="bold")
        fig.tight_layout()
        return fig

    def save_figure(self, fig: Figure, path: str, dpi: int = 150) -> None:
        """Figure를 파일로 저장.

        Parameters
        ----------
        fig:  matplotlib Figure
        path: 저장 경로 (확장자로 형식 결정: .png, .svg, .pdf)
        dpi:  해상도 (PNG 한정)
        """
        fmt = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
        fig.savefig(path, format=fmt, dpi=dpi if fmt == "png" else None,
                    bbox_inches="tight", facecolor="white")
        logger.info("Figure 저장 완료: %s", path)

    def fig_to_pixmap(self, fig: Figure):
        """matplotlib Figure를 PySide6 QPixmap으로 변환.

        Returns
        -------
        QPixmap
        """
        from PySide6.QtGui import QPixmap, QImage

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        image = QImage.fromData(buf.read())
        return QPixmap.fromImage(image)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _make_error_figure(self, message: str) -> Figure:
        """오류 Figure 생성 (Figure 객체 반환 버전)."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"차트 생성 오류:\n{message}",
                ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round", facecolor="#ffcccc", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.tight_layout()
        return fig

    # ── 유틸리티 ────────────────────────────────────────────────────────────

    def _apply_readability(self, ax, title: str, xlabel: str, ylabel: str) -> None:
        """가독성 요소 적용."""
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        
        # 그리드
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # 눈금 레이블 크기
        ax.tick_params(axis="both", labelsize=10)
        
        # 회전 (긴 레이블)
        if xlabel:
            x_labels = [str(l) for l in ax.get_xticklabels()]
            max_len = max((len(l) for l in x_labels), default=0)
            if max_len > 8:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    def _fig_to_base64(self, fig) -> str:
        """Figure를 Base64 PNG로 변환."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"
    
    def _error_image(self, message: str) -> str:
        """오류 메시지 이미지 생성."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"차트 생성 오류:\n{message}",
               ha="center", va="center", fontsize=12,
               bbox=dict(boxstyle="round", facecolor="#ffcccc", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        result = self._fig_to_base64(fig)
        plt.close(fig)
        return result

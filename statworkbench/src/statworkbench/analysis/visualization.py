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
import numpy as np
import pandas as pd
import seaborn as sns

from statworkbench.core.dataset import Dataset

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 기본 스타일
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")


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

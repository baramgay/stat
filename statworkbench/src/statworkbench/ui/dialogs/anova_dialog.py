"""ANOVA Dialog — 분산분석 다이얼로그.

일원/이원 분산분석을 수행합니다.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QRadioButton, QButtonGroup,
    QMessageBox, QTextEdit, QCheckBox
)
from PySide6.QtCore import Signal
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats

from statworkbench.core.dataset import Dataset


class ANOVADialog(QDialog):
    """분산분석 다이얼로그."""
    
    analysis_completed = Signal(dict)
    
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        
        self.setWindowTitle("📊 분산분석 (ANOVA)")
        self.setMinimumSize(500, 450)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # 분석 유형
        type_group = QGroupBox("📈 분석 유형")
        type_layout = QVBoxLayout(type_group)
        
        self.type_group = QButtonGroup(self)
        
        self.oneway_radio = QRadioButton("일원 분산분석 (One-Way ANOVA)")
        self.oneway_radio.setChecked(True)
        self.type_group.addButton(self.oneway_radio)
        type_layout.addWidget(self.oneway_radio)
        
        self.twoway_radio = QRadioButton("이원 분산분석 (Two-Way ANOVA)")
        self.type_group.addButton(self.twoway_radio)
        type_layout.addWidget(self.twoway_radio)
        
        layout.addWidget(type_group)
        
        # 변수 선택
        vars_group = QGroupBox("🔢 변수 선택")
        vars_layout = QVBoxLayout(vars_group)
        
        # 종속 변수
        dep_layout = QHBoxLayout()
        dep_layout.addWidget(QLabel("종속 변수:"))
        self.dep_combo = QComboBox()
        numeric_cols = self.dataset.data.select_dtypes(include=[np.number]).columns
        self.dep_combo.addItems(numeric_cols)
        dep_layout.addWidget(self.dep_combo)
        vars_layout.addLayout(dep_layout)
        
        # 독립 변수 1
        ind1_layout = QHBoxLayout()
        ind1_layout.addWidget(QLabel("독립 변수 1:"))
        self.ind1_combo = QComboBox()
        self.ind1_combo.addItems(self.dataset.data.columns)
        ind1_layout.addWidget(self.ind1_combo)
        vars_layout.addLayout(ind1_layout)
        
        # 독립 변수 2 (이원 ANOVA)
        ind2_layout = QHBoxLayout()
        ind2_layout.addWidget(QLabel("독립 변수 2:"))
        self.ind2_combo = QComboBox()
        self.ind2_combo.addItem("(없음)")
        self.ind2_combo.addItems(self.dataset.data.columns)
        ind2_layout.addWidget(self.ind2_combo)
        vars_layout.addLayout(ind2_layout)
        
        layout.addWidget(vars_group)
        
        # 사후 검정 (Post-hoc)
        post_hoc_group = QGroupBox("사후 검정 (Post-hoc)")
        post_hoc_layout = QVBoxLayout(post_hoc_group)

        self.chk_tukey = QCheckBox("Tukey HSD")
        self.chk_tukey.setChecked(True)
        post_hoc_layout.addWidget(self.chk_tukey)

        self.chk_bonferroni = QCheckBox("Bonferroni")
        post_hoc_layout.addWidget(self.chk_bonferroni)

        self.chk_scheffe = QCheckBox("Scheffe")
        post_hoc_layout.addWidget(self.chk_scheffe)

        layout.addWidget(post_hoc_group)

        # 옵션
        options_group = QGroupBox("⚙️ 옵션")
        options_layout = QVBoxLayout(options_group)

        self.levene_check = QCheckBox("등분산성 검정 (Levene)")
        self.levene_check.setChecked(True)
        options_layout.addWidget(self.levene_check)

        layout.addWidget(options_group)
        
        # 결과
        result_group = QGroupBox("📊 결과")
        result_layout = QVBoxLayout(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet(
            "background-color: #1a1a2e; color: #e8e8f0; "
            "font-family: Consolas; font-size: 11px;"
        )
        result_layout.addWidget(self.result_text)
        
        layout.addWidget(result_group)
        
        # 실행 버튼
        action_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("▶ 분석 실행")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_run.clicked.connect(self._run_analysis)
        action_layout.addWidget(self.btn_run)
        
        self.btn_close = QPushButton("❌ 닫기")
        self.btn_close.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_close)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
    
    def _run_analysis(self) -> None:
        """분산분석 실행."""
        dep_var = self.dep_combo.currentText()
        ind1_var = self.ind1_combo.currentText()
        ind2_var = self.ind2_combo.currentText()
        
        if ind2_var == "(없음)":
            ind2_var = None
        
        df = self.dataset.data.dropna(subset=[dep_var, ind1_var])
        
        try:
            result_lines = []
            result_lines.append("=" * 60)
            result_lines.append("분산분석 (ANOVA)")
            result_lines.append("=" * 60)
            result_lines.append(f"종속 변수: {dep_var}")
            result_lines.append(f"독립 변수 1: {ind1_var}")
            if ind2_var:
                result_lines.append(f"독립 변수 2: {ind2_var}")
            result_lines.append(f"유효 케이스: {len(df)}")
            result_lines.append("")
            
            # 등분산성 검정
            if self.levene_check.isChecked():
                groups = [group[dep_var].values for name, group in df.groupby(ind1_var)]
                if len(groups) >= 2:
                    levene_stat, levene_p = stats.levene(*groups)
                    result_lines.append("[등분산성 검정 - Levene]")
                    result_lines.append(f"  통계량: {levene_stat:.4f}")
                    result_lines.append(f"  p-value: {levene_p:.4f}")
                    if levene_p < 0.05:
                        result_lines.append("  → 등분산 가정 위반 (p < 0.05)")
                    else:
                        result_lines.append("  → 등분산 가정 충족")
                    result_lines.append("")
            
            # ANOVA
            if ind2_var is None or not self.twoway_radio.isChecked():
                # 일원 ANOVA
                groups = [group[dep_var].values for name, group in df.groupby(ind1_var)]
                f_stat, p_value = stats.f_oneway(*groups)
                
                result_lines.append("[일원 분산분석 결과]")
                result_lines.append(f"  F 통계량: {f_stat:.4f}")
                result_lines.append(f"  p-value: {p_value:.4f}")
                result_lines.append(f"  유의수준: {'유의함 (p < 0.05)' if p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}")
                result_lines.append("")
                
                # 기술통계
                result_lines.append("[그룹별 기술통계]")
                desc = df.groupby(ind1_var)[dep_var].agg(['count', 'mean', 'std', 'min', 'max'])
                for idx, row in desc.iterrows():
                    result_lines.append(f"  {idx}: N={row['count']:.0f}, M={row['mean']:.2f}, SD={row['std']:.2f}")
                result_lines.append("")
                
                # 사후 검정
                groups_data = [group[dep_var].values for name, group in df.groupby(ind1_var)]
                group_names = [name for name, group in df.groupby(ind1_var)]

                if self.chk_tukey.isChecked() and p_value < 0.05:
                    try:
                        from scipy.stats import tukey_hsd
                        result_lines.append("[사후 검정 - Tukey HSD]")
                        if len(groups_data) >= 2:
                            tukey_result = tukey_hsd(*groups_data)
                            result_lines.append("  p-value matrix:")
                            for i in range(len(group_names)):
                                for j in range(i + 1, len(group_names)):
                                    pval = tukey_result.pvalue[i, j]
                                    sig = "*" if pval < 0.05 else ""
                                    result_lines.append(
                                        f"    {group_names[i]} vs {group_names[j]}: p={pval:.4f} {sig}"
                                    )
                        result_lines.append("")
                    except ImportError:
                        result_lines.append("[사후 검정 - Tukey HSD]")
                        result_lines.append("  Tukey HSD를 위한 scipy 버전이 필요합니다.")
                        result_lines.append("")

                if self.chk_bonferroni.isChecked() and p_value < 0.05:
                    try:
                        from itertools import combinations
                        from scipy.stats import ttest_ind
                        n_pairs = len(group_names) * (len(group_names) - 1) // 2
                        result_lines.append("[사후 검정 - Bonferroni]")
                        for (i, gi), (j, gj) in combinations(enumerate(group_names), 2):
                            _, raw_p = ttest_ind(groups_data[i], groups_data[j])
                            bonf_p = min(raw_p * n_pairs, 1.0)
                            sig = "*" if bonf_p < 0.05 else ""
                            result_lines.append(
                                f"    {gi} vs {gj}: p(adjusted)={bonf_p:.4f} {sig}"
                            )
                        result_lines.append("")
                    except Exception as exc_b:
                        result_lines.append(f"  Bonferroni 계산 실패: {exc_b}")
                        result_lines.append("")

                if self.chk_scheffe.isChecked() and p_value < 0.05:
                    try:
                        from itertools import combinations
                        result_lines.append("[사후 검정 - Scheffe]")
                        k = len(groups_data)
                        grand_n = sum(len(g) for g in groups_data)
                        # Pooled within-group variance
                        ss_within = sum(
                            np.sum((g - np.mean(g)) ** 2) for g in groups_data
                        )
                        df_within = grand_n - k
                        ms_within = ss_within / df_within if df_within > 0 else np.nan
                        for (i, gi), (j, gj) in combinations(enumerate(group_names), 2):
                            ni, nj = len(groups_data[i]), len(groups_data[j])
                            diff = np.mean(groups_data[i]) - np.mean(groups_data[j])
                            se = np.sqrt(ms_within * (1 / ni + 1 / nj)) if ms_within and ms_within > 0 else np.nan
                            if se and not np.isnan(se):
                                f_scheffe = (diff / se) ** 2 / (k - 1)
                                from scipy.stats import f as f_dist
                                scheffe_p = 1 - f_dist.cdf(f_scheffe, k - 1, df_within)
                            else:
                                scheffe_p = np.nan
                            sig = "*" if (not np.isnan(scheffe_p) and scheffe_p < 0.05) else ""
                            pval_str = f"{scheffe_p:.4f}" if not np.isnan(scheffe_p) else "nan"
                            result_lines.append(
                                f"    {gi} vs {gj}: p={pval_str} {sig}"
                            )
                        result_lines.append("")
                    except Exception as exc_s:
                        result_lines.append(f"  Scheffe 계산 실패: {exc_s}")
                        result_lines.append("")
            
            else:
                # 이원 ANOVA (간단한 구현)
                result_lines.append("[이원 분산분석 결과]")
                result_lines.append("  (statsmodels 필요)")
                result_lines.append("")
            
            result_text = "\n".join(result_lines)
            self.result_text.setText(result_text)
            
            # 시그널 발생
            self.analysis_completed.emit({
                "type": "anova",
                "dependent": dep_var,
                "independent": [ind1_var, ind2_var] if ind2_var else [ind1_var],
                "post_hoc": self._selected_post_hoc(),
                "result": result_text,
            })

        except Exception as exc:
            self.result_text.setText(f"[오류]\n{exc}")

    def _selected_post_hoc(self) -> list[str]:
        """Return list of selected post-hoc method keys."""
        selected = []
        if self.chk_tukey.isChecked():
            selected.append("tukey")
        if self.chk_bonferroni.isChecked():
            selected.append("bonferroni")
        if self.chk_scheffe.isChecked():
            selected.append("scheffe")
        return selected

    def get_spec(self) -> dict:
        """Return analysis specification dict including post-hoc selection.

        Returns:
            dict with keys: dependent, independent, post_hoc, levene.
        """
        dep_var = self.dep_combo.currentText()
        ind1_var = self.ind1_combo.currentText()
        ind2_var = self.ind2_combo.currentText()
        if ind2_var == "(없음)":
            ind2_var = None

        return {
            "type": "two_way" if (ind2_var and self.twoway_radio.isChecked()) else "one_way",
            "variables": {
                "dependent": dep_var,
                "independent": [ind1_var, ind2_var] if ind2_var else [ind1_var],
            },
            "options": {
                "post_hoc": self._selected_post_hoc(),
                "levene": self.levene_check.isChecked(),
            },
        }

"""Nonparametric Tests Dialog — 비모수 검정 다이얼로그.

Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Chi-square 검정을 수행합니다.
"""

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.ui.analysis_worker import run_analysis_async


class NonparametricDialog(QDialog):
    """비모수 검정 다이얼로그."""

    analysis_completed = Signal(dict)

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self._analysis_worker = None

        self.setWindowTitle("🧪 비모수 검정")
        self.setMinimumSize(500, 450)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 검정 유형
        type_group = QGroupBox("📈 검정 유형")
        type_layout = QVBoxLayout(type_group)

        self.type_group = QButtonGroup(self)

        self.mannwhitney_radio = QRadioButton("Mann-Whitney U (독립표본)")
        self.mannwhitney_radio.setChecked(True)
        self.type_group.addButton(self.mannwhitney_radio)
        type_layout.addWidget(self.mannwhitney_radio)

        self.wilcoxon_radio = QRadioButton("Wilcoxon 부호순위 (대응표본)")
        self.type_group.addButton(self.wilcoxon_radio)
        type_layout.addWidget(self.wilcoxon_radio)

        self.kruskal_radio = QRadioButton("Kruskal-Wallis H (3+ 그룹)")
        self.type_group.addButton(self.kruskal_radio)
        type_layout.addWidget(self.kruskal_radio)

        self.chisquare_radio = QRadioButton("Chi-square 적합도/독립성")
        self.type_group.addButton(self.chisquare_radio)
        type_layout.addWidget(self.chisquare_radio)

        layout.addWidget(type_group)

        # 변수 선택
        vars_group = QGroupBox("🔢 변수 선택")
        vars_layout = QVBoxLayout(vars_group)

        # 변수 목록 구성 (variables 메타데이터 우선)
        all_vars = (
            list(self.dataset.variables.keys())
            if self.dataset.variables
            else list(self.dataset.data.columns)
        )

        # 검정 변수
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("검정 변수:"))
        self.test_combo = QComboBox()
        self.test_combo.addItems(all_vars)
        test_layout.addWidget(self.test_combo)
        vars_layout.addLayout(test_layout)

        # 그룹 변수
        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel("그룹 변수:"))
        self.group_combo = QComboBox()
        self.group_combo.addItem("(없음)")
        self.group_combo.addItems(all_vars)
        group_layout.addWidget(self.group_combo)
        vars_layout.addLayout(group_layout)

        layout.addWidget(vars_group)

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

        self.btn_run = QPushButton("▶ 검정 실행")
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

    def _current_test_type(self) -> str:
        if self.mannwhitney_radio.isChecked():
            return "mannwhitney"
        if self.wilcoxon_radio.isChecked():
            return "wilcoxon"
        if self.kruskal_radio.isChecked():
            return "kruskal"
        return "chisquare"

    def _run_analysis(self) -> None:
        """비모수 검정 실행 (P2-1: 백그라운드 스레드에서 실행, GUI 스레드 블로킹 없음)."""
        test_var = self.test_combo.currentText()
        group_var = self.group_combo.currentText()
        if group_var == "(없음)":
            group_var = None

        test_type = self._current_test_type()

        if test_type in ("mannwhitney", "kruskal") and group_var is None:
            QMessageBox.warning(self, "경고", "그룹 변수를 선택하세요")
            return

        if test_type == "mannwhitney":
            df = self.dataset.data.dropna(subset=[test_var])
            groups = df[group_var].unique()
            if len(groups) != 2:
                QMessageBox.warning(self, "경고", "그룹 변수는 2개의 범주를 가져야 합니다")
                return

        spec = {"test_type": test_type, "test_var": test_var, "group_var": group_var}

        self.btn_run.setEnabled(False)
        self.btn_run.setText("실행 중...")

        run_analysis_async(
            owner=self,
            run_fn=_compute_nonparametric_result,
            dataset=self.dataset,
            spec=spec,
            on_result=self._on_analysis_result,
            on_error=self._on_analysis_error,
        )

    def _on_analysis_result(self, result_text: str) -> None:
        self.result_text.setText(result_text)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ 검정 실행")

        self.analysis_completed.emit({
            "type": "nonparametric",
            "test": self.type_group.checkedButton().text(),
            "result": result_text,
        })

    def _on_analysis_error(self, message: str) -> None:
        self.result_text.setText(f"[오류]\n{message}")
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ 검정 실행")


def _compute_nonparametric_result(dataset: Dataset, spec: dict) -> str:
    """비모수 검정 계산 (백그라운드 스레드에서 실행, GUI 위젯 접근 없음)."""
    test_var = spec["test_var"]
    group_var = spec["group_var"]
    test_type = spec["test_type"]

    df = dataset.data.dropna(subset=[test_var])
    result_lines = ["=" * 60]

    if test_type == "mannwhitney":
        result_lines.append("Mann-Whitney U 검정")
        result_lines.append("=" * 60)

        groups = df[group_var].unique()
        group1 = df[df[group_var] == groups[0]][test_var]
        group2 = df[df[group_var] == groups[1]][test_var]

        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        result_lines.append(f"그룹 1 ({groups[0]}): N={len(group1)}, 중위수={group1.median():.2f}")
        result_lines.append(f"그룹 2 ({groups[1]}): N={len(group2)}, 중위수={group2.median():.2f}")
        result_lines.append("")
        result_lines.append(f"U 통계량: {statistic:.4f}")
        result_lines.append(f"p-value: {p_value:.4f}")
        result_lines.append(f"결과: {'유의함 (p < 0.05)' if p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}")

    elif test_type == "wilcoxon":
        result_lines.append("Wilcoxon 부호순위 검정")
        result_lines.append("=" * 60)
        result_lines.append("(대응표본: 2개 변수 선택 필요)")

    elif test_type == "kruskal":
        result_lines.append("Kruskal-Wallis H 검정")
        result_lines.append("=" * 60)

        groups = [group[test_var].values for name, group in df.groupby(group_var)]
        statistic, p_value = stats.kruskal(*groups)

        result_lines.append(f"그룹 수: {len(groups)}")
        result_lines.append(f"H 통계량: {statistic:.4f}")
        result_lines.append(f"p-value: {p_value:.4f}")
        result_lines.append(f"결과: {'유의함 (p < 0.05)' if p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}")

    elif test_type == "chisquare":
        result_lines.append("Chi-square 검정")
        result_lines.append("=" * 60)

        if group_var:
            contingency = pd.crosstab(df[test_var], df[group_var])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            result_lines.append("[독립성 검정]")
            result_lines.append(f"카이제곱: {chi2:.4f}")
            result_lines.append(f"자유도: {dof}")
            result_lines.append(f"p-value: {p_value:.4f}")
        else:
            observed = df[test_var].value_counts()
            chi2, p_value = stats.chisquare(observed)

            result_lines.append("[적합도 검정]")
            result_lines.append(f"카이제곱: {chi2:.4f}")
            result_lines.append(f"p-value: {p_value:.4f}")

        result_lines.append(f"결과: {'유의함 (p < 0.05)' if p_value < 0.05 else '유의하지 않음 (p >= 0.05)'}")

    result_lines.append("")
    result_lines.append(f"유효 케이스: {len(df)}")

    return "\n".join(result_lines)

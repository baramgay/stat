"""Descriptives Dialog — SPSS 스타일 기술통계 다이얼로그."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._dialog_helpers import (
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
    var_from_display,
)


class DescriptivesDialog(QDialog):
    """SPSS Descriptives 다이얼로그.

    척도(Scale) 변수만 목록에 표시 — SPSS 동일 동작.
    메타데이터 없는 경우 수치형 dtype 으로 대체.
    """

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("기술통계량")
        self.setMinimumSize(500, 430)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 변수 목록 (척도 변수만)
        var_group = QGroupBox("변수 (척도형만 표시됩니다)")
        var_layout = QVBoxLayout(var_group)

        self.var_list = QListWidget()
        self.var_list.setSelectionMode(QListWidget.ExtendedSelection)

        candidates = scale_vars(self._dataset) or numeric_vars(self._dataset)
        for var in candidates:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            item = QListWidgetItem(f"{icon} {label}" if icon else label)
            item.setData(0x0100, var)  # UserRole — 실제 변수명 저장
            self.var_list.addItem(item)

        var_layout.addWidget(self.var_list)

        hint = QLabel("💡 Ctrl+클릭으로 여러 변수 선택")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        var_layout.addWidget(hint)
        layout.addWidget(var_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_mean = QCheckBox("평균 (Mean)")
        self.chk_mean.setChecked(True)
        opt_layout.addWidget(self.chk_mean)

        self.chk_std = QCheckBox("표준편차 (Std. Deviation)")
        self.chk_std.setChecked(True)
        opt_layout.addWidget(self.chk_std)

        self.chk_minmax = QCheckBox("최소/최대 (Min/Max)")
        self.chk_minmax.setChecked(True)
        opt_layout.addWidget(self.chk_minmax)

        self.chk_skew = QCheckBox("왜도/첨도 (Skewness/Kurtosis)")
        self.chk_skew.setChecked(False)
        opt_layout.addWidget(self.chk_skew)

        self.chk_ci = QCheckBox("95% 신뢰구간 (CI for Mean)")
        self.chk_ci.setChecked(False)
        opt_layout.addWidget(self.chk_ci)

        layout.addWidget(opt_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        selected_items = self.var_list.selectedItems()
        if not selected_items:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 하나 이상 선택하세요.")
            return

        # UserRole 에 저장된 실제 변수명 사용
        selected = [item.data(0x0100) or var_from_display(item.text().split(" ", 1)[-1])
                    for item in selected_items]

        try:
            from nuristat.analysis.descriptive import run_analysis
            spec = {
                "variables": {"scale": selected},
                "options": {
                    "show_mean": self.chk_mean.isChecked(),
                    "show_std": self.chk_std.isChecked(),
                    "show_minmax": self.chk_minmax.isChecked(),
                    "show_skew": self.chk_skew.isChecked(),
                    "show_ci": self.chk_ci.isChecked(),
                },
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")

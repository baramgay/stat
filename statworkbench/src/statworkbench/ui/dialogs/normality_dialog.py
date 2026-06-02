"""정규성 검정 (Shapiro-Wilk) 대화상자."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
)


class NormalityDialog(QDialog):
    """Shapiro-Wilk / Kolmogorov-Smirnov 정규성 검정 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("정규성 검정 (Shapiro-Wilk)")
        self.setMinimumSize(460, 380)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        var_group = QGroupBox("변수 선택 (연속형 척도 변수)")
        var_layout = QHBoxLayout(var_group)

        left = QVBoxLayout()
        left.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        avail = scale_vars(self._dataset) or numeric_vars(self._dataset)
        for var in avail:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            self.avail_list.addItem(f"{icon} {label}" if icon else label)
        self.avail_list.setSelectionMode(QListWidget.ExtendedSelection)
        left.addWidget(self.avail_list)
        var_layout.addLayout(left)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_var)
        btn_layout.addWidget(add_btn)
        rem_btn = QPushButton("←")
        rem_btn.setFixedWidth(36)
        rem_btn.clicked.connect(self._remove_var)
        btn_layout.addWidget(rem_btn)
        btn_layout.addStretch()
        var_layout.addLayout(btn_layout)

        right = QVBoxLayout()
        right.addWidget(QLabel("선택된 변수:"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QListWidget.ExtendedSelection)
        right.addWidget(self.selected_list)
        var_layout.addLayout(right)

        layout.addWidget(var_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _add_var(self) -> None:
        already = {self.selected_list.item(i).text() for i in range(self.selected_list.count())}
        for item in self.avail_list.selectedItems():
            if item.text() not in already:
                self.selected_list.addItem(item.text())
                already.add(item.text())

    def _remove_var(self) -> None:
        for item in self.selected_list.selectedItems():
            self.selected_list.takeItem(self.selected_list.row(item))

    def _run(self) -> None:
        selected = [self.selected_list.item(i).text() for i in range(self.selected_list.count())]
        # 아이콘 접두사 제거하여 실제 변수명 추출
        from statworkbench.ui.dialogs._dialog_helpers import var_from_display
        var_names = [var_from_display(s) for s in selected]
        var_names = [v for v in var_names if v]

        if not var_names:
            QMessageBox.warning(self, "경고", "검정할 변수를 최소 1개 선택하세요.")
            return

        try:
            from statworkbench.analysis.normality import run_analysis
            spec = {
                "variables": {"target": var_names},
                "options": {},
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")

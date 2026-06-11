"""반복측정 ANOVA 대화상자."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._async_mixin import AnalysisDialogMixin
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    numeric_vars,
    scale_vars,
)


class RepeatedMeasuresDialog(QDialog, AnalysisDialogMixin):
    """SPSS General Linear Model > Repeated Measures 스타일 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("반복측정 분산분석 — Repeated Measures ANOVA")
        self.setMinimumSize(540, 540)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Within-Subjects 요인 이름
        name_group = QGroupBox("Within-Subjects 요인")
        name_layout = QHBoxLayout(name_group)
        name_layout.addWidget(QLabel("요인 레이블:"))
        self.factor_name_edit = QLineEdit("시점")
        name_layout.addWidget(self.factor_name_edit)
        layout.addWidget(name_group)

        # 측정 변수 선택
        var_group = QGroupBox("반복 측정 변수 (시점 순서대로 선택, 최소 2개)")
        var_layout = QHBoxLayout(var_group)

        # 왼쪽: 사용 가능한 변수
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        avail_vars = scale_vars(self._dataset) or numeric_vars(self._dataset) or all_vars(self._dataset)
        self.avail_list.addItems(avail_vars)
        left_layout.addWidget(self.avail_list)
        var_layout.addLayout(left_layout)

        # 버튼
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_var)
        btn_layout.addWidget(add_btn)
        remove_btn = QPushButton("←")
        remove_btn.setFixedWidth(36)
        remove_btn.clicked.connect(self._remove_var)
        btn_layout.addWidget(remove_btn)
        up_btn = QPushButton("↑")
        up_btn.setFixedWidth(36)
        up_btn.clicked.connect(self._move_up)
        btn_layout.addWidget(up_btn)
        down_btn = QPushButton("↓")
        down_btn.setFixedWidth(36)
        down_btn.clicked.connect(self._move_down)
        btn_layout.addWidget(down_btn)
        btn_layout.addStretch()
        var_layout.addLayout(btn_layout)

        # 오른쪽: 선택된 변수
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("선택된 변수 (반복 순서):"))
        self.selected_list = QListWidget()
        right_layout.addWidget(self.selected_list)
        var_layout.addLayout(right_layout)

        layout.addWidget(var_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_pairwise = QCheckBox("쌍 비교 (Bonferroni 보정)")
        self.chk_pairwise.setChecked(True)
        opt_layout.addWidget(self.chk_pairwise)

        ci_row = QHBoxLayout()
        ci_row.addWidget(QLabel("유의 수준 (α):"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 0.20)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setValue(0.05)
        self.alpha_spin.setDecimals(2)
        ci_row.addWidget(self.alpha_spin)
        ci_row.addStretch()
        opt_layout.addLayout(ci_row)
        layout.addWidget(opt_group)

        # 버튼
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
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

    def _move_up(self) -> None:
        row = self.selected_list.currentRow()
        if row > 0:
            item = self.selected_list.takeItem(row)
            self.selected_list.insertItem(row - 1, item)
            self.selected_list.setCurrentRow(row - 1)

    def _move_down(self) -> None:
        row = self.selected_list.currentRow()
        if row < self.selected_list.count() - 1:
            item = self.selected_list.takeItem(row)
            self.selected_list.insertItem(row + 1, item)
            self.selected_list.setCurrentRow(row + 1)

    def _run(self) -> None:
        measures = [
            self.selected_list.item(i).text()
            for i in range(self.selected_list.count())
        ]
        if len(measures) < 2:
            QMessageBox.warning(self, "경고", "반복 측정 변수를 최소 2개 선택하세요.")
            return

        within_name = self.factor_name_edit.text().strip() or "시점"

        from nuristat.analysis.repeated_measures_anova import run_analysis
        spec = {
            "variables": {"measures": measures},
            "options": {
                "within_name": within_name,
                "pairwise": self.chk_pairwise.isChecked(),
                "alpha": self.alpha_spin.value(),
            },
        }
        self._start_analysis(run_analysis, self._dataset, spec)

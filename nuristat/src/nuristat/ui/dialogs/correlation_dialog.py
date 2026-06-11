"""Correlation Dialog — SPSS 스타일 상관분석 다이얼로그."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._async_mixin import AnalysisDialogMixin
from nuristat.ui.dialogs._dialog_helpers import (
    display_label,
    measure_icon,
    numeric_vars,
    ordinal_or_higher_vars,
    scale_vars,
)


class CorrelationDialog(QDialog, AnalysisDialogMixin):
    """SPSS Bivariate Correlations 다이얼로그.

    Pearson: 척도(Scale) 변수 — analysis/correlation.py 모듈 사용
    Spearman/Kendall: 순서형 이상 변수
    """

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("이변량 상관분석")
        self.setMinimumSize(580, 520)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 상관계수 유형
        type_group = QGroupBox("상관계수 유형")
        type_layout = QHBoxLayout(type_group)

        self._method_grp = QButtonGroup(self)

        self.pearson_radio = QRadioButton("Pearson (척도형)")
        self.pearson_radio.setChecked(True)
        self._method_grp.addButton(self.pearson_radio, 0)
        type_layout.addWidget(self.pearson_radio)

        self.spearman_radio = QRadioButton("Spearman (순서형)")
        self._method_grp.addButton(self.spearman_radio, 1)
        type_layout.addWidget(self.spearman_radio)

        self.kendall_radio = QRadioButton("Kendall's tau-b")
        self._method_grp.addButton(self.kendall_radio, 2)
        type_layout.addWidget(self.kendall_radio)

        type_layout.addStretch()
        layout.addWidget(type_group)

        # 변수 선택 (이중 리스트)
        vars_group = QGroupBox("변수 선택")
        vars_layout = QHBoxLayout(vars_group)

        avail_layout = QVBoxLayout()
        avail_layout.addWidget(QLabel("사용 가능:"))
        self.avail_list = QListWidget()
        self.avail_list.setSelectionMode(QAbstractItemView.MultiSelection)
        avail_layout.addWidget(self.avail_list)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        self.btn_add = QPushButton("▶")
        self.btn_add.clicked.connect(self._add_vars)
        btn_layout.addWidget(self.btn_add)
        self.btn_remove = QPushButton("◀")
        self.btn_remove.clicked.connect(self._remove_vars)
        btn_layout.addWidget(self.btn_remove)
        self.btn_add_all = QPushButton("▶▶")
        self.btn_add_all.clicked.connect(self._add_all)
        btn_layout.addWidget(self.btn_add_all)
        self.btn_remove_all = QPushButton("◀◀")
        self.btn_remove_all.clicked.connect(self._remove_all)
        btn_layout.addWidget(self.btn_remove_all)
        btn_layout.addStretch()

        selected_layout = QVBoxLayout()
        selected_layout.addWidget(QLabel("선택됨:"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.MultiSelection)
        selected_layout.addWidget(self.selected_list)

        vars_layout.addLayout(avail_layout, 2)
        vars_layout.addLayout(btn_layout)
        vars_layout.addLayout(selected_layout, 2)
        layout.addWidget(vars_group)

        # 유의성 표시 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)
        self.flag_sig = QRadioButton("유의한 상관에 별표 표시 (* p<.05, ** p<.01)")
        self.flag_sig.setChecked(True)
        opt_layout.addWidget(self.flag_sig)
        self.no_flag = QRadioButton("별표 없이 상관계수만 표시")
        opt_layout.addWidget(self.no_flag)
        layout.addWidget(opt_group)

        # 버튼
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # 초기 변수 목록 채우기
        self._method_grp.buttonClicked.connect(self._refresh_avail)
        self._refresh_avail()

    # ── 변수 목록 ──────────────────────────────────────────────────────────

    def _get_candidate_vars(self) -> list[str]:
        if self.pearson_radio.isChecked():
            return scale_vars(self._dataset) or numeric_vars(self._dataset)
        else:
            return ordinal_or_higher_vars(self._dataset) or numeric_vars(self._dataset)

    def _refresh_avail(self, *_):
        """상관계수 유형 변경 시 사용 가능 변수 목록 갱신."""
        already_selected = self._selected_var_names()
        self.avail_list.clear()
        for var in self._get_candidate_vars():
            if var in already_selected:
                continue
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            text = f"{icon} {label}" if icon else label
            item = QListWidgetItem(text)
            item.setData(0x0100, var)
            self.avail_list.addItem(item)

    def _selected_var_names(self) -> list[str]:
        return [
            self.selected_list.item(i).data(0x0100)
            for i in range(self.selected_list.count())
        ]

    def _add_vars(self):
        for item in self.avail_list.selectedItems():
            new_item = QListWidgetItem(item.text())
            new_item.setData(0x0100, item.data(0x0100))
            self.selected_list.addItem(new_item)
            self.avail_list.takeItem(self.avail_list.row(item))

    def _remove_vars(self):
        for item in self.selected_list.selectedItems():
            new_item = QListWidgetItem(item.text())
            new_item.setData(0x0100, item.data(0x0100))
            self.avail_list.addItem(new_item)
            self.selected_list.takeItem(self.selected_list.row(item))

    def _add_all(self):
        while self.avail_list.count():
            item = self.avail_list.takeItem(0)
            self.selected_list.addItem(item)

    def _remove_all(self):
        while self.selected_list.count():
            item = self.selected_list.takeItem(0)
            self.avail_list.addItem(item)

    # ── 분석 실행 ──────────────────────────────────────────────────────────

    def _run(self):
        variables = self._selected_var_names()
        if len(variables) < 2:
            QMessageBox.warning(self, "경고", "변수를 2개 이상 선택하세요.")
            return

        if self.pearson_radio.isChecked():
            method = "pearson"
        elif self.spearman_radio.isChecked():
            method = "spearman"
        else:
            method = "kendall"

        from nuristat.analysis.correlation import run_analysis
        spec = {
            "variables": {"target": variables},
            "options": {
                "method": method,
                "flag_significant": self.flag_sig.isChecked(),
                "pairwise": True,
            },
        }
        self._start_analysis(run_analysis, self._dataset, spec)

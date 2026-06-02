"""주성분분석(PCA) 대화상자."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    all_vars,
    numeric_vars,
    populate_list_widget,
    scale_vars,
    user_friendly_error,
    var_from_display,
)


class PcaDialog(QDialog):
    """SPSS Analyze > Dimension Reduction > Factor (Principal Components) 대화상자."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("주성분분석 (PCA)")
        self.setMinimumSize(560, 560)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 변수 선택
        var_group = QGroupBox("분석 변수 선택 (척도형, 최소 2개)")
        var_layout = QHBoxLayout(var_group)

        left = QVBoxLayout()
        left.addWidget(QLabel("사용 가능한 변수:"))
        self.avail_list = QListWidget()
        self.avail_list.setSelectionMode(QListWidget.ExtendedSelection)
        avail = scale_vars(self._dataset) or numeric_vars(self._dataset) or all_vars(self._dataset)
        populate_list_widget(self.avail_list, self._dataset, avail,
                             "(척도형/수치형 변수 없음)")
        left.addWidget(self.avail_list)
        var_layout.addLayout(left)

        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("→")
        add_btn.setFixedWidth(36)
        add_btn.clicked.connect(self._add_vars)
        btn_layout.addWidget(add_btn)
        rem_btn = QPushButton("←")
        rem_btn.setFixedWidth(36)
        rem_btn.clicked.connect(self._remove_vars)
        btn_layout.addWidget(rem_btn)
        btn_layout.addStretch()
        var_layout.addLayout(btn_layout)

        right = QVBoxLayout()
        right.addWidget(QLabel("선택된 변수:"))
        self.sel_list = QListWidget()
        self.sel_list.setSelectionMode(QListWidget.ExtendedSelection)
        right.addWidget(self.sel_list)
        var_layout.addLayout(right)
        layout.addWidget(var_group)

        # 추출 옵션
        extract_group = QGroupBox("추출")
        extract_layout = QVBoxLayout(extract_group)

        n_comp_row = QHBoxLayout()
        n_comp_row.addWidget(QLabel("주성분 수 (0 = Kaiser 자동):"))
        self.n_comp_spin = QSpinBox()
        self.n_comp_spin.setRange(0, 50)
        self.n_comp_spin.setValue(0)
        self.n_comp_spin.setToolTip("0으로 설정하면 고유값 ≥ 1인 성분 수를 자동 결정합니다.")
        n_comp_row.addWidget(self.n_comp_spin)
        n_comp_row.addStretch()
        extract_layout.addLayout(n_comp_row)

        self.chk_standardize = QCheckBox("변수 표준화 (권장)")
        self.chk_standardize.setChecked(True)
        extract_layout.addWidget(self.chk_standardize)
        layout.addWidget(extract_group)

        # 회전 옵션
        rot_group = QGroupBox("회전 (Rotation)")
        rot_layout = QHBoxLayout(rot_group)
        rot_layout.addWidget(QLabel("회전 방법:"))
        self.rot_combo = QComboBox()
        self.rot_combo.addItem("Varimax (직교 회전)", "varimax")
        self.rot_combo.addItem("Promax (사각 회전)", "promax")
        self.rot_combo.addItem("없음 (No Rotation)", "none")
        rot_layout.addWidget(self.rot_combo)
        rot_layout.addStretch()
        layout.addWidget(rot_group)

        # 출력 옵션
        output_group = QGroupBox("출력")
        output_layout = QVBoxLayout(output_group)

        self.chk_kmo = QCheckBox("KMO 표본 적합도 및 Bartlett 구형성 검정")
        self.chk_kmo.setChecked(True)
        output_layout.addWidget(self.chk_kmo)

        self.chk_scree = QCheckBox("스크리 플롯 (Scree Plot)")
        self.chk_scree.setChecked(True)
        output_layout.addWidget(self.chk_scree)
        layout.addWidget(output_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _add_vars(self) -> None:
        already = {self.sel_list.item(i).text() for i in range(self.sel_list.count())}
        for item in self.avail_list.selectedItems():
            if item.text() not in already:
                self.sel_list.addItem(item.text())

    def _remove_vars(self) -> None:
        for item in self.sel_list.selectedItems():
            self.sel_list.takeItem(self.sel_list.row(item))

    def _run(self) -> None:
        sel_labels = [self.sel_list.item(i).text() for i in range(self.sel_list.count())]
        items = [var_from_display(s) or s for s in sel_labels]

        if len(items) < 2:
            QMessageBox.warning(self, "경고", "분석 변수를 2개 이상 선택하세요.")
            return

        try:
            from statworkbench.analysis.pca import run_analysis
            spec = {
                "variables": {"items": items},
                "options": {
                    "n_components": self.n_comp_spin.value(),
                    "rotation": self.rot_combo.currentData(),
                    "standardize": self.chk_standardize.isChecked(),
                    "scree_plot": self.chk_scree.isChecked(),
                    "kmo": self.chk_kmo.isChecked(),
                },
                "missing_policy": "listwise",
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "분석 오류", user_friendly_error(exc))

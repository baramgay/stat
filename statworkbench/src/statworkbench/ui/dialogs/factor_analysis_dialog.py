"""Factor Analysis Dialog — SPSS 스타일 요인분석 다이얼로그."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
)


class FactorAnalysisDialog(QDialog):
    """SPSS 스타일 요인분석(Factor Analysis) 다이얼로그.

    PCA, 최대우도, 주축인수 방법을 지원합니다.
    회전 방법: 없음, 베리맥스, 직접 오블리민
    추출 기준: 고유값 > 1 또는 요인수 직접 지정
    """

    analysis_requested = Signal(str, dict)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("요인분석")
        self.setMinimumSize(660, 580)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 분석 변수 선택
        var_group = QGroupBox("분석 변수 선택")
        var_layout = QHBoxLayout(var_group)

        # 사용 가능한 변수 목록
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("사용 가능한 변수:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        _vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        for var in _vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            item = QListWidgetItem(f"{icon} {label}" if icon else label)
            item.setData(0x0100, var)
            self.available_list.addItem(item)
        left_layout.addWidget(self.available_list)
        var_layout.addLayout(left_layout)

        # 이동 버튼
        move_layout = QVBoxLayout()
        move_layout.addStretch()
        btn_add = QPushButton("→")
        btn_add.setFixedWidth(36)
        btn_add.clicked.connect(self._add_vars)
        btn_remove = QPushButton("←")
        btn_remove.setFixedWidth(36)
        btn_remove.clicked.connect(self._remove_vars)
        btn_add_all = QPushButton(">>")
        btn_add_all.setFixedWidth(36)
        btn_add_all.clicked.connect(self._add_all_vars)
        btn_remove_all = QPushButton("<<")
        btn_remove_all.setFixedWidth(36)
        btn_remove_all.clicked.connect(self._remove_all_vars)
        move_layout.addWidget(btn_add_all)
        move_layout.addWidget(btn_add)
        move_layout.addWidget(btn_remove)
        move_layout.addWidget(btn_remove_all)
        move_layout.addStretch()
        var_layout.addLayout(move_layout)

        # 선택된 변수 목록
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("선택된 변수:"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        right_layout.addWidget(self.selected_list)
        var_layout.addLayout(right_layout)

        layout.addWidget(var_group)

        # 분석 방법 및 회전
        method_rotation_layout = QHBoxLayout()

        # 분석 방법
        method_group = QGroupBox("분석 방법 (Extraction)")
        method_layout = QVBoxLayout(method_group)
        self.method_combo = QComboBox()
        self.method_combo.addItem("주성분 분석 (PCA)", "pca")
        self.method_combo.addItem("최대우도 (Maximum Likelihood)", "ml")
        self.method_combo.addItem("주축인수 (Principal Axis Factoring)", "paf")
        method_layout.addWidget(self.method_combo)
        method_rotation_layout.addWidget(method_group)

        # 회전 방법
        rotation_group = QGroupBox("회전 (Rotation)")
        rotation_layout = QVBoxLayout(rotation_group)
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItem("베리맥스 (Varimax)", "varimax")
        self.rotation_combo.addItem("직접 오블리민 (Direct Oblimin)", "oblimin")
        self.rotation_combo.addItem("없음 (None)", "none")
        rotation_layout.addWidget(self.rotation_combo)
        method_rotation_layout.addWidget(rotation_group)

        layout.addLayout(method_rotation_layout)

        # 추출 기준
        extract_group = QGroupBox("추출 기준 (Extract)")
        extract_layout = QVBoxLayout(extract_group)

        self.extract_btn_group = QButtonGroup(self)

        eigenval_layout = QHBoxLayout()
        self.eigen_radio = QRadioButton("고유값 기준 (Eigenvalue >")
        self.eigen_radio.setChecked(True)
        self.extract_btn_group.addButton(self.eigen_radio)
        eigenval_layout.addWidget(self.eigen_radio)
        self.eigen_spin = QDoubleSpinBox()
        self.eigen_spin.setRange(0.1, 10.0)
        self.eigen_spin.setValue(1.0)
        self.eigen_spin.setSingleStep(0.1)
        eigenval_layout.addWidget(self.eigen_spin)
        eigenval_layout.addWidget(QLabel(")"))
        eigenval_layout.addStretch()
        extract_layout.addLayout(eigenval_layout)

        nfactor_layout = QHBoxLayout()
        self.nfactor_radio = QRadioButton("요인 수 지정:")
        self.extract_btn_group.addButton(self.nfactor_radio)
        nfactor_layout.addWidget(self.nfactor_radio)
        self.nfactor_spin = QSpinBox()
        self.nfactor_spin.setRange(1, 30)
        self.nfactor_spin.setValue(3)
        nfactor_layout.addWidget(self.nfactor_spin)
        nfactor_layout.addStretch()
        extract_layout.addLayout(nfactor_layout)

        layout.addWidget(extract_group)

        # 표시 옵션
        display_group = QGroupBox("표시 옵션")
        display_layout = QHBoxLayout(display_group)

        self.show_loadings = QCheckBox("요인 부하량 행렬")
        self.show_loadings.setChecked(True)
        display_layout.addWidget(self.show_loadings)

        self.show_communality = QCheckBox("공통성 (Communalities)")
        self.show_communality.setChecked(True)
        display_layout.addWidget(self.show_communality)

        self.show_scree = QCheckBox("스크리 플롯 (Scree Plot)")
        display_layout.addWidget(self.show_scree)

        layout.addWidget(display_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _clone_item(self, source: QListWidgetItem) -> QListWidgetItem:
        new = QListWidgetItem(source.text())
        new.setData(0x0100, source.data(0x0100))
        return new

    def _add_vars(self):
        already = {self.selected_list.item(i).data(0x0100) for i in range(self.selected_list.count())}
        for item in self.available_list.selectedItems():
            if item.data(0x0100) not in already:
                self.selected_list.addItem(self._clone_item(item))

    def _remove_vars(self):
        for item in self.selected_list.selectedItems():
            self.selected_list.takeItem(self.selected_list.row(item))

    def _add_all_vars(self):
        already = {self.selected_list.item(i).data(0x0100) for i in range(self.selected_list.count())}
        for i in range(self.available_list.count()):
            src = self.available_list.item(i)
            if src.data(0x0100) not in already:
                self.selected_list.addItem(self._clone_item(src))

    def _remove_all_vars(self):
        self.selected_list.clear()

    def get_spec(self) -> dict:
        """분석 스펙 반환."""
        variables = [
            self.selected_list.item(i).data(0x0100) or self.selected_list.item(i).text()
            for i in range(self.selected_list.count())
        ]
        return {
            "analysis_id": "factor_analysis",
            "variables": variables,
            "method": self.method_combo.currentData(),
            "rotation": self.rotation_combo.currentData(),
            "extraction": {
                "criterion": "eigenvalue" if self.eigen_radio.isChecked() else "n_factors",
                "eigenvalue_threshold": self.eigen_spin.value(),
                "n_factors": self.nfactor_spin.value(),
            },
            "display": {
                "factor_loadings": self.show_loadings.isChecked(),
                "communalities": self.show_communality.isChecked(),
                "scree_plot": self.show_scree.isChecked(),
            },
        }

    def _on_ok(self):
        variables = [
            self.selected_list.item(i).data(0x0100) or self.selected_list.item(i).text()
            for i in range(self.selected_list.count())
        ]
        if len(variables) < 2:
            QMessageBox.warning(self, "경고", "분석 변수를 2개 이상 선택하세요.")
            return

        spec = self.get_spec()
        self.analysis_requested.emit("factor_analysis", spec)
        self.accept()

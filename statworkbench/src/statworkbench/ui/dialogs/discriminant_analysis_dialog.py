"""Discriminant Analysis Dialog — SPSS 스타일 판별분석 다이얼로그."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QGroupBox, QDialogButtonBox,
    QPushButton, QCheckBox, QRadioButton, QButtonGroup,
    QFormLayout, QWidget, QMessageBox
)
from PySide6.QtCore import Qt

from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    scale_vars, numeric_vars, categorical_vars, all_vars,
    display_label, measure_icon
)


class DiscriminantAnalysisDialog(QDialog):
    """SPSS 스타일 선형 판별분석 다이얼로그.

    구성:
    - 집단 변수 선택 (단일, nominal/ordinal)
    - 독립 변수 선택 (다중, >> 버튼)
    - 입력 방법: 동시 입력 / 단계적
    - 출력 통계: Wilks Lambda, 고유값, 분류표
    """

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("판별분석 (Linear Discriminant Analysis)")
        self.setMinimumSize(620, 580)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 집단 변수 선택
        dep_group = QGroupBox("집단 변수")
        dep_form = QFormLayout(dep_group)
        self.dep_combo = QComboBox()
        self.dep_combo.addItem("-- 선택하세요 --", None)
        _cat = categorical_vars(self._dataset) or all_vars(self._dataset)
        for col in _cat:
            icon = measure_icon(self._dataset, col)
            label = display_label(self._dataset, col)
            text = f"{icon} {label}" if icon else label
            self.dep_combo.addItem(text, col)
        dep_form.addRow("집단 변수:", self.dep_combo)
        layout.addWidget(dep_group)

        # 독립 변수 선택
        pred_group = QGroupBox("독립 변수 (예측 변수)")
        pred_layout = QHBoxLayout(pred_group)

        left_vbox = QVBoxLayout()
        left_vbox.addWidget(QLabel("사용 가능한 변수:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        _pred_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        for v in _pred_vars:
            icon = measure_icon(self._dataset, v)
            label = display_label(self._dataset, v)
            item = QListWidgetItem(f"{icon} {label}" if icon else label)
            item.setData(0x0100, v)
            self.available_list.addItem(item)
        left_vbox.addWidget(self.available_list)
        pred_layout.addLayout(left_vbox)

        btn_vbox = QVBoxLayout()
        btn_vbox.addStretch()
        btn_add = QPushButton(">")
        btn_add.setFixedWidth(36)
        btn_add.clicked.connect(self._add_vars)
        btn_remove = QPushButton("<")
        btn_remove.setFixedWidth(36)
        btn_remove.clicked.connect(self._remove_vars)
        btn_add_all = QPushButton(">>")
        btn_add_all.setFixedWidth(36)
        btn_add_all.clicked.connect(self._add_all_vars)
        btn_remove_all = QPushButton("<<")
        btn_remove_all.setFixedWidth(36)
        btn_remove_all.clicked.connect(self._remove_all_vars)
        btn_vbox.addWidget(btn_add_all)
        btn_vbox.addWidget(btn_add)
        btn_vbox.addWidget(btn_remove)
        btn_vbox.addWidget(btn_remove_all)
        btn_vbox.addStretch()
        pred_layout.addLayout(btn_vbox)

        right_vbox = QVBoxLayout()
        right_vbox.addWidget(QLabel("선택된 독립 변수:"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        right_vbox.addWidget(self.selected_list)
        pred_layout.addLayout(right_vbox)

        layout.addWidget(pred_group)

        # 방법 선택
        method_group = QGroupBox("입력 방법")
        method_layout = QHBoxLayout(method_group)
        self.radio_enter = QRadioButton("동시 입력 (Enter)")
        self.radio_stepwise = QRadioButton("단계적 (Stepwise)")
        self.radio_enter.setChecked(True)
        self._method_group = QButtonGroup(self)
        self._method_group.addButton(self.radio_enter, 0)
        self._method_group.addButton(self.radio_stepwise, 1)
        method_layout.addWidget(self.radio_enter)
        method_layout.addWidget(self.radio_stepwise)
        method_layout.addStretch()
        layout.addWidget(method_group)

        # 사전 확률
        prior_group = QGroupBox("사전 확률")
        prior_layout = QHBoxLayout(prior_group)
        self.radio_proportional = QRadioButton("표본 비율 기반")
        self.radio_equal = QRadioButton("동일 확률")
        self.radio_proportional.setChecked(True)
        self._prior_group = QButtonGroup(self)
        self._prior_group.addButton(self.radio_proportional, 0)
        self._prior_group.addButton(self.radio_equal, 1)
        prior_layout.addWidget(self.radio_proportional)
        prior_layout.addWidget(self.radio_equal)
        prior_layout.addStretch()
        layout.addWidget(prior_group)

        # 출력 통계 선택
        stat_group = QGroupBox("출력 통계")
        stat_layout = QVBoxLayout(stat_group)
        self.chk_wilks = QCheckBox("Wilks' Lambda 검정")
        self.chk_wilks.setChecked(True)
        self.chk_eigenvalue = QCheckBox("고유값 및 설명 분산")
        self.chk_eigenvalue.setChecked(True)
        self.chk_structure = QCheckBox("구조 행렬 (Structure Matrix)")
        self.chk_structure.setChecked(True)
        self.chk_classification = QCheckBox("분류 결과 행렬")
        self.chk_classification.setChecked(True)
        self.chk_centroids = QCheckBox("집단 중심점")
        self.chk_centroids.setChecked(True)
        stat_layout.addWidget(self.chk_wilks)
        stat_layout.addWidget(self.chk_eigenvalue)
        stat_layout.addWidget(self.chk_structure)
        stat_layout.addWidget(self.chk_classification)
        stat_layout.addWidget(self.chk_centroids)
        layout.addWidget(stat_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ------------------------------------------------------------------
    # 변수 이동 버튼 슬롯
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 유효성 검사 및 spec 반환
    # ------------------------------------------------------------------

    def _on_ok(self):
        dep = self.dep_combo.currentData()
        if not dep:
            QMessageBox.warning(self, "경고", "집단 변수를 선택하세요.")
            return
        predictors = [
            self.selected_list.item(i).data(0x0100) or self.selected_list.item(i).text()
            for i in range(self.selected_list.count())
        ]
        if len(predictors) < 1:
            QMessageBox.warning(self, "경고", "독립 변수를 1개 이상 선택하세요.")
            return
        if dep in predictors:
            QMessageBox.warning(self, "경고", "집단 변수가 독립 변수 목록에 포함되어 있습니다.")
            return
        self.accept()

    def get_spec(self) -> dict:
        """분석 스펙 반환."""
        dep = self.dep_combo.currentData()
        predictors = [
            self.selected_list.item(i).data(0x0100) or self.selected_list.item(i).text()
            for i in range(self.selected_list.count())
        ]
        method = "enter" if self.radio_enter.isChecked() else "stepwise"
        prior = "proportional" if self.radio_proportional.isChecked() else "equal"

        return {
            "analysis_id": "discriminant_analysis",
            "variables": {
                "dependent": dep,
                "predictors": predictors,
            },
            "method": method,
            "options": {
                "method": method,
                "prior": prior,
            },
            "display": {
                "wilks_lambda": self.chk_wilks.isChecked(),
                "eigenvalue": self.chk_eigenvalue.isChecked(),
                "structure_matrix": self.chk_structure.isChecked(),
                "classification_table": self.chk_classification.isChecked(),
                "group_centroids": self.chk_centroids.isChecked(),
            },
        }

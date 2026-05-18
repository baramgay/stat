"""Logistic Regression Dialog — SPSS 스타일 로지스틱 회귀 다이얼로그."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QGroupBox, QDialogButtonBox,
    QPushButton, QCheckBox, QDoubleSpinBox, QFormLayout, QWidget,
    QMessageBox
)
from PySide6.QtCore import Qt, Signal

from statworkbench.core.dataset import Dataset


class LogisticRegressionDialog(QDialog):
    """SPSS 스타일 로지스틱 회귀 다이얼로그.

    이항 또는 다항 로지스틱 회귀를 지원합니다.
    종속 변수의 고유값 수에 따라 자동으로 이항/다항을 감지합니다.
    """

    analysis_requested = Signal(str, dict)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("로지스틱 회귀")
        self.setMinimumSize(640, 540)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 종속 변수
        dep_group = QGroupBox("종속 변수 (Dependent Variable)")
        dep_layout = QVBoxLayout(dep_group)

        self.dep_combo = QComboBox()
        for var in self._dataset.data.columns:
            self.dep_combo.addItem(var)
        self.dep_combo.currentIndexChanged.connect(self._on_dep_changed)
        dep_layout.addWidget(self.dep_combo)

        self.dep_type_label = QLabel("")
        self.dep_type_label.setStyleSheet("color: #1a5276; font-size: 11px;")
        dep_layout.addWidget(self.dep_type_label)

        layout.addWidget(dep_group)

        # 독립 변수
        ind_group = QGroupBox("독립 변수 (Independent Variables / Covariates)")
        ind_layout = QVBoxLayout(ind_group)

        ind_layout.addWidget(QLabel("사용 가능한 변수에서 선택:"))

        list_layout = QHBoxLayout()

        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for var in self._dataset.data.columns:
            self.available_list.addItem(QListWidgetItem(var))
        list_layout.addWidget(self.available_list)

        move_layout = QVBoxLayout()
        move_layout.addStretch()
        btn_add = QPushButton(">")
        btn_add.setFixedWidth(36)
        btn_add.clicked.connect(self._add_vars)
        btn_remove = QPushButton("<")
        btn_remove.setFixedWidth(36)
        btn_remove.clicked.connect(self._remove_vars)
        move_layout.addWidget(btn_add)
        move_layout.addWidget(btn_remove)
        move_layout.addStretch()
        list_layout.addLayout(move_layout)

        self.ind_list = QListWidget()
        self.ind_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        list_layout.addWidget(self.ind_list)

        ind_layout.addLayout(list_layout)
        layout.addWidget(ind_group)

        # 방법
        method_group = QGroupBox("방법 (Method)")
        method_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItem("입력 (Enter)", "enter")
        self.method_combo.addItem("전진 (Forward)", "forward")
        self.method_combo.addItem("후진 (Backward)", "backward")
        self.method_combo.addItem("단계적 (Stepwise)", "stepwise")
        method_layout.addWidget(self.method_combo)
        layout.addWidget(method_group)

        # 옵션
        option_group = QGroupBox("옵션")
        option_form = QFormLayout(option_group)

        ci_layout = QHBoxLayout()
        self.ci_check = QCheckBox("신뢰구간 (Exp(B)):")
        self.ci_check.setChecked(True)
        ci_layout.addWidget(self.ci_check)
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(80.0, 99.9)
        self.ci_spin.setValue(95.0)
        self.ci_spin.setSuffix("%")
        self.ci_spin.setSingleStep(1.0)
        ci_layout.addWidget(self.ci_spin)
        ci_layout.addStretch()
        ci_widget = QWidget()
        ci_widget.setLayout(ci_layout)
        option_form.addRow(ci_widget)

        self.hosmer_check = QCheckBox("호스머-레메쇼 검정 (Hosmer-Lemeshow)")
        option_form.addRow(self.hosmer_check)

        self.classification_check = QCheckBox("분류표 (Classification Table)")
        self.classification_check.setChecked(True)
        option_form.addRow(self.classification_check)

        layout.addWidget(option_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # 초기 종속 변수 타입 업데이트
        self._on_dep_changed()

    def _on_dep_changed(self):
        """종속 변수 변경 시 이항/다항 자동 감지."""
        dep_var = self.dep_combo.currentText()
        if dep_var and dep_var in self._dataset.data.columns:
            n_unique = self._dataset.data[dep_var].nunique()
            if n_unique == 2:
                self.dep_type_label.setText(f"감지: 이항 로지스틱 (고유값 2개)")
            elif n_unique > 2:
                self.dep_type_label.setText(f"감지: 다항 로지스틱 (고유값 {n_unique}개)")
            else:
                self.dep_type_label.setText("주의: 고유값이 너무 적습니다")

    def _add_vars(self):
        """선택된 변수를 독립변수 목록에 추가."""
        for item in self.available_list.selectedItems():
            already = [self.ind_list.item(i).text() for i in range(self.ind_list.count())]
            if item.text() not in already:
                self.ind_list.addItem(QListWidgetItem(item.text()))

    def _remove_vars(self):
        """선택된 변수를 독립변수 목록에서 제거."""
        for item in self.ind_list.selectedItems():
            self.ind_list.takeItem(self.ind_list.row(item))

    def get_spec(self) -> dict:
        """분석 스펙 반환."""
        dep_var = self.dep_combo.currentText()
        ind_vars = [self.ind_list.item(i).text() for i in range(self.ind_list.count())]
        dep_unique = self._dataset.data[dep_var].nunique() if dep_var else 0

        return {
            "analysis_id": "logistic_regression",
            "dependent": dep_var,
            "independents": ind_vars,
            "method": self.method_combo.currentData(),
            "logistic_type": "binary" if dep_unique == 2 else "multinomial",
            "options": {
                "confidence_interval": self.ci_spin.value() / 100.0 if self.ci_check.isChecked() else None,
                "hosmer_lemeshow": self.hosmer_check.isChecked(),
                "classification_table": self.classification_check.isChecked(),
            },
        }

    def _on_ok(self):
        dep_var = self.dep_combo.currentText()
        if not dep_var:
            QMessageBox.warning(self, "경고", "종속 변수를 선택하세요.")
            return

        ind_vars = [self.ind_list.item(i).text() for i in range(self.ind_list.count())]
        if not ind_vars:
            QMessageBox.warning(self, "경고", "독립 변수를 하나 이상 선택하세요.")
            return

        spec = self.get_spec()
        self.analysis_requested.emit("logistic_regression", spec)
        self.accept()

"""Survival Analysis Dialog — SPSS Kaplan-Meier/Cox 스타일 생존분석 다이얼로그."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    categorical_vars,
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
)


class SurvivalAnalysisDialog(QDialog):
    """SPSS 스타일 생존분석 다이얼로그.

    탭 구성:
    - Kaplan-Meier: 시간 변수, 상태 변수, 비교 요인
    - Cox 비례위험: 공변량 목록 선택
    """

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("생존분석 (Kaplan-Meier / Cox)")
        self.setMinimumSize(580, 540)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 탭 위젯
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._build_km_tab(), "Kaplan-Meier")
        self.tab_widget.addTab(self._build_cox_tab(), "Cox 비례위험")
        layout.addWidget(self.tab_widget)

        # 옵션 그룹
        opt_group = QGroupBox("분석 옵션")
        opt_layout = QHBoxLayout(opt_group)
        opt_layout.addWidget(QLabel("신뢰수준 (%)"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(80.0, 99.9)
        self.ci_spin.setSingleStep(0.5)
        self.ci_spin.setValue(95.0)
        self.ci_spin.setDecimals(1)
        self.ci_spin.setFixedWidth(70)
        opt_layout.addWidget(self.ci_spin)
        opt_layout.addStretch()
        self.chk_median = QCheckBox("중앙 생존 시간 표시")
        self.chk_median.setChecked(True)
        opt_layout.addWidget(self.chk_median)
        layout.addWidget(opt_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _build_km_tab(self) -> QWidget:
        """Kaplan-Meier 탭 구성."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        form = QFormLayout()

        _scale = scale_vars(self._dataset) or numeric_vars(self._dataset)
        _num = numeric_vars(self._dataset)
        _cat = categorical_vars(self._dataset) or all_vars(self._dataset)

        def _add_to(combo, var_list, placeholder=None):
            if placeholder:
                combo.addItem(placeholder, None)
            for v in var_list:
                icon = measure_icon(self._dataset, v)
                label = display_label(self._dataset, v)
                combo.addItem(f"{icon} {label}" if icon else label, v)

        # 시간 변수 — 척도형(생존 시간은 연속형)
        self.duration_combo = QComboBox()
        _add_to(self.duration_combo, _scale, "-- 선택하세요 --")
        form.addRow("시간 변수 (생존 시간):", self.duration_combo)

        # 상태 변수 — 수치형 (이벤트 0/1)
        event_row_layout = QHBoxLayout()
        self.event_combo = QComboBox()
        _add_to(self.event_combo, _num, "-- 선택하세요 --")
        event_row_layout.addWidget(self.event_combo)
        event_row_layout.addWidget(QLabel("이벤트 값:"))
        self.event_value_spin = QSpinBox()
        self.event_value_spin.setRange(0, 9999)
        self.event_value_spin.setValue(1)
        self.event_value_spin.setFixedWidth(60)
        event_row_layout.addWidget(self.event_value_spin)
        form.addRow("상태 변수:", event_row_layout)

        # 비교 요인 — 범주형 우선 (Log-rank 그룹 비교)
        self.group_combo = QComboBox()
        self.group_combo.addItem("-- 없음 (단일 집단) --", None)
        for v in _cat:
            icon = measure_icon(self._dataset, v)
            label = display_label(self._dataset, v)
            self.group_combo.addItem(f"{icon} {label}" if icon else label, v)
        form.addRow("비교 요인 (Log-rank용):", self.group_combo)

        layout.addLayout(form)
        layout.addStretch()
        return widget

    def _build_cox_tab(self) -> QWidget:
        """Cox 비례위험 탭 구성."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        cov_group = QGroupBox("공변량 선택 (Cox 회귀)")
        cov_layout = QHBoxLayout(cov_group)

        # 왼쪽: 사용 가능한 숫자형 변수
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(QLabel("사용 가능한 변수:"))
        self.cov_available = QListWidget()
        self.cov_available.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        _cov_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        for v in _cov_vars:
            icon = measure_icon(self._dataset, v)
            label = display_label(self._dataset, v)
            item = QListWidgetItem(f"{icon} {label}" if icon else label)
            item.setData(0x0100, v)
            self.cov_available.addItem(item)
        left_vbox.addWidget(self.cov_available)
        cov_layout.addLayout(left_vbox)

        # 가운데: 이동 버튼
        btn_vbox = QVBoxLayout()
        btn_vbox.addStretch()
        btn_add = QPushButton("→")
        btn_add.setFixedWidth(36)
        btn_add.clicked.connect(self._add_covariates)
        btn_remove = QPushButton("←")
        btn_remove.setFixedWidth(36)
        btn_remove.clicked.connect(self._remove_covariates)
        btn_add_all = QPushButton(">>")
        btn_add_all.setFixedWidth(36)
        btn_add_all.clicked.connect(self._add_all_covariates)
        btn_remove_all = QPushButton("<<")
        btn_remove_all.setFixedWidth(36)
        btn_remove_all.clicked.connect(self._remove_all_covariates)
        btn_vbox.addWidget(btn_add_all)
        btn_vbox.addWidget(btn_add)
        btn_vbox.addWidget(btn_remove)
        btn_vbox.addWidget(btn_remove_all)
        btn_vbox.addStretch()
        cov_layout.addLayout(btn_vbox)

        # 오른쪽: 선택된 공변량
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(QLabel("선택된 공변량:"))
        self.cov_selected = QListWidget()
        self.cov_selected.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        right_vbox.addWidget(self.cov_selected)
        cov_layout.addLayout(right_vbox)

        layout.addWidget(cov_group)

        note = QLabel(
            "공변량을 선택하지 않으면 Cox 회귀는 실행되지 않습니다.\n"
            "KM 탭의 시간·상태 변수는 공통으로 사용됩니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)
        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # 공변량 이동 버튼 슬롯
    # ------------------------------------------------------------------

    def _clone_item(self, source: QListWidgetItem) -> QListWidgetItem:
        new = QListWidgetItem(source.text())
        new.setData(0x0100, source.data(0x0100))
        return new

    def _add_covariates(self):
        already = {self.cov_selected.item(i).data(0x0100) for i in range(self.cov_selected.count())}
        for item in self.cov_available.selectedItems():
            if item.data(0x0100) not in already:
                self.cov_selected.addItem(self._clone_item(item))

    def _remove_covariates(self):
        for item in self.cov_selected.selectedItems():
            self.cov_selected.takeItem(self.cov_selected.row(item))

    def _add_all_covariates(self):
        already = {self.cov_selected.item(i).data(0x0100) for i in range(self.cov_selected.count())}
        for i in range(self.cov_available.count()):
            src = self.cov_available.item(i)
            if src.data(0x0100) not in already:
                self.cov_selected.addItem(self._clone_item(src))

    def _remove_all_covariates(self):
        self.cov_selected.clear()

    # ------------------------------------------------------------------
    # 유효성 검사 및 spec 반환
    # ------------------------------------------------------------------

    def _on_ok(self):
        duration = self.duration_combo.currentData()
        event = self.event_combo.currentData()
        if not duration:
            QMessageBox.warning(self, "경고", "시간 변수를 선택하세요.")
            return
        if not event:
            QMessageBox.warning(self, "경고", "상태(이벤트) 변수를 선택하세요.")
            return
        if duration == event:
            QMessageBox.warning(self, "경고", "시간 변수와 상태 변수는 달라야 합니다.")
            return
        self.accept()

    def get_spec(self) -> dict:
        """분석 스펙 반환."""
        duration = self.duration_combo.currentData()
        event = self.event_combo.currentData()
        event_value = self.event_value_spin.value()
        group = self.group_combo.currentData()  # None 이면 그룹 없음
        covariates = [
            self.cov_selected.item(i).data(0x0100) or self.cov_selected.item(i).text()
            for i in range(self.cov_selected.count())
        ]

        # 이벤트 값에 따른 전처리 힌트 포함
        method = "both" if covariates else "km"

        return {
            "analysis_id": "survival_analysis",
            "variables": {
                "duration": duration,
                "event": event,
                "event_value": event_value,
                "group": group,
                "covariates": covariates,
            },
            "options": {
                "method": method,
                "alpha": round(1.0 - self.ci_spin.value() / 100.0, 4),
            },
            "confidence_level": self.ci_spin.value() / 100.0,
            "display": {
                "median_survival": self.chk_median.isChecked(),
            },
        }

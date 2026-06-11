"""T-Test Dialog — SPSS 스타일 T 검정 다이얼로그."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from nuristat.analysis.result import AnalysisResult
from nuristat.core.dataset import Dataset
from nuristat.ui.dialogs._async_mixin import AnalysisDialogMixin
from nuristat.ui.dialogs._dialog_helpers import (
    all_vars,
    categorical_vars,
    display_label,
    measure_icon,
    numeric_vars,
    scale_vars,
    var_from_display,
)


def _combo_add(combo: QComboBox, dataset: Dataset, var_list: list[str]) -> None:
    """콤보박스에 변수 추가 — 아이콘 + 라벨(변수명) 형식."""
    for var in var_list:
        icon = measure_icon(dataset, var)
        label = display_label(dataset, var)
        text = f"{icon} {label}" if icon else label
        combo.addItem(text, userData=var)


class IndependentTTestDialog(QDialog, AnalysisDialogMixin):
    """SPSS Independent-Samples T Test 다이얼로그.

    검정 변수: 척도(Scale) 변수만
    그룹 변수: 명목/순서형 우선, 없으면 전체 변수
    """

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("독립표본 T 검정")
        self.setMinimumSize(500, 380)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 검정 변수 (척도)
        test_group = QGroupBox("검정 변수 (Test Variable) — 척도형")
        test_layout = QVBoxLayout(test_group)

        self.test_combo = QComboBox()
        test_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        _combo_add(self.test_combo, self._dataset, test_vars)
        test_layout.addWidget(self.test_combo)
        layout.addWidget(test_group)

        # 그룹 변수 (명목/순서형 우선)
        group_group = QGroupBox("그룹 변수 (Grouping Variable) — 명목/순서형")
        group_layout = QVBoxLayout(group_group)

        self.group_combo = QComboBox()
        cat_vars = categorical_vars(self._dataset)
        group_vars = cat_vars if cat_vars else all_vars(self._dataset)
        _combo_add(self.group_combo, self._dataset, group_vars)
        group_layout.addWidget(self.group_combo)
        layout.addWidget(group_group)

        # 그룹 정의 (Define Groups) — SPSS 방식
        define_group = QGroupBox("그룹 정의 (Define Groups)")
        define_layout = QGridLayout(define_group)
        define_layout.setColumnStretch(1, 1)
        define_layout.addWidget(QLabel("그룹 1:"), 0, 0)
        self._grp1_combo = QComboBox()
        define_layout.addWidget(self._grp1_combo, 0, 1)
        define_layout.addWidget(QLabel("그룹 2:"), 1, 0)
        self._grp2_combo = QComboBox()
        define_layout.addWidget(self._grp2_combo, 1, 1)
        layout.addWidget(define_group)

        self.group_combo.currentIndexChanged.connect(self._update_group_values)
        self._update_group_values()

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QHBoxLayout(opt_group)
        opt_layout.addWidget(QLabel("신뢰수준:"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(0.80, 0.99)
        self.ci_spin.setSingleStep(0.01)
        self.ci_spin.setValue(0.95)
        self.ci_spin.setDecimals(2)
        opt_layout.addWidget(self.ci_spin)
        opt_layout.addStretch()
        layout.addWidget(opt_group)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _val_label(self, var: str, val: object) -> str:
        meta = self._dataset.variables.get(var) if self._dataset.variables else None
        if meta and meta.value_labels:
            key = int(val) if isinstance(val, float) and float(val).is_integer() else val
            lbl = meta.value_labels.get(key) or meta.value_labels.get(str(key))
            if lbl:
                return str(lbl)
        return str(val)

    def _update_group_values(self) -> None:
        """그룹 변수 변경 시 그룹 값 콤보박스 갱신."""
        group_var = self.group_combo.currentData()
        self._grp1_combo.clear()
        self._grp2_combo.clear()
        if not group_var:
            return
        df = self._dataset.data
        if df is None or group_var not in df.columns:
            return
        raw_vals = df[group_var].dropna().unique()
        try:
            values = sorted(raw_vals.tolist() if hasattr(raw_vals, "tolist") else list(raw_vals),
                           key=lambda x: (type(x).__name__, x))
        except TypeError:
            values = list(raw_vals)
        for val in values:
            lbl = self._val_label(group_var, val)
            display = f"{val} ({lbl})" if lbl != str(val) else str(val)
            self._grp1_combo.addItem(display, userData=val)
            self._grp2_combo.addItem(display, userData=val)
        if self._grp2_combo.count() >= 2:
            self._grp2_combo.setCurrentIndex(1)

    def _run(self):
        test_var = self.test_combo.currentData() or var_from_display(self.test_combo.currentText())
        group_var = self.group_combo.currentData() or var_from_display(self.group_combo.currentText())

        if not test_var or not group_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        if test_var == group_var:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "검정 변수와 그룹 변수가 같을 수 없습니다.")
            return

        grp1_val = self._grp1_combo.currentData()
        grp2_val = self._grp2_combo.currentData()
        group_values = [grp1_val, grp2_val] if grp1_val is not None and grp2_val is not None else None

        from nuristat.analysis.ttests import run_analysis
        spec = {
            "variables": {"dependent": test_var, "group": group_var},
            "options": {"equal_var": "auto", "group_values": group_values},
            "confidence_level": self.ci_spin.value(),
        }
        self._start_analysis(run_analysis, self._dataset, spec)


class PairedTTestDialog(QDialog, AnalysisDialogMixin):
    """SPSS Paired-Samples T Test 다이얼로그.

    두 변수 모두 척도(Scale) 변수여야 함.
    """

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("대응표본 T 검정")
        self.setMinimumSize(500, 360)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        pair_group = QGroupBox("변수 쌍 (Paired Variables) — 척도형")
        pair_layout = QVBoxLayout(pair_group)

        paired_vars = scale_vars(self._dataset) or numeric_vars(self._dataset)

        pair_layout.addWidget(QLabel("변수 1 (Variable 1):"))
        self.var1_combo = QComboBox()
        _combo_add(self.var1_combo, self._dataset, paired_vars)
        pair_layout.addWidget(self.var1_combo)

        pair_layout.addWidget(QLabel("변수 2 (Variable 2):"))
        self.var2_combo = QComboBox()
        _combo_add(self.var2_combo, self._dataset, paired_vars)
        if len(paired_vars) > 1:
            self.var2_combo.setCurrentIndex(1)
        pair_layout.addWidget(self.var2_combo)

        layout.addWidget(pair_group)

        # 신뢰수준
        opt_group = QGroupBox("옵션")
        opt_layout = QHBoxLayout(opt_group)
        opt_layout.addWidget(QLabel("신뢰수준:"))
        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setRange(0.80, 0.99)
        self.ci_spin.setSingleStep(0.01)
        self.ci_spin.setValue(0.95)
        self.ci_spin.setDecimals(2)
        opt_layout.addWidget(self.ci_spin)
        opt_layout.addStretch()
        layout.addWidget(opt_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._run_btn = btn_box.button(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self):
        var1 = self.var1_combo.currentData() or var_from_display(self.var1_combo.currentText())
        var2 = self.var2_combo.currentData() or var_from_display(self.var2_combo.currentText())

        if not var1 or not var2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "변수를 선택하세요.")
            return

        if var1 == var2:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "두 변수가 같을 수 없습니다.")
            return

        from nuristat.analysis.ttests import run_analysis
        spec = {
            "variables": {"paired": [var1, var2]},
            "confidence_level": self.ci_spin.value(),
        }
        self._start_analysis(run_analysis, self._dataset, spec)

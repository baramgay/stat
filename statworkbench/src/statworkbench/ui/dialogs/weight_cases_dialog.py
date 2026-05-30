"""Weight Cases Dialog — SPSS 스타일 가중치 적용 다이얼로그.

케이스에 가중치를 적용하여 분석에 반영합니다.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)

from statworkbench.core.dataset import Dataset


class WeightCasesDialog(QDialog):
    """가중치 적용 다이얼로그."""

    weight_applied = Signal(str)  # weight_var
    weight_cleared = Signal()

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset

        self.setWindowTitle("⚖️ 가중치 적용")
        self.setMinimumSize(450, 350)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 가중치 설정
        weight_group = QGroupBox("⚙️ 가중치 설정")
        weight_layout = QVBoxLayout(weight_group)

        self.weight_group = QButtonGroup(self)

        self.no_weight_radio = QRadioButton("가중치 없음 (Do not weight cases)")
        self.no_weight_radio.setChecked(True)
        self.weight_group.addButton(self.no_weight_radio)
        weight_layout.addWidget(self.no_weight_radio)

        self.weight_radio = QRadioButton("가중치 적용 (Weight cases by):")
        self.weight_group.addButton(self.weight_radio)
        weight_layout.addWidget(self.weight_radio)

        # 가중치 변수 선택
        var_layout = QHBoxLayout()
        var_layout.addWidget(QLabel("가중치 변수:"))
        self.weight_combo = QComboBox()

        # 숫자형 변수만
        numeric_cols = self.dataset.data.select_dtypes(include=['number']).columns
        self.weight_combo.addItems(numeric_cols)
        var_layout.addWidget(self.weight_combo)

        weight_layout.addLayout(var_layout)

        layout.addWidget(weight_group)

        # 가중치 정보
        info_group = QGroupBox("📊 가중치 정보")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setStyleSheet(
            "background-color: #f1f3f4; font-family: Consolas; font-size: 11px;"
        )
        info_layout.addWidget(self.info_text)

        self.btn_info = QPushButton("🔍 가중치 정보 보기")
        self.btn_info.clicked.connect(self._show_weight_info)
        info_layout.addWidget(self.btn_info)

        layout.addWidget(info_group)

        # 실행 버튼
        action_layout = QHBoxLayout()

        self.btn_apply = QPushButton("✅ 적용")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #2ca02c; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_apply.clicked.connect(self._apply_weight)
        action_layout.addWidget(self.btn_apply)

        self.btn_cancel = QPushButton("❌ 취소")
        self.btn_cancel.clicked.connect(self.reject)
        action_layout.addWidget(self.btn_cancel)

        action_layout.addStretch()
        layout.addLayout(action_layout)

    def _show_weight_info(self) -> None:
        """가중치 변수 정보 표시."""
        var_name = self.weight_combo.currentText()
        if var_name and var_name in self.dataset.data.columns:
            series = self.dataset.data[var_name]

            info = f"""
변수: {var_name}
최소: {series.min():.4f}
최대: {series.max():.4f}
평균: {series.mean():.4f}
합계: {series.sum():.4f}
음수 값: {(series < 0).sum()}개
0 값: {(series == 0).sum()}개
            """.strip()

            self.info_text.setText(info)

    def _apply_weight(self) -> None:
        """가중치 적용."""
        if self.no_weight_radio.isChecked():
            self.weight_cleared.emit()
            QMessageBox.information(self, "완료", "가중치가 해제되었습니다.")
            self.accept()
        else:
            weight_var = self.weight_combo.currentText()
            if not weight_var:
                QMessageBox.warning(self, "경고", "가중치 변수를 선택하세요")
                return

            # 음수/0 값 확인
            series = self.dataset.data[weight_var]
            negative_count = (series < 0).sum()

            if negative_count > 0:
                reply = QMessageBox.question(
                    self,
                    "확인",
                    f"{negative_count}개의 음수 값이 있습니다. 계속하시겠습니까?"
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            self.weight_applied.emit(weight_var)
            QMessageBox.information(
                self, "완료",
                f"가중치가 적용되었습니다.\n가중치 변수: {weight_var}"
            )
            self.accept()

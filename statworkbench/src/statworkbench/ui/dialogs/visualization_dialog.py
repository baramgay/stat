"""Visualization Dialog — 고급 시각화 다이얼로그.

가독성과 검증 절차를 중시한 UI 설계.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QCheckBox, QSpinBox, QLineEdit,
    QMessageBox, QTabWidget, QTextEdit, QSplitter, QScrollArea,
    QWidget, QGridLayout, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from typing import Optional, List

from statworkbench.core.dataset import Dataset
from statworkbench.analysis.visualization import VisualizationEngine


class VisualizationDialog(QDialog):
    """시각화 다이얼로그."""
    
    chart_created = Signal(str, str)  # 차트 유형, base64 이미지
    
    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.engine = VisualizationEngine()
        
        self.setWindowTitle("📊 시각화")
        self.setMinimumSize(1000, 800)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # 상단: 데이터 정보
        info_label = QLabel(
            f"📊 {dataset.name}: {len(dataset.data)}행 × {len(dataset.data.columns)}변수 | "
            f"숫자형: {len(dataset.data.select_dtypes(include=['number']).columns)}개"
        )
        info_label.setStyleSheet(
            "font-size: 13px; color: #1a5276; padding: 8px; "
            "background-color: #d4e6f1; border-radius: 4px;"
        )
        layout.addWidget(info_label)
        
        # 메인 스플리터
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 왼쪽: 설정 패널
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # 차트 유형 선택
        chart_group = QGroupBox("📈 차트 유형")
        chart_layout = QGridLayout(chart_group)
        
        self.chart_type_group = QButtonGroup(self)
        chart_types = [
            ("막대 차트", "bar"),
            ("히스토그램", "hist"),
            ("산점도", "scatter"),
            ("상자 그림", "box"),
            ("선 차트", "line"),
            ("히트맵", "heatmap"),
            ("바이올린", "violin"),
        ]
        
        for i, (label, value) in enumerate(chart_types):
            btn = QRadioButton(label)
            btn.setProperty("chart_type", value)
            self.chart_type_group.addButton(btn)
            chart_layout.addWidget(btn, i // 2, i % 2)
            if i == 0:
                btn.setChecked(True)
        
        left_layout.addWidget(chart_group)
        
        # 변수 선택
        vars_group = QGroupBox("🔢 변수 선택")
        vars_layout = QVBoxLayout(vars_group)
        
        # X 변수
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X 변수:"))
        self.x_combo = QComboBox()
        self.x_combo.addItem("(선택)")
        self.x_combo.addItems(dataset.data.columns)
        x_layout.addWidget(self.x_combo)
        vars_layout.addLayout(x_layout)
        
        # Y 변수
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y 변수:"))
        self.y_combo = QComboBox()
        self.y_combo.addItem("(선택)")
        self.y_combo.addItems(dataset.data.columns)
        y_layout.addWidget(self.y_combo)
        vars_layout.addLayout(y_layout)
        
        # 그룹 변수
        hue_layout = QHBoxLayout()
        hue_layout.addWidget(QLabel("그룹 변수:"))
        self.hue_combo = QComboBox()
        self.hue_combo.addItem("(없음)")
        self.hue_combo.addItems(dataset.data.columns)
        hue_layout.addWidget(self.hue_combo)
        vars_layout.addLayout(hue_layout)
        
        left_layout.addWidget(vars_group)
        
        # 옵션
        options_group = QGroupBox("⚙️ 옵션")
        options_layout = QVBoxLayout(options_group)
        
        # 제목
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("제목:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("차트 제목 (선택)")
        title_layout.addWidget(self.title_edit)
        options_layout.addLayout(title_layout)
        
        # 히스토그램 빈 수
        bins_layout = QHBoxLayout()
        bins_layout.addWidget(QLabel("빈 수:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 100)
        self.bins_spin.setValue(30)
        bins_layout.addWidget(self.bins_spin)
        options_layout.addLayout(bins_layout)
        
        # KDE
        self.kde_check = QCheckBox("KDE 곡선 표시")
        self.kde_check.setChecked(True)
        options_layout.addWidget(self.kde_check)
        
        # 회귀선
        self.reg_check = QCheckBox("회귀선 표시")
        options_layout.addWidget(self.reg_check)
        
        left_layout.addWidget(options_group)
        
        # 검증 정보
        self.validation_group = QGroupBox("✅ 검증")
        self.validation_layout = QVBoxLayout(self.validation_group)
        self.validation_label = QLabel("변수를 선택하고 차트를 생성하세요")
        self.validation_label.setWordWrap(True)
        self.validation_layout.addWidget(self.validation_label)
        left_layout.addWidget(self.validation_group)
        
        # 실행 버튼
        btn_layout = QHBoxLayout()
        self.btn_create = QPushButton("📊 차트 생성")
        self.btn_create.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 10px 20px; font-size: 13px; }"
        )
        self.btn_create.clicked.connect(self._create_chart)
        btn_layout.addWidget(self.btn_create)
        
        self.btn_save = QPushButton("💾 저장")
        self.btn_save.clicked.connect(self._save_chart)
        self.btn_save.setEnabled(False)
        btn_layout.addWidget(self.btn_save)
        
        left_layout.addLayout(btn_layout)
        left_layout.addStretch()
        
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)
        
        # 오른쪽: 미리보기
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.preview_label = QLabel("차트를 생성하면 여기에 표시됩니다")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #f1f3f4; border: 2px dashed #c0c4cc; "
            "padding: 40px; color: #7a7a8a; font-size: 14px;"
        )
        self.preview_label.setMinimumSize(500, 400)
        right_layout.addWidget(self.preview_label)
        
        # 결과 텍스트
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setStyleSheet(
            "background-color: #1a1a2e; color: #e8e8f0; "
            "font-family: Consolas; font-size: 11px;"
        )
        right_layout.addWidget(self.result_text)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([350, 650])
        layout.addWidget(splitter)
    
    def _get_selected_chart_type(self) -> str:
        """선택된 차트 유형 반환."""
        for btn in self.chart_type_group.buttons():
            if btn.isChecked():
                return btn.property("chart_type")
        return "bar"
    
    def _validate_selection(self) -> dict:
        """변수 선택 검증."""
        chart_type = self._get_selected_chart_type()
        x = self.x_combo.currentText()
        y = self.y_combo.currentText()
        
        result = {"valid": True, "errors": [], "warnings": []}
        
        if chart_type in ("bar", "hist"):
            if x == "(선택)":
                result["valid"] = False
                result["errors"].append("X 변수를 선택하세요")
        
        elif chart_type in ("scatter", "line", "box", "violin"):
            if x == "(선택)" or y == "(선택)":
                result["valid"] = False
                result["errors"].append("X 변수와 Y 변수를 모두 선택하세요")
        
        elif chart_type == "heatmap":
            numeric_cols = self.dataset.data.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                result["valid"] = False
                result["errors"].append("히트맵에는 2개 이상의 숫자형 변수가 필요합니다")
        
        return result
    
    def _create_chart(self) -> None:
        """차트 생성."""
        validation = self._validate_selection()
        
        if not validation["valid"]:
            self.validation_label.setText(
                "❌ " + "\n".join(validation["errors"])
            )
            self.validation_label.setStyleSheet("color: #d62728;")
            QMessageBox.warning(self, "검증 오류", "\n".join(validation["errors"]))
            return
        
        chart_type = self._get_selected_chart_type()
        x = self.x_combo.currentText()
        y = self.y_combo.currentText()
        hue = self.hue_combo.currentText()
        if hue == "(없음)":
            hue = None
        
        title = self.title_edit.text() or ""
        
        try:
            if chart_type == "bar":
                img = self.engine.bar_chart(
                    self.dataset.data, x, y if y != "(선택)" else None,
                    hue, title
                )
            elif chart_type == "hist":
                img = self.engine.histogram(
                    self.dataset.data, x,
                    bins=self.bins_spin.value(),
                    kde=self.kde_check.isChecked(),
                    title=title
                )
            elif chart_type == "scatter":
                img = self.engine.scatter_plot(
                    self.dataset.data, x, y, hue,
                    title=title,
                    add_regression=self.reg_check.isChecked()
                )
            elif chart_type == "box":
                img = self.engine.box_plot(
                    self.dataset.data, x if x != "(선택)" else None,
                    y, hue, title
                )
            elif chart_type == "line":
                img = self.engine.line_chart(
                    self.dataset.data, x, y, hue, title
                )
            elif chart_type == "heatmap":
                cols = list(self.dataset.data.select_dtypes(include=['number']).columns)
                img = self.engine.heatmap(self.dataset.data, cols, title)
            elif chart_type == "violin":
                img = self.engine.violin_plot(
                    self.dataset.data, x, y, hue, title
                )
            else:
                QMessageBox.warning(self, "오류", f"지원하지 않는 차트 유형: {chart_type}")
                return
            
            # 미리보기 표시
            if img.startswith("data:image/png;base64,"):
                pixmap = QPixmap()
                import base64
                pixmap.loadFromData(base64.b64decode(img.split(",")[1]))
                self.preview_label.setPixmap(pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
                self.preview_label.setStyleSheet("")
                self._current_image = img
                self.btn_save.setEnabled(True)
            
            # 검증 정보 업데이트
            self.validation_label.setText("✅ 차트 생성 완료")
            self.validation_label.setStyleSheet("color: #2ca02c;")
            
            self.result_text.append(f"[성공] {chart_type} 차트 생성 완료")
            
            # 시그널 발생
            self.chart_created.emit(chart_type, img)
            
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"차트 생성 실패:\n{exc}")
            self.result_text.append(f"[오류] {exc}")
    
    def _save_chart(self) -> None:
        """차트 저장."""
        if not hasattr(self, '_current_image'):
            return
        
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "차트 저장", "", "PNG (*.png);;JPEG (*.jpg);;SVG (*.svg)"
        )
        if path:
            import base64
            img_data = base64.b64decode(self._current_image.split(",")[1])
            with open(path, "wb") as f:
                f.write(img_data)
            self.result_text.append(f"[저장] {path}")

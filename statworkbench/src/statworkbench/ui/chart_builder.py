"""Chart Builder — SPSS 스타일 차트 빌더.

SPSS Chart Builder 기능:
- 차트 유형 선택 (막대, 선, 산점도, 히스토그램, 상자 그림)
- X/Y 축 변수 할당
- 그룹화/패널 변수
- 차트 미리보기 (matplotlib)
- 차트 저장
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QGroupBox,
    QDialog, QDialogButtonBox, QSplitter, QFileDialog,
    QMessageBox, QTabWidget, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
from typing import Optional, List

import pandas as pd
import numpy as np

from statworkbench.core.dataset import Dataset


class ChartBuilderDialog(QDialog):
    """SPSS Chart Builder 다이얼로그."""

    chart_saved = Signal(str)  # 저장 경로

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("차트 빌더")
        self.setMinimumSize(900, 700)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 데이터셋 정보
        info = QLabel(f"데이터셋: {self._dataset.name} | "
                      f"케이스: {len(self._dataset.data)} | "
                      f"변수: {len(self._dataset.data.columns)}")
        info.setStyleSheet("color: #5d6d7e; font-size: 12px;")
        layout.addWidget(info)

        # 스플리터
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 설정 패널
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 차트 유형
        type_group = QGroupBox("차트 유형")
        type_layout = QVBoxLayout(type_group)

        self.chart_type = QComboBox()
        self.chart_type.addItems([
            "막대 차트 (Bar)",
            "선 차트 (Line)",
            "산점도 (Scatter)",
            "히스토그램 (Histogram)",
            "상자 그림 (Boxplot)",
            "원형 차트 (Pie)",
        ])
        self.chart_type.currentIndexChanged.connect(self._on_chart_type_changed)
        type_layout.addWidget(self.chart_type)
        left_layout.addWidget(type_group)

        # 변수 할당
        var_group = QGroupBox("변수 할당")
        var_layout = QVBoxLayout(var_group)

        var_layout.addWidget(QLabel("X 축:"))
        self.x_combo = QComboBox()
        self.x_combo.addItem("(없음)")
        for col in self._dataset.data.columns:
            self.x_combo.addItem(col)
        var_layout.addWidget(self.x_combo)

        var_layout.addWidget(QLabel("Y 축:"))
        self.y_combo = QComboBox()
        self.y_combo.addItem("(없음)")
        for col in self._dataset.data.columns:
            self.y_combo.addItem(col)
        var_layout.addWidget(self.y_combo)

        var_layout.addWidget(QLabel("그룹화:"))
        self.group_combo = QComboBox()
        self.group_combo.addItem("(없음)")
        for col in self._dataset.data.columns:
            self.group_combo.addItem(col)
        var_layout.addWidget(self.group_combo)

        left_layout.addWidget(var_group)

        # 옵션
        opt_group = QGroupBox("옵션")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_grid = QCheckBox("격자선 표시")
        self.chk_grid.setChecked(True)
        opt_layout.addWidget(self.chk_grid)

        self.chk_legend = QCheckBox("범례 표시")
        self.chk_legend.setChecked(True)
        opt_layout.addWidget(self.chk_legend)

        self.chk_title = QCheckBox("제목 표시")
        self.chk_title.setChecked(True)
        opt_layout.addWidget(self.chk_title)

        left_layout.addWidget(opt_group)

        # 버튼
        btn_layout = QHBoxLayout()

        self.btn_preview = QPushButton("미리보기")
        self.btn_preview.clicked.connect(self._generate_chart)
        btn_layout.addWidget(self.btn_preview)

        self.btn_save = QPushButton("저장")
        self.btn_save.clicked.connect(self._save_chart)
        btn_layout.addWidget(self.btn_save)

        left_layout.addLayout(btn_layout)
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # 오른쪽: 미리보기
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("미리보기:"))

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #f8f9fa; border: 1px solid #dee2e6;"
        )
        self.preview_label.setMinimumSize(500, 400)
        right_layout.addWidget(self.preview_label)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 600])

        layout.addWidget(splitter)

        # 닫기 버튼
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_chart_type_changed(self):
        """차트 유형 변경 시."""
        chart_type = self.chart_type.currentText()

        # 차트 유형에 따라 기본 변수 설정
        if "히스토그램" in chart_type:
            # 히스토그램은 Y축 불필요
            self.y_combo.setEnabled(False)
        else:
            self.y_combo.setEnabled(True)

    def _generate_chart(self):
        """차트 생성."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            chart_type = self.chart_type.currentText()
            x_var = self.x_combo.currentText()
            y_var = self.y_combo.currentText()
            group_var = self.group_combo.currentText()

            df = self._dataset.data

            # 그림 생성
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

            if "막대" in chart_type:
                self._create_bar_chart(ax, df, x_var, y_var, group_var)
            elif "선" in chart_type:
                self._create_line_chart(ax, df, x_var, y_var, group_var)
            elif "산점도" in chart_type:
                self._create_scatter_chart(ax, df, x_var, y_var, group_var)
            elif "히스토그램" in chart_type:
                self._create_histogram(ax, df, x_var, group_var)
            elif "상자" in chart_type:
                self._create_boxplot(ax, df, x_var, y_var, group_var)
            elif "원형" in chart_type:
                self._create_pie_chart(ax, df, x_var, y_var)

            # 옵션 적용
            if self.chk_grid.isChecked():
                ax.grid(True, alpha=0.3)
            if self.chk_legend.isChecked():
                ax.legend()
            if self.chk_title.isChecked():
                ax.set_title(f"{chart_type}")

            plt.tight_layout()

            # QPixmap으로 변환
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            l, b, w, h = fig.bbox.bounds
            img = QImage(buf, int(w), int(h), QImage.Format_ARGB32)
            pixmap = QPixmap.fromImage(img)

            self.preview_label.setPixmap(pixmap)
            plt.close(fig)

        except Exception as exc:
            QMessageBox.critical(self, "오류", f"차트 생성 실패:\n{exc}")

    def _create_bar_chart(self, ax, df, x_var, y_var, group_var):
        """막대 차트 생성."""
        if x_var == "(없음)" or y_var == "(없음)":
            ax.text(0.5, 0.5, "X축과 Y축 변수를 선택하세요", 
                   ha='center', va='center', transform=ax.transAxes)
            return

        if group_var != "(없음)":
            for name, group in df.groupby(group_var):
                ax.bar(group[x_var], group[y_var], label=str(name), alpha=0.7)
        else:
            ax.bar(df[x_var], df[y_var])
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    def _create_line_chart(self, ax, df, x_var, y_var, group_var):
        """선 차트 생성."""
        if x_var == "(없음)" or y_var == "(없음)":
            ax.text(0.5, 0.5, "X축과 Y축 변수를 선택하세요",
                   ha='center', va='center', transform=ax.transAxes)
            return

        if group_var != "(없음)":
            for name, group in df.groupby(group_var):
                ax.plot(group[x_var], group[y_var], marker='o', label=str(name))
        else:
            ax.plot(df[x_var], df[y_var], marker='o')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    def _create_scatter_chart(self, ax, df, x_var, y_var, group_var):
        """산점도 생성."""
        if x_var == "(없음)" or y_var == "(없음)":
            ax.text(0.5, 0.5, "X축과 Y축 변수를 선택하세요",
                   ha='center', va='center', transform=ax.transAxes)
            return

        if group_var != "(없음)":
            for name, group in df.groupby(group_var):
                ax.scatter(group[x_var], group[y_var], label=str(name), alpha=0.6)
        else:
            ax.scatter(df[x_var], df[y_var], alpha=0.6)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    def _create_histogram(self, ax, df, x_var, group_var):
        """히스토그램 생성."""
        if x_var == "(없음)":
            ax.text(0.5, 0.5, "X축 변수를 선택하세요",
                   ha='center', va='center', transform=ax.transAxes)
            return

        if group_var != "(없음)":
            for name, group in df.groupby(group_var):
                ax.hist(group[x_var], alpha=0.5, label=str(name), bins=20)
        else:
            ax.hist(df[x_var], bins=20, edgecolor='black')
        ax.set_xlabel(x_var)
        ax.set_ylabel("빈도")

    def _create_boxplot(self, ax, df, x_var, y_var, group_var):
        """상자 그림 생성."""
        if y_var == "(없음)":
            ax.text(0.5, 0.5, "Y축 변수를 선택하세요",
                   ha='center', va='center', transform=ax.transAxes)
            return

        if x_var != "(없음)" and x_var in df.columns:
            df.boxplot(column=y_var, by=x_var, ax=ax)
            ax.set_title(f"{y_var} by {x_var}")
        else:
            ax.boxplot(df[y_var].dropna())
            ax.set_ylabel(y_var)

    def _create_pie_chart(self, ax, df, x_var, y_var):
        """원형 차트 생성."""
        if x_var == "(없음)":
            ax.text(0.5, 0.5, "X축 변수를 선택하세요",
                   ha='center', va='center', transform=ax.transAxes)
            return

        counts = df[x_var].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax.set_title(f"{x_var} 분포")

    def _save_chart(self):
        """차트 저장."""
        path, _ = QFileDialog.getSaveFileName(
            self, "차트 저장", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            pixmap = self.preview_label.pixmap()
            if pixmap:
                pixmap.save(path)
                self.chart_saved.emit(path)
                QMessageBox.information(self, "저장 완료", f"차트가 저장되었습니다:\n{path}")
            else:
                QMessageBox.warning(self, "경고", "먼저 미리보기를 생성하세요.")


class ChartBuilderWidget(QWidget):
    """메인 윈도우에 통합될 차트 빌더 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("차트 빌더 — 데이터를 불러오세요")
        layout.addWidget(self.label)

    def set_dataset(self, dataset: Dataset):
        """데이터셋 설정."""
        self._dataset = dataset
        self.label.setText(f"차트 빌더 — {dataset.name}")

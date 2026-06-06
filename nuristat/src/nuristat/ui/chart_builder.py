"""Chart Builder — SPSS 스타일 차트 빌더 (완성판).

matplotlib FigureCanvas 임베딩을 활용한 실시간 미리보기.
왼쪽: 차트 유형 선택 | 가운데: 변수·옵션 | 오른쪽: 실시간 미리보기
"""

from __future__ import annotations

import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nuristat.analysis.visualization import VisualizationEngine
from nuristat.core.dataset import Dataset

logger = logging.getLogger(__name__)

# 한글 폰트 보장
plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ── 차트 유형 정의 ──────────────────────────────────────────────────────────

CHART_TYPES: list[tuple[str, str, str]] = [
    ("histogram",   "히스토그램",     "분포 분석 — 정규 곡선 선택 가능"),
    ("scatter",     "산점도",         "두 수치 변수의 관계"),
    ("bar",         "막대 그래프",    "범주별 빈도 / 평균"),
    ("line",        "선 그래프",      "추세·시계열"),
    ("boxplot",     "상자 그림",      "사분위수·이상값"),
    ("qq",          "Q-Q 플롯",       "정규성 검정"),
    ("heatmap",     "상관관계 히트맵","변수 간 상관"),
]


class _PreviewCanvas(FigureCanvas):
    """미리보기 전용 FigureCanvas."""

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, "차트를 생성하려면\n왼쪽에서 변수를 선택하고\n'미리보기' 버튼을 클릭하세요",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8))
        ax.axis("off")
        self.draw()

    def display_figure(self, fig: Figure) -> None:
        """외부 Figure의 내용을 캔버스에 표시."""
        self._fig.clear()
        # figure를 PNG 버퍼로 렌더링 후 캔버스에 이미지로 표시
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        from matplotlib.image import imread
        img = imread(buf)
        ax = self._fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.draw()

    def get_figure(self) -> Figure:
        return self._fig


class ChartBuilderDialog(QDialog):
    """SPSS Chart Builder 다이얼로그 (완성판).

    구성:
    - 왼쪽:  차트 유형 선택 패널 (라디오 버튼)
    - 가운데: 변수 선택 + 옵션 (X축, Y축, 그룹, 제목, 기타)
    - 오른쪽: 실시간 미리보기 (FigureCanvas 임베딩)
    - 하단:  확인(결과창 삽입), PNG 저장, 닫기
    """

    # 시그널: 차트 유형, figure, QPixmap
    chart_inserted = Signal(str, object, QPixmap)
    chart_saved = Signal(str)

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._engine = VisualizationEngine()
        self._engine.set_labels(dataset)   # 변수 label·값 label을 차트에 반영
        self._current_fig: Figure | None = None

        self.setWindowTitle("차트 빌더")
        self.setMinimumSize(1200, 800)
        self._setup_ui()
        self._connect_signals()

    # ── UI 구성 ─────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 10)

        # 헤더
        header = self._make_header()
        root.addWidget(header)

        # 메인 3분할 스플리터
        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._make_left_panel())   # 차트 유형
        splitter.addWidget(self._make_center_panel())  # 변수·옵션
        splitter.addWidget(self._make_right_panel())   # 미리보기

        splitter.setSizes([200, 320, 680])
        root.addWidget(splitter, 1)

        # 하단 버튼
        root.addWidget(self._make_bottom_buttons())

    def _make_header(self) -> QFrame:
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background-color: #eaf2fb; border-radius: 6px; }"
        )
        lay = QHBoxLayout(header)
        lay.setContentsMargins(12, 8, 12, 8)
        lbl = QLabel(
            f"<b>{self._dataset.name}</b> &nbsp;|&nbsp; "
            f"케이스: <b>{len(self._dataset.data):,}</b> &nbsp;|&nbsp; "
            f"변수: <b>{len(self._dataset.data.columns)}</b>"
        )
        lbl.setStyleSheet("color: #1a5276; font-size: 13px;")
        lay.addWidget(lbl)
        lay.addStretch()
        return header

    def _make_left_panel(self) -> QWidget:
        """왼쪽 패널: 차트 유형 선택."""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(0, 0, 4, 0)

        grp = QGroupBox("차트 유형")
        grp_lay = QVBoxLayout(grp)

        self._chart_type_group = QButtonGroup(self)
        for i, (key, label, desc) in enumerate(CHART_TYPES):
            btn = QRadioButton(label)
            btn.setProperty("chart_key", key)
            btn.setToolTip(desc)
            self._chart_type_group.addButton(btn, i)
            grp_lay.addWidget(btn)
            if i == 0:
                btn.setChecked(True)

        grp_lay.addStretch()
        lay.addWidget(grp)
        lay.addStretch()
        return widget

    def _make_center_panel(self) -> QWidget:
        """가운데 패널: 변수 선택 + 옵션."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setSpacing(10)

        # 변수 목록
        var_grp = QGroupBox("변수 목록 (더블클릭으로 X축 자동 할당)")
        var_lay = QVBoxLayout(var_grp)
        self._var_list = QListWidget()
        self._var_list.setMaximumHeight(130)
        for col in self._dataset.data.columns:
            dtype = self._dataset.data[col].dtype
            icon = "[수]" if pd.api.types.is_numeric_dtype(dtype) else "[범]"
            item = QListWidgetItem(f"{icon} {self._var_display(col)}")
            item.setData(Qt.UserRole, col)
            self._var_list.addItem(item)
        var_lay.addWidget(self._var_list)
        lay.addWidget(var_grp)

        # 축 변수
        axis_grp = QGroupBox("축 변수")
        axis_lay = QGridLayout(axis_grp)

        axis_lay.addWidget(QLabel("X 축:"), 0, 0)
        self._x_combo = QComboBox()
        self._populate_var_combo(self._x_combo)
        axis_lay.addWidget(self._x_combo, 0, 1)

        axis_lay.addWidget(QLabel("Y 축:"), 1, 0)
        self._y_combo = QComboBox()
        self._populate_var_combo(self._y_combo)
        axis_lay.addWidget(self._y_combo, 1, 1)

        axis_lay.addWidget(QLabel("그룹:"), 2, 0)
        self._group_combo = QComboBox()
        self._populate_var_combo(self._group_combo)
        axis_lay.addWidget(self._group_combo, 2, 1)

        lay.addWidget(axis_grp)

        # 옵션
        opt_grp = QGroupBox("옵션")
        opt_lay = QGridLayout(opt_grp)

        opt_lay.addWidget(QLabel("차트 제목:"), 0, 0)
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("제목 (선택)")
        opt_lay.addWidget(self._title_edit, 0, 1)

        opt_lay.addWidget(QLabel("구간 수 (빈):"), 1, 0)
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(5, 100)
        self._bins_spin.setValue(20)
        opt_lay.addWidget(self._bins_spin, 1, 1)

        self._normal_curve_chk = QCheckBox("정규 분포 곡선 표시")
        self._normal_curve_chk.setChecked(True)
        opt_lay.addWidget(self._normal_curve_chk, 2, 0, 1, 2)

        self._fit_line_chk = QCheckBox("회귀선 표시 (산점도)")
        self._fit_line_chk.setChecked(True)
        opt_lay.addWidget(self._fit_line_chk, 3, 0, 1, 2)

        self._error_bars_chk = QCheckBox("오차 막대 표시 (막대 그래프)")
        self._error_bars_chk.setChecked(True)
        opt_lay.addWidget(self._error_bars_chk, 4, 0, 1, 2)

        lay.addWidget(opt_grp)

        # 미리보기 버튼
        self._preview_btn = QPushButton("미리보기 생성")
        self._preview_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; "
            "font-weight: bold; padding: 10px; font-size: 13px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1f618d; }"
        )
        lay.addWidget(self._preview_btn)
        lay.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _make_right_panel(self) -> QWidget:
        """오른쪽 패널: FigureCanvas 임베딩 미리보기."""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(4, 0, 0, 0)

        lbl = QLabel("차트 미리보기")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50;")
        lay.addWidget(lbl)

        self._canvas = _PreviewCanvas(parent=widget)
        lay.addWidget(self._canvas, 1)
        return widget

    def _make_bottom_buttons(self) -> QWidget:
        """하단 버튼 영역."""
        widget = QWidget()
        lay = QHBoxLayout(widget)
        lay.setContentsMargins(0, 4, 0, 0)

        self._insert_btn = QPushButton("결과창에 삽입")
        self._insert_btn.setEnabled(False)
        self._insert_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; "
            "font-weight: bold; padding: 8px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1e8449; }"
            "QPushButton:disabled { background-color: #bdc3c7; }"
        )

        self._save_png_btn = QPushButton("PNG 저장")
        self._save_png_btn.setEnabled(False)
        self._save_png_btn.setStyleSheet(
            "QPushButton { background-color: #8e44ad; color: white; "
            "font-weight: bold; padding: 8px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #6c3483; }"
            "QPushButton:disabled { background-color: #bdc3c7; }"
        )

        close_btn = QPushButton("닫기")
        close_btn.setStyleSheet(
            "QPushButton { padding: 8px 20px; border-radius: 4px; }"
        )
        close_btn.clicked.connect(self.reject)

        lay.addWidget(self._insert_btn)
        lay.addWidget(self._save_png_btn)
        lay.addStretch()
        lay.addWidget(close_btn)
        return widget

    # ── 시그널 연결 ──────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._preview_btn.clicked.connect(self._generate_preview)
        self._insert_btn.clicked.connect(self._insert_to_output)
        self._save_png_btn.clicked.connect(self._save_png)
        self._var_list.itemDoubleClicked.connect(self._on_var_double_clicked)
        self._chart_type_group.buttonToggled.connect(self._on_chart_type_changed)
        # 차트 유형 초기 반영
        self._on_chart_type_changed()

    def _var_display(self, name: str) -> str:
        """콤보·목록 표시용 — 라벨이 있으면 '변수명 (라벨)', 없으면 변수명.

        라벨이 변수의 실질적 이름이므로 선택 시 라벨을 함께 보여 준다.
        """
        var = self._dataset.variables.get(name) if self._dataset.variables else None
        label = getattr(var, "label", "") if var else ""
        if label and label != name:
            return f"{name} ({label})"
        return name

    def _populate_var_combo(self, combo) -> None:
        """변수 콤보를 '변수명 (라벨)' 표시로 채운다. 실제 값은 컬럼명(userData)."""
        combo.addItem("(없음)", None)
        for name in self._dataset.data.columns:
            combo.addItem(self._var_display(name), name)

    def _on_var_double_clicked(self, item: QListWidgetItem) -> None:
        """변수 목록 더블클릭 → X축에 자동 할당."""
        var = item.data(Qt.UserRole)
        idx = self._x_combo.findData(var)
        if idx >= 0:
            self._x_combo.setCurrentIndex(idx)

    def _on_chart_type_changed(self, *_) -> None:
        """차트 유형 변경 시 옵션 UI 상태 조정."""
        key = self._current_chart_key()
        self._y_combo.setEnabled(key in ("scatter", "line", "boxplot", "bar"))
        self._group_combo.setEnabled(key in ("scatter", "line", "boxplot", "bar"))
        self._bins_spin.setEnabled(key == "histogram")
        self._normal_curve_chk.setEnabled(key in ("histogram", "qq"))
        self._fit_line_chk.setEnabled(key == "scatter")
        self._error_bars_chk.setEnabled(key == "bar")

    def _current_chart_key(self) -> str:
        btn = self._chart_type_group.checkedButton()
        return btn.property("chart_key") if btn else "histogram"

    # ── 미리보기 생성 ────────────────────────────────────────────────────────

    def _generate_preview(self) -> None:
        """선택된 변수·옵션으로 차트를 생성하고 캔버스에 표시."""
        key = self._current_chart_key()
        title = self._title_edit.text().strip()
        df = self._dataset.data

        x = self._x_combo.currentData()
        y = self._y_combo.currentData()
        grp = self._group_combo.currentData()

        try:
            fig = self._build_figure(key, df, x, y, grp, title)
        except Exception as exc:
            logger.exception("차트 생성 오류")
            QMessageBox.critical(self, "오류", f"차트 생성 실패:\n{exc}")
            return

        self._current_fig = fig
        self._canvas.display_figure(fig)
        self._insert_btn.setEnabled(True)
        self._save_png_btn.setEnabled(True)

    def _build_figure(
        self,
        key: str,
        df: pd.DataFrame,
        x: str | None,
        y: str | None,
        grp: str | None,
        title: str,
    ) -> Figure:
        """차트 유형에 따라 VisualizationEngine 메서드를 호출."""
        eng = self._engine

        if key == "histogram":
            if not x:
                raise ValueError("히스토그램은 X 변수를 선택하세요.")
            fig = eng.plot_histogram(df, x,
                                     bins=self._bins_spin.value(),
                                     normal_curve=self._normal_curve_chk.isChecked())
            if title:
                fig.axes[0].set_title(title, fontsize=14, fontweight="bold")

        elif key == "scatter":
            if not x or not y:
                raise ValueError("산점도는 X, Y 변수를 모두 선택하세요.")
            fig = eng.plot_scatter(df, x, y, color_var=grp,
                                   fit_line=self._fit_line_chk.isChecked())
            if title:
                fig.axes[0].set_title(title, fontsize=14, fontweight="bold")

        elif key == "bar":
            if not x:
                raise ValueError("막대 그래프는 X 변수를 선택하세요.")
            fig = eng.plot_bar(df, x, y_var=y,
                               error_bars=self._error_bars_chk.isChecked())
            if title:
                fig.axes[0].set_title(title, fontsize=14, fontweight="bold")

        elif key == "line":
            if not x or not y:
                raise ValueError("선 그래프는 X, Y 변수를 모두 선택하세요.")
            fig = eng.plot_line(df, x, y, by_group=grp)
            if title:
                fig.axes[0].set_title(title, fontsize=14, fontweight="bold")

        elif key == "boxplot":
            if not x:
                raise ValueError("상자 그림은 X 변수를 선택하세요.")
            fig = eng.plot_boxplot(df, x, y_var=y, by_group=(y is not None))
            if title:
                fig.axes[0].set_title(title, fontsize=14, fontweight="bold")

        elif key == "qq":
            if not x:
                raise ValueError("Q-Q 플롯은 X 변수를 선택하세요.")
            fig = eng.plot_qq(df, x)
            if title:
                fig.suptitle(title, fontsize=15, fontweight="bold")

        elif key == "heatmap":
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) < 2:
                raise ValueError("히트맵에는 숫자형 변수가 2개 이상 필요합니다.")
            fig = eng.plot_correlation_heatmap(df, numeric_cols)
            if title:
                fig.axes[0].set_title(title, fontsize=14, fontweight="bold")

        else:
            raise ValueError(f"알 수 없는 차트 유형: {key}")

        return fig

    # ── 결과창 삽입 ──────────────────────────────────────────────────────────

    def _insert_to_output(self) -> None:
        """현재 Figure를 QPixmap으로 변환 후 chart_inserted 시그널 발생."""
        if self._current_fig is None:
            return
        try:
            pixmap = self._engine.fig_to_pixmap(self._current_fig)
            key = self._current_chart_key()
            chart_name = next(
                (label for k, label, _ in CHART_TYPES if k == key), key
            )
            self.chart_inserted.emit(chart_name, self._current_fig, pixmap)
            QMessageBox.information(self, "삽입 완료", "차트가 결과창에 삽입되었습니다.")
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"삽입 실패:\n{exc}")

    # ── PNG 저장 ─────────────────────────────────────────────────────────────

    def _save_png(self) -> None:
        """현재 Figure를 PNG 파일로 저장."""
        if self._current_fig is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "차트 저장", "",
            "PNG 이미지 (*.png);;SVG 이미지 (*.svg);;PDF 문서 (*.pdf)"
        )
        if not path:
            return
        try:
            self._engine.save_figure(self._current_fig, path)
            QMessageBox.information(self, "저장 완료", f"차트가 저장되었습니다:\n{path}")
            self.chart_saved.emit(path)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"저장 실패:\n{exc}")

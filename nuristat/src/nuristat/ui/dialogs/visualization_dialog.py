"""Visualization Dialog — 빠른 시각화 다이얼로그 (완성판).

분석 완료 후 결과 시각화를 즉시 확인하고 저장할 수 있는 다이얼로그.
VisualizationEngine 과 완전 연결.
"""

from __future__ import annotations

import base64
import io as _io
import logging

import matplotlib.pyplot as _plt
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nuristat.analysis.visualization import VisualizationEngine
from nuristat.core.dataset import Dataset

logger = logging.getLogger(__name__)


class VisualizationDialog(QDialog):
    """빠른 시각화 다이얼로그.

    분석 후 바로 호출하여 차트를 즉시 확인할 수 있습니다.
    차트 유형, 변수 선택, 옵션을 설정하고 실제 차트를 생성·저장합니다.
    """

    chart_created = Signal(str, str)   # 차트 유형, base64 이미지

    def __init__(self, dataset: Dataset, parent=None, preset_chart_type: str | None = None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self.engine = VisualizationEngine()
        self.engine.set_labels(dataset)   # 변수 label·값 label을 차트에 반영
        self._current_image: str | None = None  # base64 data-URI

        self.setWindowTitle("시각화")
        self.setMinimumSize(1050, 820)
        self._setup_ui()

        if preset_chart_type:
            for btn in self.chart_type_group.buttons():
                if btn.property("chart_type") == preset_chart_type:
                    btn.setChecked(True)
                    break

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 12)

        # 데이터 정보 헤더
        info_label = QLabel(
            f"{self.dataset.name}: "
            f"{len(self.dataset.data):,}행 x {len(self.dataset.data.columns)}변수  |  "
            f"숫자형: {len(self.dataset.data.select_dtypes(include=['number']).columns)}개"
        )
        info_label.setStyleSheet(
            "font-size: 13px; color: #1a5276; padding: 8px 12px; "
            "background-color: #d4e6f1; border-radius: 4px;"
        )
        layout.addWidget(info_label)

        # 메인 스플리터
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 설정 패널
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        # 차트 유형 선택
        chart_group = QGroupBox("차트 유형")
        chart_layout = QGridLayout(chart_group)

        self.chart_type_group = QButtonGroup(self)
        chart_types = [
            ("막대 차트",   "bar"),
            ("히스토그램", "hist"),
            ("산점도",     "scatter"),
            ("상자 그림",  "box"),
            ("선 차트",    "line"),
            ("히트맵",     "heatmap"),
            ("바이올린",   "violin"),
            ("Q-Q 플롯",   "qq"),
        ]

        for i, (label, value) in enumerate(chart_types):
            btn = QRadioButton(label)
            btn.setProperty("chart_type", value)
            self.chart_type_group.addButton(btn)
            chart_layout.addWidget(btn, i // 2, i % 2)
            if i == 0:
                btn.setChecked(True)

        self.chart_type_group.buttonToggled.connect(self._on_chart_type_changed)
        left_layout.addWidget(chart_group)

        # 변수 선택
        vars_group = QGroupBox("변수 선택")
        vars_layout = QVBoxLayout(vars_group)

        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X 변수:"))
        self.x_combo = QComboBox()
        self._populate_var_combo(self.x_combo, "(선택)")
        x_layout.addWidget(self.x_combo)
        vars_layout.addLayout(x_layout)

        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y 변수:"))
        self.y_combo = QComboBox()
        self._populate_var_combo(self.y_combo, "(선택)")
        y_layout.addWidget(self.y_combo)
        vars_layout.addLayout(y_layout)

        hue_layout = QHBoxLayout()
        hue_layout.addWidget(QLabel("그룹 변수:"))
        self.hue_combo = QComboBox()
        self._populate_var_combo(self.hue_combo, "(없음)")
        hue_layout.addWidget(self.hue_combo)
        vars_layout.addLayout(hue_layout)

        left_layout.addWidget(vars_group)

        # 옵션
        options_group = QGroupBox("옵션")
        options_layout = QVBoxLayout(options_group)

        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("제목:"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("차트 제목 (선택)")
        title_layout.addWidget(self.title_edit)
        options_layout.addLayout(title_layout)

        bins_layout = QHBoxLayout()
        bins_layout.addWidget(QLabel("구간 수 (히스토그램):"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 100)
        self.bins_spin.setValue(20)
        bins_layout.addWidget(self.bins_spin)
        options_layout.addLayout(bins_layout)

        self.kde_check = QCheckBox("정규 곡선 / KDE 표시")
        self.kde_check.setChecked(True)
        options_layout.addWidget(self.kde_check)

        self.reg_check = QCheckBox("회귀선 표시 (산점도)")
        self.reg_check.setChecked(True)
        options_layout.addWidget(self.reg_check)

        self.error_bars_check = QCheckBox("오차 막대 표시 (막대 그래프)")
        self.error_bars_check.setChecked(True)
        options_layout.addWidget(self.error_bars_check)

        left_layout.addWidget(options_group)

        # ── 차트 편집 (생성 후 사용자 수정) ──────────────────────────────────
        edit_group = QGroupBox("차트 편집")
        edit_layout = QVBoxLayout(edit_group)

        xlabel_layout = QHBoxLayout()
        xlabel_layout.addWidget(QLabel("X축 제목:"))
        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setPlaceholderText("비우면 자동")
        xlabel_layout.addWidget(self.xlabel_edit)
        edit_layout.addLayout(xlabel_layout)

        ylabel_layout = QHBoxLayout()
        ylabel_layout.addWidget(QLabel("Y축 제목:"))
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("비우면 자동")
        ylabel_layout.addWidget(self.ylabel_edit)
        edit_layout.addLayout(ylabel_layout)

        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("글꼴 배율:"))
        self.font_scale_spin = QDoubleSpinBox()
        self.font_scale_spin.setRange(0.5, 2.5)
        self.font_scale_spin.setSingleStep(0.1)
        self.font_scale_spin.setValue(1.0)
        font_layout.addWidget(self.font_scale_spin)
        edit_layout.addLayout(font_layout)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("크기(인치) 너비:"))
        self.fig_w_spin = QDoubleSpinBox()
        self.fig_w_spin.setRange(4.0, 20.0)
        self.fig_w_spin.setSingleStep(0.5)
        self.fig_w_spin.setValue(10.0)
        size_layout.addWidget(self.fig_w_spin)
        size_layout.addWidget(QLabel("높이:"))
        self.fig_h_spin = QDoubleSpinBox()
        self.fig_h_spin.setRange(3.0, 16.0)
        self.fig_h_spin.setSingleStep(0.5)
        self.fig_h_spin.setValue(6.0)
        size_layout.addWidget(self.fig_h_spin)
        edit_layout.addLayout(size_layout)

        self.grid_check = QCheckBox("격자 표시")
        self.grid_check.setChecked(True)
        edit_layout.addWidget(self.grid_check)

        self.legend_check = QCheckBox("범례 표시")
        self.legend_check.setChecked(True)
        edit_layout.addWidget(self.legend_check)

        self.stats_check = QCheckBox("통계 상자/상관계수 표시 (산점도·히스토그램)")
        self.stats_check.setChecked(True)
        edit_layout.addWidget(self.stats_check)

        left_layout.addWidget(edit_group)

        # 검증 정보
        self.validation_group = QGroupBox("검증")
        self.validation_layout = QVBoxLayout(self.validation_group)
        self.validation_label = QLabel("변수를 선택하고 차트를 생성하세요")
        self.validation_label.setWordWrap(True)
        self.validation_layout.addWidget(self.validation_label)
        left_layout.addWidget(self.validation_group)

        # 실행 버튼
        btn_layout = QHBoxLayout()
        self.btn_create = QPushButton("차트 생성")
        self.btn_create.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; "
            "font-weight: bold; padding: 10px 20px; font-size: 13px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #155787; }"
        )
        self.btn_create.clicked.connect(self._create_chart)
        btn_layout.addWidget(self.btn_create)

        self.btn_save = QPushButton("저장")
        self.btn_save.setStyleSheet(
            "QPushButton { padding: 10px 20px; border-radius: 4px; }"
        )
        self.btn_save.clicked.connect(self._save_chart)
        self.btn_save.setEnabled(False)
        btn_layout.addWidget(self.btn_save)

        left_layout.addLayout(btn_layout)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # 오른쪽: 미리보기 + 로그
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("차트를 생성하면 여기에 표시됩니다")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #f1f3f4; border: 2px dashed #c0c4cc; "
            "border-radius: 6px; padding: 40px; color: #55555f; font-size: 14px;"
        )
        self.preview_label.setMinimumSize(540, 420)
        right_layout.addWidget(self.preview_label, 1)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(130)
        self.result_text.setStyleSheet(
            "background-color: #1a1a2e; color: #e8e8f0; "
            "font-family: Consolas; font-size: 11px; border-radius: 4px;"
        )
        right_layout.addWidget(self.result_text)

        splitter.addWidget(right_widget)
        splitter.setSizes([360, 680])
        layout.addWidget(splitter, 1)

    # ── 유틸리티 ──────────────────────────────────────────────────────────────

    def _var_display(self, name: str) -> str:
        """콤보 표시용 텍스트 — 라벨이 있으면 '변수명 (라벨)', 없으면 변수명.

        라벨이 변수의 실질적 이름이므로 선택 시 라벨을 함께 보여 준다.
        """
        var = self.dataset.variables.get(name) if self.dataset.variables else None
        label = getattr(var, "label", "") if var else ""
        if label and label != name:
            return f"{name} ({label})"
        return name

    def _populate_var_combo(self, combo, sentinel: str) -> None:
        """변수 콤보를 채운다. 표시는 '변수명 (라벨)', 실제 값은 컬럼명(userData)."""
        combo.addItem(sentinel, None)
        for name in self.dataset.data.columns:
            combo.addItem(self._var_display(name), name)

    def _combo_col(self, combo) -> str | None:
        """콤보에서 선택된 실제 컬럼명을 반환 (sentinel이면 None)."""
        return combo.currentData()

    def _get_selected_chart_type(self) -> str:
        for btn in self.chart_type_group.buttons():
            if btn.isChecked():
                return btn.property("chart_type")
        return "bar"

    def _on_chart_type_changed(self, *_) -> None:
        """차트 유형 변경 시 옵션 활성/비활성."""
        ctype = self._get_selected_chart_type()
        self.y_combo.setEnabled(ctype in ("scatter", "line", "box", "violin"))
        self.hue_combo.setEnabled(ctype in ("bar", "scatter", "line", "box", "violin"))
        self.bins_spin.setEnabled(ctype in ("hist",))
        self.kde_check.setEnabled(ctype in ("hist", "qq"))
        self.reg_check.setEnabled(ctype == "scatter")
        self.error_bars_check.setEnabled(ctype == "bar")

    def _validate_selection(self) -> dict:
        chart_type = self._get_selected_chart_type()
        x = self._combo_col(self.x_combo)
        y = self._combo_col(self.y_combo)
        result = {"valid": True, "errors": [], "warnings": []}

        if chart_type in ("bar", "hist", "qq"):
            if x is None:
                result["valid"] = False
                result["errors"].append("X 변수를 선택하세요")

        elif chart_type in ("scatter", "line"):
            if x is None or y is None:
                result["valid"] = False
                result["errors"].append("X, Y 변수를 모두 선택하세요")

        elif chart_type in ("box", "violin"):
            if x is None:
                result["valid"] = False
                result["errors"].append("X 변수를 선택하세요 (Y는 선택사항)")

        elif chart_type == "heatmap":
            numeric_cols = self.dataset.data.select_dtypes(include=["number"]).columns
            if len(numeric_cols) < 2:
                result["valid"] = False
                result["errors"].append("히트맵에는 숫자형 변수가 2개 이상 필요합니다")

        return result

    # ── 차트 생성 ─────────────────────────────────────────────────────────────

    def _create_chart(self) -> None:
        validation = self._validate_selection()
        if not validation["valid"]:
            self.validation_label.setText("오류: " + "\n".join(validation["errors"]))
            self.validation_label.setStyleSheet("color: #d62728;")
            QMessageBox.warning(self, "검증 오류", "\n".join(validation["errors"]))
            return

        chart_type = self._get_selected_chart_type()
        x = self._combo_col(self.x_combo)
        y = self._combo_col(self.y_combo)
        hue = self._combo_col(self.hue_combo)
        title = self.title_edit.text().strip()
        df = self.dataset.data

        fig = None
        try:
            fig = self._build_figure(chart_type, df, x, y, hue, title)

            # 사용자 편집(축 제목·글꼴 배율·크기·격자·범례) 후처리 적용.
            # 크기는 기본값(10×6)에서 바꿨을 때만 적용 — 히트맵·Q-Q 등 고유 비율 보존.
            w, h = self.fig_w_spin.value(), self.fig_h_spin.value()
            figsize = (w, h) if (w, h) != (10.0, 6.0) else None
            self.engine.apply_edits(
                fig,
                xlabel=self.xlabel_edit.text().strip(),
                ylabel=self.ylabel_edit.text().strip(),
                font_scale=self.font_scale_spin.value(),
                figsize=figsize,
                show_grid=self.grid_check.isChecked(),
                show_legend=self.legend_check.isChecked(),
            )

            # base64로 변환 (300 DPI — 논문 인쇄 품질)
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                        facecolor="white")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            img_uri = f"data:image/png;base64,{b64}"

            # 미리보기 표시
            image = QImage.fromData(base64.b64decode(b64))
            pixmap = QPixmap.fromImage(image)
            self.preview_label.setPixmap(
                pixmap.scaled(self.preview_label.size(),
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            )
            self.preview_label.setStyleSheet("")

            self._current_image = img_uri
            self.btn_save.setEnabled(True)

            self.validation_label.setText("차트 생성 완료")
            self.validation_label.setStyleSheet("color: #2ca02c; font-weight: bold;")
            self.result_text.append(f"[성공] {chart_type} 차트 생성 완료 | {x}")

            self.chart_created.emit(chart_type, img_uri)

        except Exception as exc:
            logger.exception("차트 생성 오류")
            QMessageBox.critical(self, "오류", f"차트 생성 실패:\n{exc}")
            self.result_text.append(f"[오류] {exc}")
        finally:
            if fig is not None:
                _plt.close(fig)  # Figure 객체 즉시 해제 — 반복 생성 시 메모리 누수 방지

    def _build_figure(
        self,
        chart_type: str,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: str | None,
        title: str,
    ):
        """VisualizationEngine 메서드 호출. 사용자 제목은 모든 차트에 적용."""
        eng = self.engine

        if chart_type == "bar":
            fig = eng.plot_bar(df, x, y_var=y,
                               error_bars=self.error_bars_check.isChecked())
        elif chart_type == "hist":
            fig = eng.plot_histogram(df, x,
                                     bins=self.bins_spin.value(),
                                     normal_curve=self.kde_check.isChecked(),
                                     show_stats=self.stats_check.isChecked())
        elif chart_type == "scatter":
            fig = eng.plot_scatter(df, x, y, color_var=hue,
                                   fit_line=self.reg_check.isChecked(),
                                   show_stats=self.stats_check.isChecked())
        elif chart_type == "box":
            fig = eng.plot_boxplot(df, x, y_var=y, by_group=(y is not None))
        elif chart_type == "line":
            fig = eng.plot_line(df, x, y, by_group=hue)
        elif chart_type == "heatmap":
            cols = list(df.select_dtypes(include=[np.number]).columns)
            fig = eng.plot_correlation_heatmap(df, cols)
        elif chart_type == "violin":
            if y is None:
                raise ValueError("바이올린 플롯은 Y 변수가 필요합니다.")
            fig = eng.plot_violin(df, x, y, group_var=hue)
        elif chart_type == "qq":
            fig = eng.plot_qq(df, x)
        else:
            raise ValueError(f"지원하지 않는 차트 유형: {chart_type}")

        # 사용자 지정 제목을 모든 차트 유형에 일관 적용
        if title:
            if chart_type == "qq":
                # Q-Q는 2개 서브플롯 → 전체 제목(suptitle)로 표시
                fig.suptitle(title, fontsize=15, fontweight="bold")
            else:
                fig.axes[0].set_title(title, fontweight="bold")

        return fig

    # ── 저장 ─────────────────────────────────────────────────────────────────

    def _save_chart(self) -> None:
        if not self._current_image:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "차트 저장", "",
            "PNG (*.png);;JPEG (*.jpg);;SVG (*.svg)"
        )
        if path:
            try:
                b64 = self._current_image.split(",", 1)[1]
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))
                self.result_text.append(f"[저장] {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"저장 실패:\n{exc}")

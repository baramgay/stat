"""차트 편집기 대화상자 — 생성된 matplotlib Figure의 텍스트/레이아웃을 후처리 수정."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class ChartEditorDialog(QDialog):
    """matplotlib Figure 텍스트·레이아웃을 편집하는 대화상자.

    Parameters
    ----------
    fig : matplotlib Figure
        편집할 Figure 객체.  ``apply_edits_fn`` 을 통해 수정이 반영된다.
    apply_edits_fn : callable
        ``(fig, **kwargs) -> Figure`` 시그니처의 함수.
        VisualizationEngine.apply_edits 를 전달한다.
    parent : QWidget | None
    """

    def __init__(
        self,
        fig: Any,
        apply_edits_fn: Any,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._fig = fig
        self._apply_edits_fn = apply_edits_fn
        self.setWindowTitle("차트 편집")
        self.setMinimumWidth(380)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # 현재 Figure에서 기존 텍스트 가져오기
        axes = fig.axes
        ax0 = axes[0] if axes else None
        cur_title = ax0.get_title() if ax0 else ""
        cur_xlabel = ax0.get_xlabel() if ax0 else ""
        cur_ylabel = ax0.get_ylabel() if ax0 else ""

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── 텍스트 그룹 ──────────────────────────────────────────────────────
        text_group = QGroupBox("텍스트")
        form = QFormLayout(text_group)
        form.setLabelAlignment(Qt.AlignRight)

        self._title_edit = QLineEdit(cur_title)
        self._xlabel_edit = QLineEdit(cur_xlabel)
        self._ylabel_edit = QLineEdit(cur_ylabel)

        form.addRow("차트 제목:", self._title_edit)
        form.addRow("X축 레이블:", self._xlabel_edit)
        form.addRow("Y축 레이블:", self._ylabel_edit)
        layout.addWidget(text_group)

        # ── 글꼴 크기 배율 ───────────────────────────────────────────────────
        font_group = QGroupBox("글꼴")
        font_form = QFormLayout(font_group)
        font_form.setLabelAlignment(Qt.AlignRight)

        self._font_scale_spin = QDoubleSpinBox()
        self._font_scale_spin.setRange(0.5, 3.0)
        self._font_scale_spin.setSingleStep(0.1)
        self._font_scale_spin.setValue(1.0)
        self._font_scale_spin.setToolTip("1.0 = 기본 크기, 1.2 = 20% 크게")
        font_form.addRow("폰트 배율:", self._font_scale_spin)
        layout.addWidget(font_group)

        # ── 표시 옵션 ────────────────────────────────────────────────────────
        display_group = QGroupBox("표시 옵션")
        disp_layout = QVBoxLayout(display_group)

        self._grid_check = QCheckBox("격자 표시")
        self._grid_check.setChecked(True)
        self._legend_check = QCheckBox("범례 표시")
        self._legend_check.setChecked(True)

        disp_layout.addWidget(self._grid_check)
        disp_layout.addWidget(self._legend_check)
        layout.addWidget(display_group)

        # ── 안내 ─────────────────────────────────────────────────────────────
        hint = QLabel("빈 칸으로 두면 해당 항목은 변경되지 않습니다.")
        hint.setStyleSheet("color: #44475a; font-size: 11px;")
        layout.addWidget(hint)

        # ── 버튼 ─────────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
        )
        buttons.button(QDialogButtonBox.Ok).setText("적용")
        buttons.button(QDialogButtonBox.Cancel).setText("취소")
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def edited_figure(self) -> Any:
        """편집이 반영된 Figure 를 반환한다."""
        return self._fig

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def _apply(self) -> None:
        title = self._title_edit.text().strip()
        xlabel = self._xlabel_edit.text().strip()
        ylabel = self._ylabel_edit.text().strip()
        font_scale = self._font_scale_spin.value()
        show_grid = self._grid_check.isChecked()
        show_legend = self._legend_check.isChecked()

        try:
            self._fig = self._apply_edits_fn(
                self._fig,
                title=title or None,
                xlabel=xlabel or None,
                ylabel=ylabel or None,
                font_scale=font_scale,
                show_grid=show_grid,
                show_legend=show_legend,
            )
        except Exception:
            pass
        self.accept()

"""Tests for OutputWindow — tab creation, add_analysis_result, clipboard copy."""

from __future__ import annotations

import pandas as pd
import pytest

from nuristat.analysis.result import AnalysisResult, ResultTable
from nuristat.ui.output_window import OutputWindow


def _make_result(title: str = "빈도분석") -> AnalysisResult:
    df = pd.DataFrame({"값": ["A", "B", "C"], "빈도": [10, 20, 30]})
    tbl = ResultTable(title="빈도표", dataframe=df)
    return AnalysisResult(id="test-1", title=title, tables=[tbl])


# -------------------------------------------------------------------------
# 탭 생성
# -------------------------------------------------------------------------

class TestAddAnalysisResult:
    def test_tab_added_on_result(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        assert win.tab_widget.count() == 1  # 로그 탭만

        result = _make_result("빈도분석")
        win.add_analysis_result(result)

        assert win.tab_widget.count() == 2

    def test_result_stored_in_list(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        r1 = _make_result("분석1")
        r2 = _make_result("분석2")
        win.add_analysis_result(r1)
        win.add_analysis_result(r2)

        assert len(win._results) == 2
        assert win._results[0].title == "분석1"
        assert win._results[1].title == "분석2"

    def test_current_tab_switches_to_new(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        win.add_analysis_result(_make_result("첫번째"))
        win.add_analysis_result(_make_result("두번째"))

        assert win.tab_widget.currentIndex() == 2  # 두 번째 분석 탭

    def test_log_tab_not_closable(self, qtbot):
        from PySide6.QtWidgets import QTabBar
        win = OutputWindow()
        qtbot.addWidget(win)
        # 로그 탭(index 0)은 close 버튼 없음
        close_btn = win.tab_widget.tabBar().tabButton(0, QTabBar.ButtonPosition.RightSide)
        assert close_btn is None

    def test_multiple_results(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        for i in range(5):
            win.add_analysis_result(_make_result(f"분석{i}"))
        assert win.tab_widget.count() == 6  # 로그 + 5


# -------------------------------------------------------------------------
# 로그 탭 (add_output 호환)
# -------------------------------------------------------------------------

class TestAddOutput:
    def test_log_output_does_not_add_tab(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        win.add_output("분석 성공", "success")
        win.add_output("경고 메시지", "warning")

        assert win.tab_widget.count() == 1  # 로그 탭만

    def test_log_lines_accumulate(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        win.add_output("라인1", "text")
        win.add_output("라인2", "text")
        assert len(win._log_lines) == 2


# -------------------------------------------------------------------------
# 탭 닫기
# -------------------------------------------------------------------------

class TestTabClose:
    def test_close_removes_result(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        win.add_analysis_result(_make_result("결과1"))
        win.add_analysis_result(_make_result("결과2"))
        assert len(win._results) == 2

        win._on_tab_close(1)  # 첫 번째 분석 탭 닫기

        assert len(win._results) == 1
        assert win.tab_widget.count() == 2  # 로그 + 결과2

    def test_close_log_tab_is_noop(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        win._on_tab_close(0)  # 로그 탭은 닫기 불가
        assert win.tab_widget.count() == 1


# -------------------------------------------------------------------------
# 지우기
# -------------------------------------------------------------------------

class TestClearOutput:
    def test_clear_removes_all_analysis_tabs(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        for i in range(3):
            win.add_analysis_result(_make_result(f"결과{i}"))
        win.add_output("로그", "text")

        win.clear_output()

        assert win.tab_widget.count() == 1  # 로그 탭만 남음
        assert win._results == []
        assert win._log_lines == []


# -------------------------------------------------------------------------
# 클립보드 복사
# -------------------------------------------------------------------------

class TestClipboardCopy:
    def test_copy_tables_sets_html_and_text(self, qtbot, qapp):
        win = OutputWindow()
        qtbot.addWidget(win)
        result = _make_result("빈도분석")

        win._copy_tables(result)

        mime = qapp.clipboard().mimeData()
        assert mime.hasHtml()
        assert mime.hasText()
        html = mime.html()
        assert "<table" in html

    def test_copy_tables_text_tab_separated(self, qtbot, qapp):
        win = OutputWindow()
        qtbot.addWidget(win)
        result = _make_result("빈도분석")

        win._copy_tables_text(result)

        text = qapp.clipboard().text()
        assert "\t" in text or "값" in text  # 탭구분 CSV 또는 컬럼명 포함


# -------------------------------------------------------------------------
# HTML 내보내기 (파일 시스템 없이 내용만 검증)
# -------------------------------------------------------------------------

class TestHtmlWrap:
    def test_wrap_html_includes_title(self, qtbot):
        win = OutputWindow()
        qtbot.addWidget(win)
        html = win._wrap_html("<p>결과내용</p>", "빈도분석", "12:34:56")
        assert "빈도분석" in html
        assert "12:34:56" in html
        assert "결과내용" in html

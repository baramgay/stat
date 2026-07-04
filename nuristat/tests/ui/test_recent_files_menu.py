"""최근 파일 메뉴 검증 (SPSS: 파일 > 최근 사용 데이터).

백엔드(SettingsManager.add_recent_file 등)는 완비돼 있었으나 메뉴·연동이
없어 도달 불가능하던 기능을 노출 — 열기/가져오기/저장 시 자동 기록 + 재열기.

담당 에이전트: frontend, tester-unit
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(app):
    from nuristat.ui.main_window import MainWindow
    w = MainWindow()
    # 테스트 격리: 최근 파일 목록 비우기
    w._settings.clear_recent_files()
    w._rebuild_recent_menu()
    return w   # close()는 미저장 확인 모달 때문에 호출하지 않음


class TestRecentMenu:
    def test_empty_shows_placeholder(self, window):
        window._settings.clear_recent_files()
        window._rebuild_recent_menu()
        acts = window._recent_menu.actions()
        assert len(acts) == 1
        assert not acts[0].isEnabled()   # "(없음)"

    def test_remember_adds_entry(self, window, tmp_path):
        p = tmp_path / "sample.csv"
        p.write_text("a,b\n1,2\n")
        window._remember_recent(str(p))
        texts = [a.text() for a in window._recent_menu.actions()]
        assert any("sample.csv" in t for t in texts)

    def test_most_recent_first(self, window, tmp_path):
        window._settings.clear_recent_files()
        a = tmp_path / "a.csv"
        a.write_text("x\n1\n")
        b = tmp_path / "b.sav"
        b.write_text("dummy")
        window._remember_recent(str(a))
        window._remember_recent(str(b))
        texts = [t.text() for t in window._recent_menu.actions() if t.isEnabled()]
        assert "b.sav" in texts[0]   # 최근 항목이 맨 위
        assert "a.csv" in texts[1]

    def test_clear_recent(self, window, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("x\n1\n")
        window._remember_recent(str(p))
        window._clear_recent()
        acts = window._recent_menu.actions()
        assert len(acts) == 1 and not acts[0].isEnabled()

    def test_open_recent_missing_file_warns(self, window, monkeypatch):
        """존재하지 않는 경로는 경고만(예외 없음)."""
        warned = {}
        import nuristat.ui.main_window as mw
        monkeypatch.setattr(mw.QMessageBox, "warning",
                            lambda *a, **k: warned.setdefault("hit", True))
        window._open_recent("C:/no/such/file.csv")
        assert warned.get("hit") is True

    def test_open_recent_csv_roundtrip(self, window, app, tmp_path):
        """실제 CSV를 최근 파일로 다시 열면 데이터가 로드된다(백그라운드 완료 대기)."""
        p = tmp_path / "rt.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
        window._open_recent(str(p))

        worker = window._file_task_worker
        if worker is not None:
            worker.wait(3000)
        for _ in range(20):
            app.processEvents()

        assert window.current_dataset is not None
        assert list(window.current_dataset.data.columns) == ["a", "b"]
        # 재열기 후 최근 목록에 기록됨
        texts = [a.text() for a in window._recent_menu.actions()]
        assert any("rt.csv" in t for t in texts)

"""SettingsManager 단위 테스트.

QSettings를 FakeQSettings로 대체하여 레지스트리/파일 I/O 없이 테스트합니다.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import QPoint, QSize


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_qsettings_cls():
    """QSettings를 인메모리 FakeQSettings로 교체하는 픽스처."""
    storage: dict = {}

    class FakeQSettings:
        def __init__(self, *args):
            self._storage = storage
            self._group = ""

        def setValue(self, key, value):
            self._storage[key] = value

        def value(self, key, default=None):
            return self._storage.get(key, default)

        def beginGroup(self, group):
            self._group = group

        def endGroup(self):
            self._group = ""

        def childKeys(self):
            prefix = self._group + "/" if self._group else ""
            return [
                k[len(prefix):]
                for k in self._storage
                if k.startswith(prefix) and "/" not in k[len(prefix):]
            ]

        def clear(self):
            self._storage.clear()

    with patch("statworkbench.core.settings.QSettings", FakeQSettings):
        yield FakeQSettings, storage


@pytest.fixture
def manager(mock_qsettings_cls):
    """패치된 QSettings를 사용하는 SettingsManager 인스턴스."""
    from statworkbench.core.settings import SettingsManager
    return SettingsManager()


# ─────────────────────────────────────────────────────────────────────────────
# 1. 윈도우 크기/위치 왕복
# ─────────────────────────────────────────────────────────────────────────────

class TestWindowGeometry:
    def test_save_and_load_window_size(self, manager):
        """저장한 QSize를 그대로 불러온다."""
        size = QSize(1280, 720)
        pos = QPoint(50, 50)
        manager.save_window_geometry(size, pos)
        assert manager.load_window_size() == size

    def test_save_and_load_window_position(self, manager):
        """저장한 QPoint를 그대로 불러온다."""
        size = QSize(800, 600)
        pos = QPoint(200, 300)
        manager.save_window_geometry(size, pos)
        assert manager.load_window_position() == pos

    def test_save_geometry_sets_maximized_false(self, manager):
        """save_window_geometry 호출 시 maximized가 False로 저장된다."""
        manager.save_window_geometry(QSize(1024, 768), QPoint(0, 0))
        assert manager.load_window_maximized() is False

    def test_load_window_size_default(self, manager):
        """저장된 값이 없으면 기본값 QSize(1400, 900)을 반환한다."""
        assert manager.load_window_size() == QSize(1400, 900)

    def test_load_window_position_default(self, manager):
        """저장된 값이 없으면 기본값 QPoint(100, 100)을 반환한다."""
        assert manager.load_window_position() == QPoint(100, 100)

    def test_overwrite_window_size(self, manager):
        """동일 키에 두 번 저장하면 마지막 값이 남는다."""
        manager.save_window_geometry(QSize(800, 600), QPoint(10, 10))
        manager.save_window_geometry(QSize(1920, 1080), QPoint(20, 20))
        assert manager.load_window_size() == QSize(1920, 1080)

    def test_overwrite_window_position(self, manager):
        """위치도 마지막으로 저장한 값이 남는다."""
        manager.save_window_geometry(QSize(800, 600), QPoint(10, 10))
        manager.save_window_geometry(QSize(800, 600), QPoint(300, 400))
        assert manager.load_window_position() == QPoint(300, 400)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 최대화 상태 왕복
# ─────────────────────────────────────────────────────────────────────────────

class TestWindowMaximized:
    def test_save_maximized_true(self, manager):
        """True 저장 후 True를 반환한다."""
        manager.save_window_maximized(True)
        assert manager.load_window_maximized() is True

    def test_save_maximized_false(self, manager):
        """False 저장 후 False를 반환한다."""
        manager.save_window_maximized(True)
        manager.save_window_maximized(False)
        assert manager.load_window_maximized() is False

    def test_load_maximized_default(self, manager):
        """저장된 값이 없으면 기본값 False를 반환한다."""
        assert manager.load_window_maximized() is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. 테마 설정 왕복
# ─────────────────────────────────────────────────────────────────────────────

class TestTheme:
    def test_save_and_load_dark_mode_true(self, manager):
        """다크 모드 True 왕복."""
        manager.save_theme(True)
        assert manager.load_theme() is True

    def test_save_and_load_dark_mode_false(self, manager):
        """라이트 모드 False 왕복."""
        manager.save_theme(False)
        assert manager.load_theme() is False

    def test_load_theme_default(self, manager):
        """저장된 값이 없으면 기본값 False(라이트)를 반환한다."""
        assert manager.load_theme() is False

    def test_overwrite_theme(self, manager):
        """테마를 두 번 저장하면 마지막 값이 남는다."""
        manager.save_theme(True)
        manager.save_theme(False)
        assert manager.load_theme() is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. 최근 파일 목록
# ─────────────────────────────────────────────────────────────────────────────

class TestRecentFiles:
    def test_add_recent_file_appears_in_list(self, manager, tmp_path):
        """추가한 파일이 목록 맨 앞에 있다."""
        f = tmp_path / "a.csv"
        f.touch()
        manager.add_recent_file(str(f))
        assert manager.load_recent_files()[0] == str(f)

    def test_add_duplicate_moves_to_top(self, manager, tmp_path):
        """이미 있는 파일을 추가하면 맨 앞으로 이동하고 중복은 제거된다."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.touch(); f2.touch()
        manager.add_recent_file(str(f1))
        manager.add_recent_file(str(f2))
        manager.add_recent_file(str(f1))  # 중복 추가
        files = manager.load_recent_files()
        assert files[0] == str(f1)
        assert files.count(str(f1)) == 1

    def test_max_10_recent_files(self, manager, tmp_path):
        """11개 추가하면 최신 10개만 유지한다."""
        paths = []
        for i in range(11):
            p = tmp_path / f"file{i}.csv"
            p.touch()
            paths.append(str(p))
            manager.add_recent_file(str(p))
        assert len(manager.load_recent_files()) == 10

    def test_max_10_keeps_most_recent(self, manager, tmp_path):
        """11번째 파일이 목록 맨 앞, 첫 번째 파일은 제거된다."""
        paths = []
        for i in range(11):
            p = tmp_path / f"file{i}.csv"
            p.touch()
            paths.append(str(p))
            manager.add_recent_file(str(p))
        files = manager.load_recent_files()
        assert files[0] == paths[10]      # 가장 최근
        assert paths[0] not in files      # 가장 오래된 것은 제거

    def test_load_recent_files_filters_nonexistent(self, manager, tmp_path):
        """실제로 존재하지 않는 파일은 필터링된다."""
        existing = tmp_path / "real.csv"
        existing.touch()
        manager._settings.setValue("recent/files", ["/nonexistent/ghost.csv", str(existing)])
        files = manager.load_recent_files()
        assert str(existing) in files
        assert "/nonexistent/ghost.csv" not in files

    def test_load_recent_files_empty_when_none_saved(self, manager):
        """저장된 파일이 없으면 빈 리스트를 반환한다."""
        assert manager.load_recent_files() == []

    def test_clear_recent_files(self, manager, tmp_path):
        """clear_recent_files 호출 후 목록이 빈 리스트다."""
        f = tmp_path / "a.csv"
        f.touch()
        manager.add_recent_file(str(f))
        manager.clear_recent_files()
        # clear 후 빈 리스트가 저장돼야 함 (존재 파일 없어도 빈 리스트)
        assert manager._settings.value("recent/files") == []


# ─────────────────────────────────────────────────────────────────────────────
# 5. 분석 기본값
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisDefaults:
    def test_save_and_load_analysis_defaults(self, manager):
        """저장한 분석 기본값을 그대로 불러온다."""
        defaults = {"alpha": 0.05, "method": "pearson", "iterations": 1000}
        manager.save_analysis_defaults("correlation", defaults)
        loaded = manager.load_analysis_defaults("correlation")
        assert loaded["alpha"] == 0.05
        assert loaded["method"] == "pearson"
        assert loaded["iterations"] == 1000

    def test_analysis_defaults_empty_when_not_saved(self, manager):
        """저장된 기본값이 없으면 빈 딕셔너리를 반환한다."""
        assert manager.load_analysis_defaults("regression") == {}

    def test_analysis_defaults_different_types_isolated(self, manager):
        """분석 유형이 다르면 서로 격리된다."""
        manager.save_analysis_defaults("t_test", {"alpha": 0.01})
        manager.save_analysis_defaults("anova", {"alpha": 0.05})
        assert manager.load_analysis_defaults("t_test")["alpha"] == 0.01
        assert manager.load_analysis_defaults("anova")["alpha"] == 0.05

    def test_analysis_defaults_overwrite(self, manager):
        """동일 분석 유형에 다시 저장하면 값이 덮어쓰인다."""
        manager.save_analysis_defaults("regression", {"method": "ols"})
        manager.save_analysis_defaults("regression", {"method": "ridge"})
        loaded = manager.load_analysis_defaults("regression")
        assert loaded["method"] == "ridge"


# ─────────────────────────────────────────────────────────────────────────────
# 6. 데이터 뷰 설정
# ─────────────────────────────────────────────────────────────────────────────

class TestDataViewSettings:
    def test_save_and_load_data_view_settings(self, manager):
        """저장한 데이터 뷰 설정을 그대로 불러온다."""
        settings = {"rows_per_page": 50, "show_index": True, "font_size": 12}
        manager.save_data_view_settings(settings)
        loaded = manager.load_data_view_settings()
        assert loaded["rows_per_page"] == 50
        assert loaded["show_index"] is True
        assert loaded["font_size"] == 12

    def test_data_view_settings_empty_when_not_saved(self, manager):
        """저장된 설정이 없으면 빈 딕셔너리를 반환한다."""
        assert manager.load_data_view_settings() == {}

    def test_data_view_settings_partial_update(self, manager):
        """일부 키만 다시 저장해도 기존 키는 유지된다."""
        manager.save_data_view_settings({"rows_per_page": 50, "show_index": True})
        manager.save_data_view_settings({"rows_per_page": 100})
        loaded = manager.load_data_view_settings()
        assert loaded["rows_per_page"] == 100
        assert loaded["show_index"] is True

    def test_data_view_settings_string_value(self, manager):
        """문자열 값도 올바르게 저장·불러온다."""
        manager.save_data_view_settings({"sort_column": "이름", "sort_order": "asc"})
        loaded = manager.load_data_view_settings()
        assert loaded["sort_column"] == "이름"
        assert loaded["sort_order"] == "asc"


# ─────────────────────────────────────────────────────────────────────────────
# 7. clear_all
# ─────────────────────────────────────────────────────────────────────────────

class TestClearAll:
    def test_clear_all_empties_storage(self, manager, mock_qsettings_cls):
        """clear_all 호출 후 내부 storage가 비어 있다."""
        _, storage = mock_qsettings_cls
        manager.save_theme(True)
        manager.save_window_maximized(True)
        assert len(storage) > 0
        manager.clear_all()
        assert storage == {}

    def test_clear_all_resets_theme_to_default(self, manager):
        """clear_all 후 테마 기본값(False)이 반환된다."""
        manager.save_theme(True)
        manager.clear_all()
        assert manager.load_theme() is False

    def test_clear_all_resets_window_size_to_default(self, manager):
        """clear_all 후 윈도우 크기 기본값이 반환된다."""
        manager.save_window_geometry(QSize(800, 600), QPoint(0, 0))
        manager.clear_all()
        assert manager.load_window_size() == QSize(1400, 900)

    def test_clear_all_resets_window_position_to_default(self, manager):
        """clear_all 후 윈도우 위치 기본값이 반환된다."""
        manager.save_window_geometry(QSize(800, 600), QPoint(500, 500))
        manager.clear_all()
        assert manager.load_window_position() == QPoint(100, 100)

    def test_clear_all_resets_analysis_defaults(self, manager):
        """clear_all 후 분석 기본값이 빈 딕셔너리다."""
        manager.save_analysis_defaults("t_test", {"alpha": 0.05})
        manager.clear_all()
        assert manager.load_analysis_defaults("t_test") == {}

    def test_clear_all_resets_data_view_settings(self, manager):
        """clear_all 후 데이터 뷰 설정이 빈 딕셔너리다."""
        manager.save_data_view_settings({"rows_per_page": 50})
        manager.clear_all()
        assert manager.load_data_view_settings() == {}

    def test_clear_all_can_be_called_twice_safely(self, manager):
        """clear_all을 두 번 호출해도 예외가 발생하지 않는다."""
        manager.clear_all()
        manager.clear_all()  # 예외 없어야 함

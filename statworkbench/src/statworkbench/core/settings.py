"""Settings Manager — 사용자 설정 관리자.

QSettings를 사용하여 사용자 환경 설정을 저장하고 불러옵니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QSettings, QSize, QPoint

logger = logging.getLogger(__name__)


class SettingsManager:
    """사용자 설정 관리자.
    
    Features:
    - 윈도우 위치/크기 저장
    - 테마 설정 저장
    - 최근 파일 목록
    - 사용자 기본값
    """
    
    def __init__(self) -> None:
        self._settings = QSettings("StatWorkbench", "StatWorkbench")
    
    # ── 윈도우 설정 ────────────────────────────────────────────────────────
    
    def save_window_geometry(self, size: QSize, pos: QPoint) -> None:
        """윈도우 크기와 위치 저장."""
        self._settings.setValue("window/size", size)
        self._settings.setValue("window/position", pos)
        self._settings.setValue("window/maximized", False)
    
    def save_window_maximized(self, maximized: bool) -> None:
        """최대화 상태 저장."""
        self._settings.setValue("window/maximized", maximized)
    
    def load_window_size(self) -> QSize:
        """저장된 윈도우 크기 반환."""
        return self._settings.value("window/size", QSize(1400, 900))
    
    def load_window_position(self) -> QPoint:
        """저장된 윈도우 위치 반환."""
        return self._settings.value("window/position", QPoint(100, 100))
    
    def load_window_maximized(self) -> bool:
        """저장된 최대화 상태 반환."""
        return self._settings.value("window/maximized", False)
    
    # ── 테마 설정 ──────────────────────────────────────────────────────────
    
    def save_theme(self, dark_mode: bool) -> None:
        """테마 설정 저장."""
        self._settings.setValue("theme/dark_mode", dark_mode)
    
    def load_theme(self) -> bool:
        """저장된 테마 설정 반환."""
        return self._settings.value("theme/dark_mode", False)
    
    # ── 최근 파일 ──────────────────────────────────────────────────────────
    
    def add_recent_file(self, path: str) -> None:
        """최근 파일 목록에 추가."""
        recent_files = self.load_recent_files()
        
        # 중복 제거 및 최대 10개 유지
        if path in recent_files:
            recent_files.remove(path)
        recent_files.insert(0, path)
        recent_files = recent_files[:10]
        
        self._settings.setValue("recent/files", recent_files)
    
    def load_recent_files(self) -> list[str]:
        """최근 파일 목록 반환."""
        files = self._settings.value("recent/files", [])
        if files is None:
            return []
        return [f for f in files if Path(f).exists()]
    
    def clear_recent_files(self) -> None:
        """최근 파일 목록 초기화."""
        self._settings.setValue("recent/files", [])
    
    # ── 분석 기본값 ────────────────────────────────────────────────────────
    
    def save_analysis_defaults(self, analysis_type: str, defaults: dict[str, Any]) -> None:
        """분석 기본값 저장."""
        for key, value in defaults.items():
            self._settings.setValue(f"analysis/{analysis_type}/{key}", value)
    
    def load_analysis_defaults(self, analysis_type: str) -> dict[str, Any]:
        """분석 기본값 불러오기."""
        self._settings.beginGroup(f"analysis/{analysis_type}")
        keys = self._settings.childKeys()
        self._settings.endGroup()
        
        defaults = {}
        for key in keys:
            defaults[key] = self._settings.value(f"analysis/{analysis_type}/{key}")
        return defaults
    
    # ── 데이터 뷰 설정 ─────────────────────────────────────────────────────
    
    def save_data_view_settings(self, settings: dict[str, Any]) -> None:
        """데이터 뷰 설정 저장."""
        for key, value in settings.items():
            self._settings.setValue(f"data_view/{key}", value)
    
    def load_data_view_settings(self) -> dict[str, Any]:
        """데이터 뷰 설정 불러오기."""
        self._settings.beginGroup("data_view")
        keys = self._settings.childKeys()
        self._settings.endGroup()
        
        settings = {}
        for key in keys:
            settings[key] = self._settings.value(f"data_view/{key}")
        return settings
    
    # ── 전체 초기화 ────────────────────────────────────────────────────────
    
    def clear_all(self) -> None:
        """모든 설정 초기화."""
        self._settings.clear()
        logger.info("모든 설정이 초기화되었습니다.")

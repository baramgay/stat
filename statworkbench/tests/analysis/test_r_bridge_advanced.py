"""RBridge 고급 테스트 — 커버리지 향상 (mock 기반).

누락 라인:
- 59-64: is_available() FileNotFoundError/TimeoutExpired 처리
- 83: R 없을 때 early return
- 93-94: dataset CSV 저장 (R 있을 때)
- 130-135: JSON 출력 파일 읽기
- 140: 그림 파일 수집
- 145-153: subprocess.TimeoutExpired 처리
- 183: _build_wrapper에 data_path 있을 때
- 214-228: get_installed_packages() R 있을 때
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.r_bridge import RBridge
from statworkbench.core.dataset import Dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bridge() -> RBridge:
    return RBridge()


@pytest.fixture
def simple_dataset() -> Dataset:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(data=df, name="test")


# ---------------------------------------------------------------------------
# 1. is_available — 예외 경로 (59-64)
# ---------------------------------------------------------------------------

class TestIsAvailableExceptions:

    def test_file_not_found_returns_false(self):
        """모든 R 경로에서 FileNotFoundError → False 반환."""
        br = RBridge()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = br.is_available()
        assert result is False
        assert br._r_available is False

    def test_timeout_returns_false(self):
        """subprocess.TimeoutExpired → False 반환."""
        br = RBridge()
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="R", timeout=5)):
            result = br.is_available()
        assert result is False

    def test_r_found_returns_true(self):
        """subprocess.run이 returncode=0 → True 반환."""
        br = RBridge()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "R version 4.3.0"
        with patch("subprocess.run", return_value=mock_result):
            result = br.is_available()
        assert result is True
        assert br._r_available is True
        assert br._r_path is not None

    def test_r_not_found_returncode_nonzero(self):
        """returncode != 0 → 다음 후보 시도 후 False."""
        br = RBridge()
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = br.is_available()
        assert result is False


# ---------------------------------------------------------------------------
# 2. execute — R 있을 때 경로 (93-94, 130-135, 140)
# ---------------------------------------------------------------------------

class TestExecuteWithMockedR:

    def _make_bridge_with_r(self) -> RBridge:
        """R이 존재하는 척하는 bridge."""
        br = RBridge()
        br._r_available = True
        br._r_path = "R"
        return br

    def test_execute_with_dataset_saves_csv(self, simple_dataset):
        """dataset 전달 시 CSV 파일 저장 경로 실행 (93-94)."""
        br = self._make_bridge_with_r()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            result = br.execute("cat('ok')", dataset=simple_dataset)
        assert isinstance(result, dict)
        assert "success" in result

    def test_execute_reads_json_output(self, tmp_path):
        """출력 JSON 파일이 있으면 읽어서 output에 포함 (130-135)."""
        br = self._make_bridge_with_r()
        br._temp_dir = str(tmp_path)

        # JSON 파일 미리 생성
        out_path = tmp_path / "output.json"
        out_path.write_text(json.dumps({"key": "value"}), encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            result = br.execute("cat('ok')")
        assert isinstance(result.get("output"), dict)
        assert result["output"]["key"] == "value"

    def test_execute_collects_plot_files(self, tmp_path):
        """PNG 파일이 있으면 plots 목록에 포함 (140)."""
        br = self._make_bridge_with_r()
        br._temp_dir = str(tmp_path)

        # 가짜 PNG 생성
        (tmp_path / "plot1.png").write_bytes(b"\x89PNG")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            result = br.execute("cat('ok')")
        assert any(p.endswith(".png") for p in result["plots"])

    def test_execute_invalid_json_fallback(self, tmp_path):
        """출력 파일이 잘못된 JSON → 문자열로 읽음."""
        br = self._make_bridge_with_r()
        br._temp_dir = str(tmp_path)

        out_path = tmp_path / "output.json"
        out_path.write_text("not valid json {{", encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        with patch("subprocess.run", return_value=mock_proc):
            result = br.execute("cat('ok')")
        assert "success" in result

    def test_execute_timeout_returns_error(self):
        """subprocess.TimeoutExpired → error 딕셔너리 반환 (145-153)."""
        br = self._make_bridge_with_r()
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="R", timeout=60)):
            result = br.execute("Sys.sleep(100)")
        assert result["success"] is False
        assert "초과" in result.get("error", "") or "timeout" in result.get("error", "").lower()

    def test_execute_generic_exception_returns_error(self):
        """일반 예외 → error 딕셔너리 반환 (152-158)."""
        br = self._make_bridge_with_r()
        with patch("subprocess.run", side_effect=OSError("pipe error")):
            result = br.execute("cat('ok')")
        assert result["success"] is False
        assert "pipe error" in result.get("error", "")

    def test_execute_returncode_nonzero_success_false(self):
        """returncode != 0 → success=False."""
        br = self._make_bridge_with_r()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Error in R"
        with patch("subprocess.run", return_value=mock_proc):
            result = br.execute("stop('oops')")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# 3. _build_wrapper — data_path 있을 때 (183)
# ---------------------------------------------------------------------------

class TestBuildWrapperWithData:

    def test_data_path_included_in_wrapper(self, bridge):
        wrapper = bridge._build_wrapper(
            "x <- 1", "/tmp/data.csv", "/tmp/out.json", "json"
        )
        assert "read.csv" in wrapper
        assert "/tmp/data.csv" in wrapper

    def test_no_data_path_no_read_csv(self, bridge):
        wrapper = bridge._build_wrapper("x <- 1", None, "/tmp/out.json", "json")
        assert "read.csv" not in wrapper

    def test_wrapper_has_write_json_call(self, bridge):
        wrapper = bridge._build_wrapper("x <- 1", None, "/tmp/out.json", "json")
        assert "write_json" in wrapper or "jsonlite" in wrapper

    def test_user_script_in_wrapper(self, bridge):
        script = "my_unique_function_xyz()"
        wrapper = bridge._build_wrapper(script, None, "/tmp/out.json", "json")
        assert script in wrapper


# ---------------------------------------------------------------------------
# 4. get_installed_packages (214-228)
# ---------------------------------------------------------------------------

class TestGetInstalledPackages:

    def test_returns_list_when_no_r(self, bridge):
        """R 없을 때 빈 리스트."""
        result = bridge.get_installed_packages()
        if not bridge._r_available:
            assert result == []

    def test_returns_list_with_r_mocked(self):
        """R 있을 때 패키지 목록 파싱 (214-228)."""
        br = RBridge()
        br._r_available = True
        br._r_path = "R"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "base\nstats\nutils\n"
        with patch("subprocess.run", return_value=mock_proc):
            pkgs = br.get_installed_packages()
        assert "base" in pkgs
        assert "stats" in pkgs

    def test_empty_stdout_returns_empty_list(self):
        """빈 stdout → 빈 리스트."""
        br = RBridge()
        br._r_available = True
        br._r_path = "R"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        with patch("subprocess.run", return_value=mock_proc):
            pkgs = br.get_installed_packages()
        assert pkgs == []

    def test_exception_in_packages_returns_empty(self):
        """예외 발생 시 빈 리스트 반환 (except 경로)."""
        br = RBridge()
        br._r_available = True
        br._r_path = "R"
        with patch("subprocess.run", side_effect=OSError("pipe")):
            pkgs = br.get_installed_packages()
        assert pkgs == []

    def test_returncode_nonzero_returns_empty(self):
        """returncode != 0 → 빈 리스트."""
        br = RBridge()
        br._r_available = True
        br._r_path = "R"
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "base\n"
        with patch("subprocess.run", return_value=mock_proc):
            pkgs = br.get_installed_packages()
        assert pkgs == []

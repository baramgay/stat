"""R Bridge — R 스크립트 실행 엔진.

R이 설치된 환경에서 R 스크립트를 실행하고 결과를 파싱합니다.
R이 없는 환경에서는 mock 결과를 반환합니다.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any

from nuristat.core.dataset import Dataset

logger = logging.getLogger(__name__)


class RBridge:
    """R 연동 브리지.

    Features:
    - R 스크립트 실행 (subprocess)
    - DataFrame 전달 (CSV 임시 파일)
    - 결과 파싱 (JSON, 표, 그림)
    - R 패키지 자동 설치 확인
    """

    def __init__(self) -> None:
        self._r_available: bool | None = None
        self._r_path: str | None = None
        self._temp_dir = tempfile.mkdtemp(prefix="nuristat_r_")

    def is_available(self) -> bool:
        """R 실행 환경이 있는지 확인."""
        if self._r_available is not None:
            return self._r_available

        # R 실행 파일 찾기
        r_candidates = ["R", "R.exe", "/usr/bin/R", "/usr/local/bin/R"]
        for cmd in r_candidates:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self._r_path = cmd
                    self._r_available = True
                    logger.info(f"R found: {cmd}")
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        self._r_available = False
        logger.warning("R not found. R integration disabled.")
        return False

    def execute(
        self,
        script: str,
        dataset: Dataset | None = None,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """R 스크립트를 실행합니다.

        Args:
            script: 실행할 R 스크립트
            dataset: 입력 데이터셋 (선택)
            output_format: 출력 형식 (json, table, plot)

        Returns:
            실행 결과 딕셔너리
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "R이 설치되지 않았습니다. R을 설치하거나 Python 분석을 사용하세요.",
                "output": "",
                "plots": [],
            }

        # 데이터셋 저장
        data_path = None
        if dataset is not None and dataset.data is not None:
            data_path = os.path.join(self._temp_dir, "input_data.csv")
            dataset.data.to_csv(data_path, index=False, encoding="utf-8")

        # 출력 파일 경로
        output_path = os.path.join(self._temp_dir, "output.json")
        plot_paths = []

        # R 스크립트 래퍼
        wrapper_script = self._build_wrapper(script, data_path, output_path, output_format)

        # R 스크립트 임시 파일
        script_path = os.path.join(self._temp_dir, "script.R")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapper_script)

        # 실행
        assert self._r_path is not None
        try:
            result = subprocess.run(
                [self._r_path, "--vanilla", "--slave", "-f", script_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self._temp_dir,
            )

            # 결과 파싱
            output = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output": "",
                "plots": [],
            }

            # JSON 출력 읽기
            if os.path.exists(output_path):
                with open(output_path, encoding="utf-8") as f:
                    try:
                        json_output = json.load(f)
                        output["output"] = json_output
                    except json.JSONDecodeError:
                        output["output"] = f.read()

            # 그림 파일 수집
            for fname in os.listdir(self._temp_dir):
                if fname.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
                    plot_paths.append(os.path.join(self._temp_dir, fname))
            output["plots"] = plot_paths

            return output

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "R 스크립트 실행 시간 초과 (60초)",
                "output": "",
                "plots": [],
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "output": "",
                "plots": [],
            }

    def _build_wrapper(
        self,
        script: str,
        data_path: str | None,
        output_path: str,
        output_format: str,
    ) -> str:
        """R 스크립트 래퍼를 생성합니다."""
        lines = [
            '# NuriStat R Bridge Wrapper',
            'options(encoding = "UTF-8")',
            'options(warn = 1)',
            '',
            '# 필요한 패키지 로드',
            'suppressPackageStartupMessages({',
            '  if (!require("jsonlite", quietly = TRUE)) install.packages("jsonlite", repos = "https://cloud.r-project.org/")',
            '  library(jsonlite)',
            '})',
            '',
        ]

        # 데이터 로드 (Windows 경로 → R 호환 슬래시)
        if data_path:
            r_data_path = data_path.replace("\\", "/")
            lines.extend([
                '# 데이터 로드',
                f'df <- read.csv("{r_data_path}", stringsAsFactors = FALSE, encoding = "UTF-8")',
                'cat("데이터 로드 완료:", nrow(df), "행 x", ncol(df), "열\\n")',
                '',
            ])

        # 사용자 스크립트
        lines.extend([
            '# --- 사용자 스크립트 시작 ---',
            script,
            '# --- 사용자 스크립트 종료 ---',
            '',
        ])

        # 결과 저장 (Windows 경로 → R 호환 슬래시)
        if output_format == "json":
            r_output_path = output_path.replace("\\", "/")
            lines.extend([
                '# 결과를 JSON으로 저장',
                'result <- list(',
                '  success = TRUE,',
                '  message = "R 스크립트 실행 완료",',
                '  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")',
                ')',
                f'write_json(result, "{r_output_path}", auto_unbox = TRUE, pretty = TRUE)',
            ])

        return "\n".join(lines)

    def get_installed_packages(self) -> list[str]:
        """설치된 R 패키지 목록을 반환합니다."""
        if not self.is_available():
            return []

        assert self._r_path is not None
        try:
            result = subprocess.run(
                [self._r_path, "--vanilla", "--slave", "-e", "cat(paste(rownames(installed.packages()), collapse='\\n'))"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        except Exception:
            pass
        return []

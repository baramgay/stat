"""Python Bridge — Python 스크립트 실행 엔진.

안전한 샌드박스 환경에서 Python 스크립트를 실행합니다.
"""

from __future__ import annotations

import ast
import logging
import os
import sys
import tempfile
import traceback
from io import StringIO
from typing import Any

import pandas as pd

from nuristat.core.dataset import Dataset

logger = logging.getLogger(__name__)


class PythonBridge:
    """Python 연동 브리지.

    Features:
    - 안전한 샌드박스 실행 (제한된 내장 함수)
    - DataFrame 직접 전달
    - matplotlib/seaborn/plotly 시각화 지원
    - 결과 자동 직렬화
    """

    # 허용된 내장 함수
    ALLOWED_BUILTINS = {
        "abs", "all", "any", "bin", "bool", "bytearray", "bytes",
        "chr", "complex", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "hasattr", "hash", "hex",
        "int", "isinstance", "issubclass", "iter", "len", "list",
        "map", "max", "min", "next", "oct", "ord", "pow", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "str",
        "sum", "tuple", "type", "zip",
        # pandas/numpy
        "pd", "np",
    }

    # 금지된 구문
    FORBIDDEN_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )

    def __init__(self) -> None:
        self._temp_dir = tempfile.mkdtemp(prefix="nuristat_py_")
        self._execution_count = 0

    def execute(
        self,
        script: str,
        dataset: Dataset | None = None,
        output_format: str = "auto",
    ) -> dict[str, Any]:
        """Python 스크립트를 안전하게 실행합니다.

        Args:
            script: 실행할 Python 스크립트
            dataset: 입력 데이터셋 (선택)
            output_format: 출력 형식 (auto, json, plot)

        Returns:
            실행 결과 딕셔너리
        """
        # 보안 검증
        security_check = self._validate_script(script)
        if not security_check["safe"]:
            return {
                "success": False,
                "error": f"보안 위반: {security_check['reason']}",
                "output": "",
                "plots": [],
            }

        # 실행 환경 준비
        exec_globals = self._prepare_globals(dataset)
        exec_locals: dict[str, Any] = {}

        # stdout/stderr 캡처
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        try:
            # 스크립트 컴파일 및 실행
            compiled = compile(script, "<user_script>", "exec")
            exec(compiled, exec_globals, exec_locals)

            # 결과 수집
            output: dict[str, Any] = {
                "success": True,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "variables": {},
                "plots": [],
            }

            # 주요 변수 수집
            for key, value in exec_locals.items():
                if key.startswith("_"):
                    continue
                try:
                    if isinstance(value, pd.DataFrame):
                        output["variables"][key] = {
                            "type": "DataFrame",
                            "shape": value.shape,
                            "preview": value.head(5).to_dict(),
                        }
                    elif isinstance(value, (int, float, str, bool, list, dict)):
                        output["variables"][key] = {
                            "type": type(value).__name__,
                            "value": value,
                        }
                except Exception:
                    pass

            # 그림 파일 수집
            plot_dir = exec_globals.get("_plot_dir", self._temp_dir)
            if os.path.exists(plot_dir):
                for fname in os.listdir(plot_dir):
                    if fname.endswith((".png", ".jpg", ".jpeg", ".svg", ".html")):
                        output["plots"].append(os.path.join(plot_dir, fname))

            return output

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "plots": [],
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _validate_script(self, script: str) -> dict[str, Any]:
        """스크립트 보안 검증."""
        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            return {"safe": False, "reason": f"구문 오류: {exc}"}

        for node in ast.walk(tree):
            if isinstance(node, self.FORBIDDEN_NODES):
                node_type = type(node).__name__
                return {"safe": False, "reason": f"금지된 구문: {node_type}"}

            if isinstance(node, ast.Call):
                # 위험한 함수 호출 검사
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("open", "exec", "eval", "compile", "__import__"):
                        return {"safe": False, "reason": f"금지된 함수 호출: {node.func.id}"}

                # getattr으로 우회하는 경우 검사
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "call", "run"):
                        return {"safe": False, "reason": f"금지된 메서드 호출: {node.func.attr}"}

        return {"safe": True, "reason": ""}

    def _prepare_globals(self, dataset: Dataset | None) -> dict[str, Any]:
        """실행 환경 전역 변수 준비."""
        import matplotlib
        import numpy as np
        matplotlib.use("Agg")  # 헤드리스 모드
        import matplotlib.pyplot as plt

        # 그림 저장 디렉토리
        plot_dir = os.path.join(self._temp_dir, f"plots_{self._execution_count}")
        os.makedirs(plot_dir, exist_ok=True)
        self._execution_count += 1

        # 기본 전역 변수
        globals_dict = {
            "__builtins__": {name: __builtins__[name] for name in self.ALLOWED_BUILTINS if name in __builtins__} if isinstance(__builtins__, dict) else {name: getattr(__builtins__, name) for name in self.ALLOWED_BUILTINS if hasattr(__builtins__, name)},  # type: ignore[index]
            "pd": pd,
            "np": np,
            "plt": plt,
            "_plot_dir": plot_dir,
        }

        # 데이터셋 전달
        if dataset is not None and dataset.data is not None:
            globals_dict["df"] = dataset.data.copy()
            globals_dict["dataset"] = dataset

        # 시각화 헬퍼 함수
        globals_dict["save_plot"] = lambda name: plt.savefig(
            os.path.join(plot_dir, name),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )

        return globals_dict

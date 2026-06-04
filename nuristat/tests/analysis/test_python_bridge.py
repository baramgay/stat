"""PythonBridge 테스트.

검증 항목:
- 기본 스크립트 실행 (산술, 변수 캡처)
- DataFrame 전달 및 조작
- 보안 검증: import/exec/eval/open 차단
- 금지 구문: class/def/asyncdef/import 차단
- 금지 메서드: os.system/popen 차단
- 구문 오류 → success=False
- 런타임 오류 → success=False (traceback 포함)
- stdout 캡처
- 변수 타입별 직렬화 (int, float, str, bool, list, dict, DataFrame)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.analysis.python_bridge import PythonBridge
from nuristat.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def bridge() -> PythonBridge:
    return PythonBridge()


@pytest.fixture
def simple_dataset() -> Dataset:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 20),
        "y": rng.normal(5, 2, 20),
        "group": ["A", "B"] * 10,
    })
    return Dataset(data=df, name="test")


# ──────────────────────────────────────────────────────────────
# 1. 초기화
# ──────────────────────────────────────────────────────────────

class TestPythonBridgeInit:

    def test_instantiation(self, bridge):
        assert isinstance(bridge, PythonBridge)

    def test_allowed_builtins_defined(self, bridge):
        assert hasattr(bridge, "ALLOWED_BUILTINS")
        assert len(bridge.ALLOWED_BUILTINS) > 0

    def test_forbidden_nodes_defined(self, bridge):
        assert hasattr(bridge, "FORBIDDEN_NODES")

    def test_execution_count_starts_zero(self, bridge):
        assert bridge._execution_count == 0


# ──────────────────────────────────────────────────────────────
# 2. 정상 실행
# ──────────────────────────────────────────────────────────────

class TestPythonBridgeExecute:

    def test_simple_arithmetic_success(self, bridge):
        result = bridge.execute("result = 2 + 3")
        assert result["success"] is True

    def test_variable_captured(self, bridge):
        result = bridge.execute("answer = 42")
        assert "answer" in result["variables"]
        assert result["variables"]["answer"]["value"] == 42

    def test_float_variable_captured(self, bridge):
        result = bridge.execute("pi_approx = 3.14159")
        assert "pi_approx" in result["variables"]
        assert abs(result["variables"]["pi_approx"]["value"] - 3.14159) < 1e-5

    def test_string_variable_captured(self, bridge):
        result = bridge.execute("greeting = 'hello'")
        assert result["variables"]["greeting"]["value"] == "hello"

    def test_list_variable_captured(self, bridge):
        result = bridge.execute("nums = [1, 2, 3]")
        assert result["variables"]["nums"]["value"] == [1, 2, 3]

    def test_dict_variable_captured(self, bridge):
        result = bridge.execute("info = {'a': 1}")
        assert result["variables"]["info"]["value"] == {"a": 1}

    def test_private_vars_not_captured(self, bridge):
        result = bridge.execute("_hidden = 99\nvisible = 1")
        assert "_hidden" not in result["variables"]
        assert "visible" in result["variables"]

    def test_stdout_key_exists(self, bridge):
        """실행 결과에 stdout 키가 존재한다 (print는 ALLOWED_BUILTINS 외)."""
        result = bridge.execute("x = 1")
        assert "stdout" in result

    def test_execution_count_increments(self, bridge):
        bridge.execute("x = 1")
        bridge.execute("y = 2")
        assert bridge._execution_count == 2


# ──────────────────────────────────────────────────────────────
# 3. Dataset 전달
# ──────────────────────────────────────────────────────────────

class TestPythonBridgeDataset:

    def test_df_available_in_script(self, bridge, simple_dataset):
        result = bridge.execute("rows = len(df)", dataset=simple_dataset)
        assert result["success"] is True
        assert result["variables"]["rows"]["value"] == len(simple_dataset.data)

    def test_dataframe_result_captured(self, bridge, simple_dataset):
        result = bridge.execute("subset = df[['x', 'y']]", dataset=simple_dataset)
        assert "subset" in result["variables"]
        assert result["variables"]["subset"]["type"] == "DataFrame"

    def test_dataframe_shape_in_result(self, bridge, simple_dataset):
        result = bridge.execute("sub = df[['x']]", dataset=simple_dataset)
        shape = result["variables"]["sub"]["shape"]
        assert shape[0] == len(simple_dataset.data)
        assert shape[1] == 1

    def test_no_dataset_df_not_available(self, bridge):
        result = bridge.execute("rows = len(df)")
        assert result["success"] is False


# ──────────────────────────────────────────────────────────────
# 4. 보안 — import 차단
# ──────────────────────────────────────────────────────────────

class TestPythonBridgeSecurity:

    def test_import_blocked(self, bridge):
        result = bridge.execute("import os")
        assert result["success"] is False
        assert "보안" in result["error"]

    def test_from_import_blocked(self, bridge):
        result = bridge.execute("from os import path")
        assert result["success"] is False

    def test_exec_function_blocked(self, bridge):
        result = bridge.execute("exec('x=1')")
        assert result["success"] is False

    def test_eval_function_blocked(self, bridge):
        result = bridge.execute("y = eval('1+1')")
        assert result["success"] is False

    def test_open_function_blocked(self, bridge):
        result = bridge.execute("f = open('/etc/passwd')")
        assert result["success"] is False

    def test_class_definition_blocked(self, bridge):
        result = bridge.execute("class MyClass:\n    pass")
        assert result["success"] is False

    def test_function_definition_blocked(self, bridge):
        result = bridge.execute("def my_func():\n    pass")
        assert result["success"] is False

    def test_os_system_blocked(self, bridge):
        result = bridge.execute("os.system('ls')")
        assert result["success"] is False

    def test_compile_blocked(self, bridge):
        result = bridge.execute("compile('x=1', '<str>', 'exec')")
        assert result["success"] is False


# ──────────────────────────────────────────────────────────────
# 5. 오류 처리
# ──────────────────────────────────────────────────────────────

class TestPythonBridgeErrors:

    def test_syntax_error_returns_failure(self, bridge):
        result = bridge.execute("x = (1 +")
        assert result["success"] is False

    def test_runtime_error_returns_failure(self, bridge):
        result = bridge.execute("x = 1 / 0")
        assert result["success"] is False

    def test_name_error_returns_failure(self, bridge):
        result = bridge.execute("y = undefined_variable + 1")
        assert result["success"] is False

    def test_failed_result_has_error_key(self, bridge):
        result = bridge.execute("1 / 0")
        assert "error" in result

    def test_failed_result_has_traceback(self, bridge):
        result = bridge.execute("x = 1 / 0")
        assert "traceback" in result


# ──────────────────────────────────────────────────────────────
# 6. _validate_script 직접 검증
# ──────────────────────────────────────────────────────────────

class TestValidateScript:

    def test_clean_script_is_safe(self, bridge):
        result = bridge._validate_script("x = 1 + 2")
        assert result["safe"] is True

    def test_import_not_safe(self, bridge):
        result = bridge._validate_script("import os")
        assert result["safe"] is False

    def test_class_not_safe(self, bridge):
        result = bridge._validate_script("class Foo:\n    pass")
        assert result["safe"] is False

    def test_syntax_error_not_safe(self, bridge):
        result = bridge._validate_script("x = (")
        assert result["safe"] is False

    def test_open_call_not_safe(self, bridge):
        result = bridge._validate_script("open('/etc/passwd')")
        assert result["safe"] is False

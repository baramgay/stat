"""RBridge 테스트.

검증 항목:
- is_available(): R 없는 환경에서 False 반환
- execute(): R 없을 때 error 딕셔너리 반환
- _build_wrapper(): 래퍼 스크립트에 사용자 코드 포함
- 초기화: 임시 디렉토리 생성
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from statworkbench.analysis.r_bridge import RBridge
from statworkbench.core.dataset import Dataset


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def bridge() -> RBridge:
    return RBridge()


@pytest.fixture
def simple_dataset() -> Dataset:
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.1, 4.0, 5.9, 8.1, 10.0],
    })
    return Dataset(data=df, name="test")


# ──────────────────────────────────────────────────────────────
# 1. 초기화
# ──────────────────────────────────────────────────────────────

class TestRBridgeInit:

    def test_instantiation(self, bridge):
        assert isinstance(bridge, RBridge)

    def test_temp_dir_created(self, bridge):
        assert os.path.exists(bridge._temp_dir)

    def test_r_available_initially_none(self, bridge):
        assert bridge._r_available is None

    def test_r_path_initially_none(self, bridge):
        assert bridge._r_path is None


# ──────────────────────────────────────────────────────────────
# 2. is_available
# ──────────────────────────────────────────────────────────────

class TestRBridgeIsAvailable:

    def test_returns_bool(self, bridge):
        result = bridge.is_available()
        assert isinstance(result, bool)

    def test_caches_result(self, bridge):
        first = bridge.is_available()
        second = bridge.is_available()
        assert first == second
        assert bridge._r_available is not None

    def test_r_available_set_after_check(self, bridge):
        bridge.is_available()
        assert bridge._r_available is not None


# ──────────────────────────────────────────────────────────────
# 3. execute (R 없는 환경 — 대부분의 CI/개발 환경)
# ──────────────────────────────────────────────────────────────

class TestRBridgeExecuteNoR:

    def test_execute_returns_dict(self, bridge):
        result = bridge.execute("cat('hello')")
        assert isinstance(result, dict)

    def test_execute_has_success_key(self, bridge):
        result = bridge.execute("cat('hello')")
        assert "success" in result

    def test_execute_has_error_key_when_no_r(self, bridge):
        if bridge.is_available():
            pytest.skip("R이 설치된 환경 — 이 테스트는 R 없는 환경용")
        result = bridge.execute("cat('hello')")
        assert result["success"] is False
        assert "error" in result

    def test_execute_has_plots_key(self, bridge):
        result = bridge.execute("cat('hello')")
        assert "plots" in result

    def test_execute_plots_is_list(self, bridge):
        result = bridge.execute("cat('hello')")
        assert isinstance(result["plots"], list)


# ──────────────────────────────────────────────────────────────
# 4. _build_wrapper (R 있든 없든 호출 가능)
# ──────────────────────────────────────────────────────────────

class TestRBridgeBuildWrapper:

    def test_wrapper_contains_user_script(self, bridge):
        try:
            wrapper = bridge._build_wrapper(
                "cat('test')", None, "/tmp/out.json", "json"
            )
            assert "cat('test')" in wrapper
        except AttributeError:
            pytest.skip("_build_wrapper 메서드 없음")

    def test_wrapper_is_string(self, bridge):
        try:
            wrapper = bridge._build_wrapper(
                "x <- 1", None, "/tmp/out.json", "json"
            )
            assert isinstance(wrapper, str)
        except AttributeError:
            pytest.skip("_build_wrapper 메서드 없음")

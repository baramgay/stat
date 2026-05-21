"""AuditLog 테스트.

검증 항목:
- append / to_list / clear / __len__
- 세부 정보(details) 포함·미포함 엔트리
- timestamp UTC ISO 형식
- save_jsonl / load_jsonl 왕복 저장
- details 딕셔너리 독립 복사 (참조 공유 없음)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from statworkbench.core.audit import AuditLog


# ──────────────────────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def log() -> AuditLog:
    return AuditLog()


# ──────────────────────────────────────────────────────────────
# 1. 기본 동작
# ──────────────────────────────────────────────────────────────

class TestAuditLogBasic:

    def test_initial_length_zero(self, log):
        assert len(log) == 0

    def test_repr(self, log):
        assert "AuditLog" in repr(log)
        assert "0" in repr(log)

    def test_append_increments_length(self, log):
        log.append("action_a")
        assert len(log) == 1
        log.append("action_b")
        assert len(log) == 2

    def test_to_list_returns_copy(self, log):
        log.append("x")
        lst = log.to_list()
        lst.append({"injected": True})
        assert len(log) == 1

    def test_to_list_empty(self, log):
        assert log.to_list() == []

    def test_clear(self, log):
        log.append("a")
        log.append("b")
        log.clear()
        assert len(log) == 0
        assert log.to_list() == []


# ──────────────────────────────────────────────────────────────
# 2. 엔트리 구조
# ──────────────────────────────────────────────────────────────

class TestAuditLogEntryStructure:

    def test_entry_has_timestamp(self, log):
        log.append("op")
        entry = log.to_list()[0]
        assert "timestamp" in entry

    def test_entry_has_action(self, log):
        log.append("variable_rename")
        entry = log.to_list()[0]
        assert entry["action"] == "variable_rename"

    def test_entry_without_details_has_no_details_key(self, log):
        log.append("op")
        entry = log.to_list()[0]
        assert "details" not in entry

    def test_entry_with_details(self, log):
        log.append("analysis_run", {"procedure": "t_test", "n": 30})
        entry = log.to_list()[0]
        assert entry["details"]["procedure"] == "t_test"
        assert entry["details"]["n"] == 30

    def test_details_dict_is_independent_copy(self, log):
        """외부 딕셔너리 변경이 저장된 엔트리에 영향을 주지 않아야 한다."""
        d = {"key": "original"}
        log.append("op", d)
        d["key"] = "mutated"
        entry = log.to_list()[0]
        assert entry["details"]["key"] == "original"

    def test_timestamp_is_iso_format(self, log):
        from datetime import datetime
        log.append("ts_check")
        ts = log.to_list()[0]["timestamp"]
        # ISO 8601 파싱 성공 여부로 검증
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_multiple_entries_ordered(self, log):
        for i in range(5):
            log.append(f"action_{i}")
        entries = log.to_list()
        actions = [e["action"] for e in entries]
        assert actions == [f"action_{i}" for i in range(5)]


# ──────────────────────────────────────────────────────────────
# 3. JSON Lines 저장 / 불러오기
# ──────────────────────────────────────────────────────────────

class TestAuditLogJsonl:

    def test_save_and_load_roundtrip(self, log, tmp_path):
        log.append("op1", {"x": 1})
        log.append("op2")
        path = tmp_path / "audit.jsonl"
        log.save_jsonl(str(path))

        loaded = AuditLog.load_jsonl(str(path))
        assert len(loaded) == 2
        entries = loaded.to_list()
        assert entries[0]["action"] == "op1"
        assert entries[0]["details"]["x"] == 1
        assert entries[1]["action"] == "op2"

    def test_save_creates_parent_dirs(self, log, tmp_path):
        log.append("op")
        nested = tmp_path / "a" / "b" / "c" / "audit.jsonl"
        log.save_jsonl(str(nested))
        assert nested.exists()

    def test_saved_file_is_valid_jsonl(self, log, tmp_path):
        log.append("op", {"k": "v"})
        path = tmp_path / "audit.jsonl"
        log.save_jsonl(str(path))

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["action"] == "op"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            AuditLog.load_jsonl("/nonexistent/path/audit.jsonl")

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        loaded = AuditLog.load_jsonl(str(path))
        assert len(loaded) == 0

    def test_save_unicode(self, log, tmp_path):
        log.append("변수_이름_변경", {"이전": "age", "이후": "나이"})
        path = tmp_path / "audit.jsonl"
        log.save_jsonl(str(path))
        loaded = AuditLog.load_jsonl(str(path))
        entry = loaded.to_list()[0]
        assert entry["action"] == "변수_이름_변경"
        assert entry["details"]["이전"] == "age"

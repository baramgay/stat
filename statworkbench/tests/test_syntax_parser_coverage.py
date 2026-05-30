"""syntax/parser.py 커버리지 보강 테스트.

미커버 라인:
  47     : `continue` — 빈 블록 건너뜀
  51-53  : ValueError → continue (parse_single 실패)
  72     : parse_single("") → raise ValueError("Empty command text")
  109    : _extract_command_name — match 없음 → raise ValueError
  147    : value.endswith(".") → value[:-1].strip()
  214    : 마지막 블록(period 없음) → blocks.append
"""

from __future__ import annotations

import pytest

from statworkbench.syntax.parser import SyntaxParser


@pytest.fixture
def parser():
    return SyntaxParser()


# ---------------------------------------------------------------------------
# Line 47: 빈 블록 → continue
# ---------------------------------------------------------------------------

class TestEmptyBlock:

    def test_blank_lines_between_commands_skipped(self, parser):
        """명령어 사이 빈 줄 → continue(47)."""
        syntax = "FREQUENCIES VARIABLES=age.\n\n\nDESCRIPTIVES VARIABLES=score."
        cmds = parser.parse(syntax)
        assert len(cmds) == 2

    def test_only_whitespace_block(self, parser):
        """전체가 공백 → 빈 블록 건너뜀."""
        cmds = parser.parse("   \n   \n   ")
        assert cmds == []


# ---------------------------------------------------------------------------
# Lines 51-53: parse_single ValueError → parse에서 continue
# ---------------------------------------------------------------------------

class TestParseValueError:

    def test_unparsable_block_skipped(self, parser):
        """숫자로 시작하는 블록 → ValueError → continue(51-53)."""
        # '123 INVALID.' → 명령어 이름이 숫자로 시작 → ValueError
        syntax = "123 INVALID.\nFREQUENCIES VARIABLES=age."
        cmds = parser.parse(syntax)
        # 두 번째 정상 명령은 파싱됨
        assert any(c.command == "FREQUENCIES" for c in cmds)


# ---------------------------------------------------------------------------
# Line 72: parse_single("") → raise ValueError
# ---------------------------------------------------------------------------

class TestParseSingleEmpty:

    def test_empty_string_raises(self, parser):
        """parse_single('') → ValueError(72)."""
        with pytest.raises(ValueError, match="Empty command text"):
            parser.parse_single("")

    def test_whitespace_string_raises(self, parser):
        """parse_single('  ') → ValueError(72)."""
        with pytest.raises(ValueError):
            parser.parse_single("   ")


# ---------------------------------------------------------------------------
# Line 109: _extract_command_name no match → raise ValueError
# ---------------------------------------------------------------------------

class TestExtractCommandNameFail:

    def test_non_alpha_start_raises(self, parser):
        """'123...' → 명령어 이름 추출 실패 → ValueError(109)."""
        with pytest.raises(ValueError):
            parser.parse_single("123 INVALID_COMMAND")

    def test_special_char_start_raises(self, parser):
        """'@CMD' → ValueError(109)."""
        with pytest.raises(ValueError):
            parser.parse_single("@CMD VARIABLES=x")


# ---------------------------------------------------------------------------
# Line 147: value ending with '.' → stripped
# ---------------------------------------------------------------------------

class TestValueWithTrailingPeriod:

    def test_key_value_trailing_period_stripped(self, parser):
        """'KEY=value.' → value에서 마지막 '.' 제거(147)."""
        cmd = parser.parse_single("DESCRIPTIVES VARIABLES=age.")
        # VARIABLES 값에서 trailing period 제거되어야 함
        assert "." not in cmd.parameters.get("VARIABLES", "")


# ---------------------------------------------------------------------------
# Line 214: 마지막 블록 period 없음 → blocks.append
# ---------------------------------------------------------------------------

class TestBlockWithoutTrailingPeriod:

    def test_command_without_period(self, parser):
        """period 없는 명령 → 마지막 블록으로 추가(214)."""
        syntax = "FREQUENCIES VARIABLES=age"  # period 없음
        cmds = parser.parse(syntax)
        assert len(cmds) == 1
        assert cmds[0].command == "FREQUENCIES"

    def test_mixed_period_and_no_period(self, parser):
        """period 있는 명령 + period 없는 명령 혼합."""
        syntax = "FREQUENCIES VARIABLES=age.\nDESCRIPTIVES VARIABLES=score"
        cmds = parser.parse(syntax)
        assert len(cmds) == 2

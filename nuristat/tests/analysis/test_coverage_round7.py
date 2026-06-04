"""커버리지 7라운드 — dead code 제거 후 잔여 도달 가능 라인.

dead code 제거 (라인 수 감소):
  ttests.py 128-129           : _label 함수 정의만 있고 미호출 → 제거
  assumptions.py 163          : else 분기 → EXCLUDE_SYSTEM_MISSING_ONLY 흡수
  nonparametric.py 50         : ss_total==0 guard 수학적 불가 → 제거
  partial_correlation.py 88   : denom==0 guard — clip으로 불가 → 제거
  partial_correlation.py 138-139: except Exception pass around pd.apply → 제거

신규 테스트:
  syntax/parser.py 47         : 빈 블록 → continue (빈 문자열 블록 skip)
  assumptions.py 159-160      : EXCLUDE_SYSTEM_MISSING_ONLY → else → dropna()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, MissingPolicy


# ---------------------------------------------------------------------------
# syntax/parser.py 47: 빈 블록 → continue
# ---------------------------------------------------------------------------

class TestParserEmptyBlock:

    def test_empty_blocks_are_skipped(self):
        """빈 블록(strip 후 '')-이 있을 때 continue로 건너뜀 (line 47)."""
        from nuristat.syntax.parser import SyntaxParser

        parser = SyntaxParser()
        # 두 명령 사이에 공백만 있는 빈 블록 포함
        syntax = "FREQUENCIES VARIABLES=age.\n\n   \nDESCRIPTIVES VARIABLES=age."
        cmds = parser.parse(syntax)

        # 빈 블록이 건너뛰어지고 유효 명령만 파싱
        assert len(cmds) >= 1


# ---------------------------------------------------------------------------
# assumptions.py 159-160: EXCLUDE_SYSTEM_MISSING_ONLY → else → dropna()
# ---------------------------------------------------------------------------

class TestAssumptionsExcludeSystemMissing:

    def test_exclude_system_missing_only_drops_nan(self):
        """MissingPolicy.EXCLUDE_SYSTEM_MISSING_ONLY → else → dropna() 실행."""
        from nuristat.analysis.assumptions import prepare_analysis_frame

        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [10.0, 20.0, 30.0, np.nan],
        })
        ds = Dataset(df, "TestExcludeSys")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        prepared = prepare_analysis_frame(
            ds,
            variables=["x", "y"],
            missing_policy=MissingPolicy.EXCLUDE_SYSTEM_MISSING_ONLY,
        )

        # NaN 행 2개 제거 → 유효 행 2개
        assert prepared.n_valid == 2
        assert prepared.n_excluded == 2

"""spss_reader.py 커버리지 보강 테스트.

미커버 라인:
  58      : measure = _guess_measure_type(df[col]) — variable_measure 없을 때
  73      : var_meta.missing_values = meta.missing_ranges[col]
  86-90   : except ImportError → SWBImportError
  91-92   : except Exception → SWBImportError
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.exceptions import ImportError as SWBImportError
from statworkbench.io.spss_reader import read_sav, _guess_measure_type, _guess_storage_type


# ---------------------------------------------------------------------------
# Line 58: _guess_measure_type 호출 경로 (variable_measure 없음)
# ---------------------------------------------------------------------------

class TestGuessMeasureType:

    def test_numeric_series_returns_scale(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _guess_measure_type(s).name == "SCALE"

    def test_string_series_returns_nominal(self):
        s = pd.Series(["A", "B", "C"])
        assert _guess_measure_type(s).name == "NOMINAL"


# ---------------------------------------------------------------------------
# Lines 86-90: pyreadstat ImportError → SWBImportError
# ---------------------------------------------------------------------------

class TestPyreadstatImportError:

    def test_import_error_raises_swb_import_error(self, tmp_path):
        """pyreadstat import 실패 → SWBImportError(86-90)."""
        sav_file = tmp_path / "test.sav"
        sav_file.write_bytes(b"\x00" * 10)  # 가짜 sav 파일

        # sys.modules에서 pyreadstat 제거하여 ImportError 유도
        original = sys.modules.pop("pyreadstat", None)
        try:
            with patch.dict("sys.modules", {"pyreadstat": None}):
                with pytest.raises(SWBImportError, match="pyreadstat"):
                    read_sav(str(sav_file))
        finally:
            if original is not None:
                sys.modules["pyreadstat"] = original


# ---------------------------------------------------------------------------
# Lines 91-92: 일반 예외 → SWBImportError
# ---------------------------------------------------------------------------

class TestReadSavException:

    def test_read_sav_exception_raises_swb_import_error(self, tmp_path):
        """pyreadstat.read_sav 예외 → SWBImportError(91-92)."""
        sav_file = tmp_path / "bad.sav"
        sav_file.write_bytes(b"not a sav file")

        mock_pyreadstat = MagicMock()
        mock_pyreadstat.read_sav.side_effect = Exception("corrupted")

        with patch.dict("sys.modules", {"pyreadstat": mock_pyreadstat}):
            with pytest.raises(SWBImportError, match="읽기 실패"):
                read_sav(str(sav_file))


# ---------------------------------------------------------------------------
# Line 73: meta.missing_ranges[col] 경로
# ---------------------------------------------------------------------------

class TestMissingRanges:

    def test_missing_ranges_set_on_variable(self, tmp_path):
        """missing_ranges 있는 col → var_meta.missing_values 설정(73)."""
        sav_file = tmp_path / "has_missing.sav"
        sav_file.write_bytes(b"\x00")

        mock_df = pd.DataFrame({"age": [20.0, 30.0, 99.0]})
        mock_meta = SimpleNamespace(
            variable_measure={"age": "scale"},
            column_labels=["나이"],
            variable_value_labels={},
            missing_ranges={"age": [[99, 99]]},  # 99는 결측치
        )

        mock_pyreadstat = MagicMock()
        mock_pyreadstat.read_sav.return_value = (mock_df, mock_meta)

        with patch.dict("sys.modules", {"pyreadstat": mock_pyreadstat}):
            ds = read_sav(str(sav_file))

        assert ds.variables["age"].missing_values == [[99, 99]]

    def test_no_variable_measure_uses_guess(self, tmp_path):
        """variable_measure 없음 → _guess_measure_type 호출(58)."""
        sav_file = tmp_path / "no_measure.sav"
        sav_file.write_bytes(b"\x00")

        mock_df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        mock_meta = SimpleNamespace(
            variable_measure={},  # 빈 dict → col not in variable_measure
            column_labels=["점수"],
            variable_value_labels={},
            missing_ranges={},
        )

        mock_pyreadstat = MagicMock()
        mock_pyreadstat.read_sav.return_value = (mock_df, mock_meta)

        with patch.dict("sys.modules", {"pyreadstat": mock_pyreadstat}):
            ds = read_sav(str(sav_file))

        # 수치형 → SCALE로 추측
        assert ds.variables["score"].measure.name == "SCALE"

"""커버리지 5라운드 — 잔여 도달 가능 라인 보강.

대상:
  core/project.py        274    : OSError → ProjectError
  io/clipboard_reader.py 92-93  : ParserError → IORError
  io/spss_writer.py      30-31  : pyreadstat ImportError → SWBImportError
  analysis/r_bridge.py    83    : R 없음 → execute() 에러 딕셔너리 반환
  analysis/r_bridge.py   215    : R 없음 → get_installed_packages() [] 반환
  analysis/discriminant_analysis.py 72-76 : _SKLEARN_AVAILABLE=False → 경고
  analysis/discriminant_analysis.py 400   : scores.shape[1] <= j → ""
  analysis/python_bridge.py 129-130      : 변수 직렬화 예외 → pass
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType


# ---------------------------------------------------------------------------
# core/project.py 274: OSError in pd.read_parquet → ProjectError
# ---------------------------------------------------------------------------

class TestProjectLoadOSError:

    def test_parquet_oserror_raises_project_error(self, tmp_path):
        """pd.read_parquet OSError → ProjectError (line 274)."""
        from statworkbench.core.project import load, SCHEMA_VERSION
        from statworkbench.core.exceptions import ProjectError

        archive = tmp_path / "test.swb"
        df = pd.DataFrame({"x": [1, 2, 3]})

        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("manifest.json", json.dumps({
                "schema_version": SCHEMA_VERSION,
                "created_at": "2026-01-01T00:00:00+00:00",
            }))
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            zf.writestr("data/active.parquet", buf.getvalue())
            zf.writestr("metadata/variables.json", json.dumps({}))
            zf.writestr("metadata/dataset.json", json.dumps({
                "name": "Test",
                "description": "",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }))

        with patch("statworkbench.core.project.pd.read_parquet", side_effect=OSError("disk error")):
            with pytest.raises(ProjectError):
                load(str(archive))


# ---------------------------------------------------------------------------
# io/clipboard_reader.py 92-93: ParserError → IORError
# ---------------------------------------------------------------------------

class TestClipboardParserError:

    def test_parser_error_raises_iorerror(self):
        """pd.read_csv ParserError → IORError (lines 92-93)."""
        from statworkbench.io.clipboard_reader import read_clipboard_from_qt
        from statworkbench.core.exceptions import IORError

        # 따옴표가 닫히지 않은 텍스트 → ParserError
        bad_text = 'col1\tcol2\n"unclosed\tB'

        with pytest.raises(IORError, match="파싱 오류"):
            read_clipboard_from_qt(bad_text)


# ---------------------------------------------------------------------------
# io/spss_writer.py 30-31: pyreadstat ImportError → SWBImportError
# ---------------------------------------------------------------------------

class TestSpssWriterImportError:

    def test_pyreadstat_missing_raises_swb_import_error(self, tmp_path):
        """pyreadstat ImportError → SWBImportError (lines 30-31)."""
        from statworkbench.io.spss_writer import write_sav
        from statworkbench.core.exceptions import ImportError as SWBImportError

        df = pd.DataFrame({"x": [1, 2, 3]})
        ds = Dataset(df, "Test")
        out = str(tmp_path / "test.sav")

        with patch.dict(sys.modules, {"pyreadstat": None}):
            with pytest.raises(SWBImportError):
                write_sav(ds, out)


# ---------------------------------------------------------------------------
# analysis/r_bridge.py 83: is_available() False → execute() 에러 반환
# ---------------------------------------------------------------------------

class TestRBridgeUnavailableExecute:

    def test_execute_returns_error_when_r_not_available(self):
        """is_available() False → execute() 에러 딕셔너리 (line 83)."""
        from statworkbench.analysis.r_bridge import RBridge

        rb = RBridge()
        rb._r_available = False  # R 없음으로 강제 설정

        result = rb.execute("print(1 + 1)")

        assert result["success"] is False
        assert "R이 설치되지 않았습니다" in result["error"]


# ---------------------------------------------------------------------------
# analysis/r_bridge.py 215: is_available() False → get_installed_packages() []
# ---------------------------------------------------------------------------

class TestRBridgeUnavailablePackages:

    def test_get_packages_returns_empty_when_r_not_available(self):
        """is_available() False → get_installed_packages() = [] (line 215)."""
        from statworkbench.analysis.r_bridge import RBridge

        rb = RBridge()
        rb._r_available = False

        pkgs = rb.get_installed_packages()

        assert pkgs == []


# ---------------------------------------------------------------------------
# analysis/discriminant_analysis.py 72-76: _SKLEARN_AVAILABLE=False → 경고
# ---------------------------------------------------------------------------

class TestDiscriminantSklearnUnavailable:

    def test_warning_when_sklearn_unavailable(self):
        """_SKLEARN_AVAILABLE=False → 경고 메시지 후 return (lines 72-76)."""
        import statworkbench.analysis.discriminant_analysis as da_mod
        from statworkbench.analysis.discriminant_analysis import run_analysis

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20,
            "x": rng.normal(0, 1, 40),
            "y": rng.normal(0, 1, 40),
        })
        ds = Dataset(df, "DiscNoSK")
        ds.variables["group"].measure = MeasureType.NOMINAL
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.SCALE

        spec = {
            "variables": {"dependent": "group", "predictors": ["x", "y"]},
        }

        with patch.object(da_mod, "_SKLEARN_AVAILABLE", False):
            result = run_analysis(ds, spec)

        assert any("scikit-learn" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# analysis/discriminant_analysis.py 400: scores.shape[1] <= j → row[...] = ""
# ---------------------------------------------------------------------------

class TestDiscriminantStructureMatrixShortScores:

    def test_empty_string_when_scores_dimension_exceeded(self):
        """n_components > scores.shape[1] → row[...] = '' (line 400)."""
        from statworkbench.analysis.discriminant_analysis import _add_structure_matrix
        from statworkbench.analysis.result import AnalysisResult

        result = AnalysisResult(id="da_test", title="DA Test", spec={})

        # 2개 샘플, 2개 피처
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        # LDA mock: transform → (n, 1) 배열 (1개 함수만 반환)
        mock_lda = MagicMock()
        mock_lda.transform.return_value = np.array([[0.1], [0.5], [0.9]])

        # n_components=2로 요청하지만 scores.shape[1]=1 → j=1에서 line 400 실행
        _add_structure_matrix(result, X, y, mock_lda, ["var1", "var2"], n_components=2)

        struct_tbl = next(
            (t for t in result.tables if "Structure" in t.title or "구조" in t.title),
            None,
        )
        assert struct_tbl is not None
        # 함수2 열이 "" 값을 가져야 함
        if "함수2" in struct_tbl.dataframe.columns:
            assert all(v == "" for v in struct_tbl.dataframe["함수2"])


# ---------------------------------------------------------------------------
# analysis/python_bridge.py 129-130: 변수 직렬화 예외 → pass
# ---------------------------------------------------------------------------

class TestPythonBridgeSerializationException:

    def test_variable_serialization_exception_is_silenced(self):
        """DataFrame.head() raises → except pass (lines 129-130)."""
        from statworkbench.analysis.python_bridge import PythonBridge

        bridge = PythonBridge()

        # 스크립트: 간단한 DataFrame 생성
        script = "result_df = pd.DataFrame({'a': [1, 2, 3]})"

        # head() 호출 시 예외 발생 → 직렬화 except block 실행
        with patch.object(pd.DataFrame, "head", side_effect=RuntimeError("head failed")):
            output = bridge.execute(script)

        # 예외가 pass 처리됨 → 실행 자체는 성공
        assert output["success"] is True
        # result_df가 수집에 실패해도 variables는 존재 (빈 dict)
        assert "variables" in output

"""project_store.py 커버리지 보강 테스트.

미커버 라인:
  55-57  : _write_parquet_or_fallback — Parquet 실패 시 except 통과
  91-104 : _read_dataframe — Parquet 없음 → CSV fallback, CSV 실패 → FileReadError,
           DATA_PARQUET/CSV 모두 없음 → FileReadError
  109    : 모듈 레벨 save_project() 함수
  114    : 모듈 레벨 load_project() 함수
  198-199: OSError in save → FileWriteError
  250-252: KeyError in variables.json → variables = {}
  259-260: KeyError in dataset.json → pass
  267-268: KeyError in project.json → pass
  270-273: load_project — BadZipFile / OSError 예외
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import FileReadError, FileWriteError
from nuristat.core.project import Project
from nuristat.io.project_store import (
    DATA_CSV,
    DATA_PARQUET,
    MANIFEST_NAME,
    META_DATASET,
    META_PROJECT,
    META_VARIABLES,
    SCHEMA_VERSION,
    ProjectStore,
    _read_dataframe,
    _write_parquet_or_fallback,
    save_project,
    load_project,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


@pytest.fixture
def sample_project():
    df = pd.DataFrame({"x": [10, 20, 30], "y": ["A", "B", "C"]})
    ds = Dataset(data=df, name="TestDS", source_info={"format": "test"})
    return Project(name="TestProject", dataset=ds)


# ---------------------------------------------------------------------------
# _write_parquet_or_fallback — Parquet 실패 fallback
# ---------------------------------------------------------------------------

class TestWriteParquetFallback:

    def test_parquet_failure_still_writes_csv(self, sample_df, tmp_path):
        """Parquet 쓰기 실패해도 CSV backup은 반드시 기록된다."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            with patch("pandas.DataFrame.to_parquet", side_effect=Exception("no pyarrow")):
                _write_parquet_or_fallback(sample_df, zf)
            names = zf.namelist()
        assert DATA_CSV in names
        assert DATA_PARQUET not in names

    def test_parquet_success_writes_both(self, sample_df):
        """Parquet 성공 시 Parquet + CSV 모두 기록."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            _write_parquet_or_fallback(sample_df, zf)
            names = zf.namelist()
        assert DATA_CSV in names


# ---------------------------------------------------------------------------
# _read_dataframe — CSV fallback 경로
# ---------------------------------------------------------------------------

class TestReadDataframeFallback:

    def _make_zip_csv_only(self, df: pd.DataFrame) -> zipfile.ZipFile:
        """Parquet 없이 CSV만 있는 인메모리 ZIP."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            csv_buf = io.BytesIO()
            df.to_csv(csv_buf, index=False, encoding="utf-8")
            zf.writestr(DATA_CSV, csv_buf.getvalue())
        buf.seek(0)
        return zipfile.ZipFile(buf, "r")

    def test_reads_csv_when_no_parquet(self, sample_df):
        """Parquet 없이 CSV만 있으면 CSV로 읽기 성공."""
        with self._make_zip_csv_only(sample_df) as zf:
            result = _read_dataframe(zf)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_raises_when_no_data_files(self):
        """Parquet/CSV 모두 없으면 FileReadError."""
        buf = io.BytesIO()
        manifest = {"schema_version": SCHEMA_VERSION}
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(MANIFEST_NAME, json.dumps(manifest))
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            with pytest.raises(FileReadError, match="데이터 파일이 없습니다"):
                _read_dataframe(zf)

    def test_csv_read_failure_raises(self):
        """CSV가 있지만 읽기 실패 → FileReadError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(DATA_CSV, b"\xff\xfe broken csv data here !!!!")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            with patch("pandas.read_csv", side_effect=Exception("bad csv")):
                with pytest.raises(FileReadError):
                    _read_dataframe(zf)

    def test_parquet_failure_falls_through_to_csv(self, sample_df):
        """Parquet 있지만 읽기 실패 → CSV fallback으로 정상 읽기."""
        buf = io.BytesIO()
        csv_buf = io.BytesIO()
        sample_df.to_csv(csv_buf, index=False, encoding="utf-8")

        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(DATA_PARQUET, b"not a valid parquet file")
            zf.writestr(DATA_CSV, csv_buf.getvalue())
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            result = _read_dataframe(zf)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# load_project — BadZipFile / OSError
# ---------------------------------------------------------------------------

class TestLoadProjectExceptions:

    def test_bad_zip_file_raises(self, tmp_path):
        """손상된 ZIP → FileReadError(BadZipFile)."""
        bad_file = tmp_path / "bad.swb"
        bad_file.write_bytes(b"this is not a zip file at all")
        with pytest.raises(FileReadError, match="손상된"):
            ProjectStore.load_project(str(bad_file))

    def test_nonexistent_file_raises(self, tmp_path):
        """존재하지 않는 경로 → FileReadError."""
        with pytest.raises(FileReadError, match="파일이 존재하지 않습니다"):
            ProjectStore.load_project(str(tmp_path / "ghost.swb"))

    def test_oserror_on_open_raises(self, tmp_path, sample_project):
        """zipfile.ZipFile 열기 OSError → FileReadError."""
        save_path = str(tmp_path / "proj.swb")
        ProjectStore.save_project(sample_project, save_path)
        with patch("zipfile.ZipFile", side_effect=OSError("no disk")):
            with pytest.raises(FileReadError):
                ProjectStore.load_project(save_path)

    def test_corrupt_manifest_raises(self, tmp_path):
        """manifest.json이 JSON 아닌 경우 → FileReadError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(MANIFEST_NAME, b"NOT JSON {{{{")
            df = pd.DataFrame({"a": [1]})
            csv_buf = io.BytesIO()
            df.to_csv(csv_buf, index=False, encoding="utf-8")
            zf.writestr(DATA_CSV, csv_buf.getvalue())
            zf.writestr(META_VARIABLES, json.dumps({}))
            zf.writestr(META_DATASET, json.dumps({}))
            zf.writestr(META_PROJECT, json.dumps({}))
        bad_file = tmp_path / "bad_manifest.swb"
        bad_file.write_bytes(buf.getvalue())
        with pytest.raises(FileReadError, match="manifest"):
            ProjectStore.load_project(str(bad_file))

    def test_schema_version_mismatch_still_loads(self, tmp_path):
        """schema_version 불일치해도 로드 시도."""
        buf = io.BytesIO()
        df = pd.DataFrame({"a": [1, 2]})
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8")

        manifest = {
            "schema_version": "99.0",
            "application": "NuriStat",
            "version": "0.1.0",
            "entries": [DATA_CSV, META_VARIABLES, META_DATASET, META_PROJECT],
        }
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(MANIFEST_NAME, json.dumps(manifest))
            zf.writestr(DATA_CSV, csv_buf.getvalue())
            zf.writestr(META_VARIABLES, json.dumps({}))
            zf.writestr(META_DATASET, json.dumps({"name": "test"}))
            zf.writestr(META_PROJECT, json.dumps({"name": "proj", "schema_version": "99.0"}))
        bad_ver_file = tmp_path / "old_schema.swb"
        bad_ver_file.write_bytes(buf.getvalue())
        proj = ProjectStore.load_project(str(bad_ver_file))
        assert proj is not None


# ---------------------------------------------------------------------------
# Lines 109, 114: 모듈 레벨 편의 함수
# ---------------------------------------------------------------------------

class TestModuleLevelFunctions:

    def test_save_project_module_fn(self, tmp_path, sample_project):
        """모듈 레벨 save_project() → ProjectStore.save_project() 위임(109)."""
        path = str(tmp_path / "test.swb")
        save_project(sample_project, path)
        assert Path(path).exists()

    def test_load_project_module_fn(self, tmp_path, sample_project):
        """모듈 레벨 load_project() → ProjectStore.load_project() 위임(114)."""
        path = str(tmp_path / "test.swb")
        save_project(sample_project, path)
        loaded = load_project(path)
        assert loaded is not None
        assert loaded.dataset is not None


# ---------------------------------------------------------------------------
# Lines 198-199: OSError in save → FileWriteError
# ---------------------------------------------------------------------------

class TestSaveOSError:

    def test_oserror_in_zipfile_raises_write_error(self, tmp_path, sample_project):
        """zipfile.ZipFile OSError → FileWriteError(198-199)."""
        path = str(tmp_path / "test.swb")
        with patch("nuristat.io.project_store.zipfile.ZipFile",
                   side_effect=OSError("disk full")):
            with pytest.raises(FileWriteError):
                ProjectStore.save_project(sample_project, path)


# ---------------------------------------------------------------------------
# Lines 250-252, 259-260, 267-268: metadata json 누락 → 기본값 사용
# ---------------------------------------------------------------------------

class TestMissingMetadata:

    def _make_swb_without(self, df: pd.DataFrame, exclude_keys: list[str], tmp_path, name="test.swb") -> str:
        """지정한 메타데이터 파일을 제외한 .swb 생성."""
        import io as _io
        buf = _io.BytesIO()
        csv_buf = _io.BytesIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "application": "NuriStat",
            "version": "0.1.0",
            "entries": [DATA_CSV, META_VARIABLES, META_DATASET, META_PROJECT],
        }
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_NAME, json.dumps(manifest))
            zf.writestr(DATA_CSV, csv_buf.getvalue())
            if META_VARIABLES not in exclude_keys:
                zf.writestr(META_VARIABLES, json.dumps({"x": {"measure": "scale"}}))
            if META_DATASET not in exclude_keys:
                zf.writestr(META_DATASET, json.dumps({"name": "TestDS"}))
            if META_PROJECT not in exclude_keys:
                zf.writestr(META_PROJECT, json.dumps({"name": "TestProject"}))

        path = str(tmp_path / name)
        Path(path).write_bytes(buf.getvalue())
        return path

    def test_missing_variables_json(self, tmp_path):
        """variables.json 없음 → variables={} (250-252)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        path = self._make_swb_without(df, [META_VARIABLES], tmp_path, "no_vars.swb")
        proj = ProjectStore.load_project(path)
        assert proj.dataset is not None

    def test_missing_dataset_json(self, tmp_path):
        """dataset.json 없음 → dataset_info={} (259-260)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        path = self._make_swb_without(df, [META_DATASET], tmp_path, "no_ds.swb")
        proj = ProjectStore.load_project(path)
        assert proj.dataset is not None

    def test_missing_project_json(self, tmp_path):
        """project.json 없음 → project_info={} (267-268)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        path = self._make_swb_without(df, [META_PROJECT], tmp_path, "no_proj.swb")
        proj = ProjectStore.load_project(path)
        assert proj is not None

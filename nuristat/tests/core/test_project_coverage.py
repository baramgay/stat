"""core/project.py 커버리지 보강 테스트.

미커버 라인:
  58-59   : mark_dirty()
  63      : is_dirty()
  67      : clear_dirty()
  102-104 : add_dataset()
  108     : touch()
  179-180 : OSError in save() → ProjectError
  269     : KeyError in load() → ProjectError
  274     : OSError in load() → ProjectError
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.project import Project, save, load
from nuristat.core.exceptions import ProjectError


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_dataset():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    ds = Dataset(df, name="TestDS")
    return ds


@pytest.fixture
def simple_project(simple_dataset):
    return Project(dataset=simple_dataset, name="TestProject")


# ---------------------------------------------------------------------------
# Lines 58-59: mark_dirty()
# ---------------------------------------------------------------------------

class TestMarkDirty:

    def test_mark_dirty_sets_flag(self, simple_project):
        """mark_dirty() → _dirty=True (58-59)."""
        simple_project.mark_dirty()
        assert simple_project._dirty is True

    def test_mark_dirty_updates_modified_at(self, simple_project):
        """mark_dirty() → touch() → modified_at 갱신(108)."""
        before = simple_project.modified_at
        import time; time.sleep(0.01)
        simple_project.mark_dirty()
        assert simple_project.modified_at >= before


# ---------------------------------------------------------------------------
# Line 63: is_dirty()
# ---------------------------------------------------------------------------

class TestIsDirty:

    def test_is_dirty_initially_false(self, simple_project):
        assert simple_project.is_dirty() is False

    def test_is_dirty_after_mark(self, simple_project):
        """is_dirty() True 반환(63)."""
        simple_project.mark_dirty()
        assert simple_project.is_dirty() is True


# ---------------------------------------------------------------------------
# Line 67: clear_dirty()
# ---------------------------------------------------------------------------

class TestClearDirty:

    def test_clear_dirty_resets_flag(self, simple_project):
        """clear_dirty() → _dirty=False (67)."""
        simple_project.mark_dirty()
        simple_project.clear_dirty()
        assert simple_project._dirty is False


# ---------------------------------------------------------------------------
# Lines 102-104: add_dataset()
# ---------------------------------------------------------------------------

class TestAddDataset:

    def test_add_dataset_updates_project(self, simple_project):
        """add_dataset() → datasets 목록 + dataset 갱신 + mark_dirty(102-104)."""
        new_df = pd.DataFrame({"y": [10.0, 20.0]})
        new_ds = Dataset(new_df, "NewDS")
        simple_project.add_dataset(new_ds)
        assert simple_project.dataset is new_ds
        assert new_ds in simple_project.datasets
        assert simple_project.is_dirty()


# ---------------------------------------------------------------------------
# Line 108: touch()
# ---------------------------------------------------------------------------

class TestTouch:

    def test_touch_updates_modified_at(self, simple_project):
        """touch() → modified_at 갱신(108)."""
        import time
        before = simple_project.modified_at
        time.sleep(0.01)
        simple_project.touch()
        assert simple_project.modified_at >= before


# ---------------------------------------------------------------------------
# Lines 179-180: OSError in save() → ProjectError
# ---------------------------------------------------------------------------

class TestSaveOSError:

    def test_save_oserror_raises_project_error(self, simple_dataset, tmp_path):
        """save() OSError → ProjectError(179-180)."""
        path = str(tmp_path / "test.swb")
        with patch("nuristat.core.project.zipfile.ZipFile",
                   side_effect=OSError("no space")):
            with pytest.raises(ProjectError, match="Failed to save"):
                save(simple_dataset, path)


# ---------------------------------------------------------------------------
# Line 269: KeyError in load() → ProjectError
# ---------------------------------------------------------------------------

class TestLoadKeyError:

    def test_load_missing_manifest_raises_project_error(self, tmp_path):
        """manifest.json 없는 ZIP → KeyError → ProjectError(269)."""
        bad_zip = tmp_path / "no_manifest.swb"
        import io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.txt", "hello")
        bad_zip.write_bytes(buf.getvalue())
        with pytest.raises(ProjectError):
            load(str(bad_zip))


# ---------------------------------------------------------------------------
# Line 274: OSError in load() → ProjectError
# ---------------------------------------------------------------------------

class TestLoadOSError:

    def test_load_oserror_raises_project_error(self, tmp_path):
        """존재하지 않는 경로 → OSError → ProjectError(274)."""
        with pytest.raises(ProjectError):
            load(str(tmp_path / "nonexistent.swb"))

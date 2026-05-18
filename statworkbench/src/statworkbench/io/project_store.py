"""Project storage for StatWorkbench.

Projects are saved as ``.swb`` files — ZIP archives containing:

- ``manifest.json``        — schema version and archive index
- ``data/active.parquet``  — primary dataset (Parquet format)
- ``data/backup.csv``      — CSV fallback of the dataset
- ``metadata/variables.json`` — variable metadata
- ``metadata/dataset.json``   — dataset metadata (source info, etc.)
- ``metadata/project.json``   — project-level metadata

If Parquet write/read fails, the system transparently falls back to CSV.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import FileReadError, FileWriteError
from statworkbench.core.project import Project
from statworkbench.core.variable import VariableMeta

MANIFEST_NAME = "manifest.json"
DATA_PARQUET = "data/active.parquet"
DATA_CSV = "data/backup.csv"
META_VARIABLES = "metadata/variables.json"
META_DATASET = "metadata/dataset.json"
META_PROJECT = "metadata/project.json"
SCHEMA_VERSION = "1.0"


def _write_parquet_or_fallback(
    df: pd.DataFrame, archive: zipfile.ZipFile
) -> None:
    """Write DataFrame as Parquet with CSV fallback.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    archive : zipfile.ZipFile
        Open ZIP archive in write mode.
    """
    buf = BytesIO()
    try:
        df.to_parquet(buf, index=False, engine="pyarrow")
        archive.writestr(DATA_PARQUET, buf.getvalue())
    except Exception:
        # Parquet failed — skip; we still have CSV fallback
        pass

    # Always write CSV as fallback
    csv_buf = BytesIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8")
    archive.writestr(DATA_CSV, csv_buf.getvalue())


def _read_dataframe(
    archive: zipfile.ZipFile, encoding: str = "utf-8"
) -> pd.DataFrame:
    """Read DataFrame from archive, preferring Parquet over CSV.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open ZIP archive in read mode.
    encoding : str
        Encoding for CSV fallback.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileReadError
        If neither Parquet nor CSV can be read.
    """
    # Try Parquet first
    if DATA_PARQUET in archive.namelist():
        try:
            with archive.open(DATA_PARQUET) as f:
                return pd.read_parquet(f, engine="pyarrow")
        except Exception:
            pass  # Fall through to CSV

    # Fallback to CSV
    if DATA_CSV in archive.namelist():
        try:
            with archive.open(DATA_CSV) as f:
                return pd.read_csv(f, encoding=encoding)
        except Exception as exc:
            raise FileReadError(
                "archive", f"CSV fallback 읽기 실패: {exc}"
            ) from exc

    raise FileReadError("archive", "데이터 파일이 없습니다 (Parquet/CSV)")


def save_project(project: Project, path: str) -> None:
    """모듈 레벨 편의 함수."""
    return ProjectStore.save_project(project, path)


def load_project(path: str) -> Project:
    """모듈 레벨 편의 함수."""
    return ProjectStore.load_project(path)


class ProjectStore:
    """Static utilities for saving and loading StatWorkbench projects."""

    @staticmethod
    def save_project(project: Project, path: str) -> None:
        """Save a Project to a ``.swb`` file (ZIP archive).

        Parameters
        ----------
        project : Project
            The project to save.
        path : str
            Destination file path.  The ``.swb`` extension is appended
            automatically if missing.

        Raises
        ------
        FileWriteError
            If the project cannot be saved.
        """
        if project.dataset is None:
            raise FileWriteError(path, "저장할 데이터셋이 없습니다")

        save_path = Path(path)
        if not save_path.suffix.lower() == ".swb":
            save_path = save_path.with_suffix(".swb")

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = project.dataset

        try:
            with zipfile.ZipFile(
                save_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                # --- manifest.json ---
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "application": "StatWorkbench",
                    "version": "0.1.0",
                    "entries": [
                        DATA_PARQUET,
                        DATA_CSV,
                        META_VARIABLES,
                        META_DATASET,
                        META_PROJECT,
                    ],
                }
                zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))

                # --- data/active.parquet (with CSV fallback) ---
                _write_parquet_or_fallback(dataset.data, zf)

                # --- metadata/variables.json ---
                variables_dict = {
                    name: var.to_dict()
                    for name, var in dataset.variables.items()
                }
                zf.writestr(
                    META_VARIABLES,
                    json.dumps(variables_dict, indent=2, ensure_ascii=False),
                )

                # --- metadata/dataset.json ---
                zf.writestr(
                    META_DATASET,
                    json.dumps(
                        dataset.to_dict(), indent=2, ensure_ascii=False
                    ),
                )

                # --- metadata/project.json ---
                zf.writestr(
                    META_PROJECT,
                    json.dumps(
                        project.to_metadata_dict(),
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
        except OSError as exc:
            raise FileWriteError(str(save_path), str(exc)) from exc

    @staticmethod
    def load_project(path: str) -> Project:
        """Load a Project from a ``.swb`` file (ZIP archive).

        Parameters
        ----------
        path : str
            Path to the ``.swb`` file.

        Returns
        -------
        Project
            Restored project with dataset and metadata.

        Raises
        ------
        FileReadError
            If the project cannot be loaded.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileReadError(path, "파일이 존재하지 않습니다")

        try:
            with zipfile.ZipFile(load_path, "r") as zf:
                # --- manifest.json ---
                try:
                    manifest_raw = zf.read(MANIFEST_NAME)
                    manifest = json.loads(manifest_raw)
                except (KeyError, json.JSONDecodeError) as exc:
                    raise FileReadError(
                        path, f"manifest.json 읽기 실패: {exc}"
                    ) from exc

                schema_version = manifest.get("schema_version", "unknown")
                if schema_version != SCHEMA_VERSION:
                    # We still try to load but note the mismatch
                    pass

                # --- data ---
                df = _read_dataframe(zf)

                # --- metadata/variables.json ---
                variables: dict[str, VariableMeta] = {}
                try:
                    vars_raw = zf.read(META_VARIABLES)
                    vars_dict = json.loads(vars_raw)
                    for name, vdict in vars_dict.items():
                        variables[name] = VariableMeta.from_dict(vdict)
                except (KeyError, json.JSONDecodeError) as exc:
                    # If variables metadata is missing/corrupt, rebuild defaults
                    variables = {}

                # --- metadata/dataset.json ---
                dataset_info: dict[str, Any] = {}
                try:
                    ds_raw = zf.read(META_DATASET)
                    dataset_info = json.loads(ds_raw)
                except (KeyError, json.JSONDecodeError):
                    pass

                # --- metadata/project.json ---
                project_info: dict[str, Any] = {}
                try:
                    proj_raw = zf.read(META_PROJECT)
                    project_info = json.loads(proj_raw)
                except (KeyError, json.JSONDecodeError):
                    pass

        except zipfile.BadZipFile as exc:
            raise FileReadError(path, f"손상된 .swb 파일: {exc}") from exc
        except OSError as exc:
            raise FileReadError(path, str(exc)) from exc

        # Reconstruct Dataset
        dataset = Dataset(
            data=df,
            name=dataset_info.get("name", load_path.stem),
            variables=variables,
            source_info=dataset_info.get("source_info", {}),
        )

        # Reconstruct Project
        project = Project.from_metadata_dict(project_info, dataset=dataset)
        project.schema_version = schema_version

        return project

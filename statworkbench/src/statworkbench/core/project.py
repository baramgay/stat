"""Project save/load using ZIP-based ``.swb`` archives.

A project bundles the dataset (as Parquet and backup CSV), variable metadata
(JSON), and a manifest into a single ZIP file with the ``.swb`` extension.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from statworkbench.core.audit import AuditLog
from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import ProjectError

SCHEMA_VERSION: str = "1.0"

# Archive member paths
_MANIFEST_PATH = "manifest.json"
_ACTIVE_PARQUET = "data/active.parquet"
_BACKUP_CSV = "data/backup.csv"
_VARIABLES_JSON = "metadata/variables.json"
_DATASET_JSON = "metadata/dataset.json"


class Project:
    """A StatWorkbench project.

    Contains the active dataset, analysis output history, syntax log,
    and project-level settings.
    """

    def __init__(
        self,
        name: str = "Untitled Project",
        dataset: Dataset | None = None,
        settings: dict[str, Any] | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.name: str = name
        self.dataset: Dataset | None = dataset
        self.datasets: list[Dataset] = [dataset] if dataset else []
        self.settings: dict[str, Any] = settings or {}
        self.schema_version: str = schema_version
        self.syntax_history: list[str] = []
        self.created_at: datetime = datetime.now()
        self.modified_at: datetime = datetime.now()
        self.file_path: str | None = None
        self._dirty: bool = False

    def mark_dirty(self) -> None:
        """Mark the project as having unsaved changes."""
        self._dirty = True
        self.touch()

    def is_dirty(self) -> bool:
        """Return whether the project has unsaved changes."""
        return self._dirty

    def clear_dirty(self) -> None:
        """Clear the dirty flag after saving."""
        self._dirty = False

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize project metadata (without data) to dict."""
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "settings": self.settings,
            "syntax_history": self.syntax_history,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }

    @classmethod
    def from_metadata_dict(
        cls, data: dict[str, Any], dataset: Dataset | None = None
    ) -> Project:
        """Deserialize project metadata dict to a Project."""
        proj = cls(
            name=data.get("name", "Untitled Project"),
            dataset=dataset,
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            settings=data.get("settings", {}),
        )
        proj.syntax_history = data.get("syntax_history", [])
        proj.created_at = datetime.fromisoformat(
            data.get("created_at", datetime.now().isoformat())
        )
        proj.modified_at = datetime.fromisoformat(
            data.get("modified_at", datetime.now().isoformat())
        )
        return proj

    def add_dataset(self, dataset: Dataset) -> None:
        """데이터셋을 프로젝트에 추가합니다."""
        self.datasets.append(dataset)
        self.dataset = dataset
        self.mark_dirty()

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.modified_at = datetime.now()


def save(dataset: Dataset, path: str | Path, audit: AuditLog | None = None) -> None:
    """Save a dataset and metadata into a ``.swb`` ZIP archive.

    The archive contains:

    - ``manifest.json`` — schema version and save metadata
    - ``data/active.parquet`` — primary data in Apache Parquet format
    - ``data/backup.csv`` — CSV backup for portability
    - ``metadata/variables.json`` — serialised variable metadata
    - ``metadata/dataset.json`` — dataset-level metadata

    Args:
        dataset: The :class:`Dataset` to save.
        path: Destination file path. Should end with ``.swb`` but any
            extension is accepted.
        audit: Optional :class:`AuditLog` to include in the manifest.

    Raises:
        ProjectError: If an I/O error occurs during writing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Manifest
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset.name,
                "n_rows": dataset.n_rows,
                "n_cols": dataset.n_cols,
            }
            if audit is not None:
                manifest["audit_log"] = audit.to_list()
            zf.writestr(_MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False))

            # 2. Active data (Parquet)
            parquet_bytes = dataset.data.to_parquet(index=False)
            zf.writestr(_ACTIVE_PARQUET, parquet_bytes)

            # 3. Backup data (CSV)
            csv_bytes = dataset.data.to_csv(index=False).encode("utf-8")
            zf.writestr(_BACKUP_CSV, csv_bytes)

            # 4. Variable metadata
            variables_dict = {
                name: meta.to_dict()
                for name, meta in dataset.variables.items()
            }
            zf.writestr(
                _VARIABLES_JSON,
                json.dumps(variables_dict, indent=2, ensure_ascii=False, default=str),
            )

            # 5. Dataset metadata
            dataset_meta = {
                "name": dataset.name,
                "description": dataset.description,
                "shape": dataset.shape,
                "created_at": dataset.created_at.isoformat(),
                "updated_at": dataset.updated_at.isoformat(),
            }
            zf.writestr(
                _DATASET_JSON,
                json.dumps(dataset_meta, indent=2, ensure_ascii=False),
            )

    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectError(
            f"Failed to save project to '{target}': {exc}",
            details={"path": str(target)},
        ) from exc


def load(path: str | Path) -> Dataset:
    """Load a :class:`Dataset` from a ``.swb`` ZIP archive.

    Args:
        path: Path to the ``.swb`` file.

    Returns:
        A fully reconstructed :class:`Dataset`.

    Raises:
        ProjectError: If the archive is corrupt, a required member is
            missing, or the schema version is unsupported.
    """
    target = Path(path)

    if not target.exists():
        raise ProjectError(
            f"Project file not found: '{target}'.",
            details={"path": str(target)},
        )

    try:
        with zipfile.ZipFile(target, "r") as zf:
            # Validate required members
            required = {_MANIFEST_PATH, _ACTIVE_PARQUET, _VARIABLES_JSON, _DATASET_JSON}
            missing = required - set(zf.namelist())
            if missing:
                raise ProjectError(
                    f"Project archive is missing required members: {sorted(missing)}.",
                    details={"missing": sorted(missing)},
                )

            # 1. Manifest
            manifest_raw = zf.read(_MANIFEST_PATH).decode("utf-8")
            manifest: dict[str, Any] = json.loads(manifest_raw)
            version = manifest.get("schema_version", "unknown")
            if version != SCHEMA_VERSION:
                raise ProjectError(
                    f"Unsupported schema version '{version}'. "
                    f"Expected '{SCHEMA_VERSION}'.",
                    details={
                        "version": version,
                        "expected": SCHEMA_VERSION,
                    },
                )

            # 2. Active data (Parquet)
            parquet_bytes = zf.read(_ACTIVE_PARQUET)
            data = pd.read_parquet(__import__("io").BytesIO(parquet_bytes))

            # 3. Variable metadata
            variables_raw = zf.read(_VARIABLES_JSON).decode("utf-8")
            from statworkbench.core.variable import VariableMeta

            variables_dict: dict[str, Any] = json.loads(variables_raw)
            variables = {
                name: VariableMeta.from_dict(meta_dict)
                for name, meta_dict in variables_dict.items()
            }

            # 4. Dataset metadata
            dataset_meta_raw = zf.read(_DATASET_JSON).decode("utf-8")
            dataset_meta: dict[str, Any] = json.loads(dataset_meta_raw)

            # Reconstruct
            ds = Dataset(
                data=data,
                variables=variables,
                name=dataset_meta.get("name", ""),
                description=dataset_meta.get("description", ""),
            )
            ds.created_at = datetime.fromisoformat(dataset_meta["created_at"])
            ds.updated_at = datetime.fromisoformat(dataset_meta["updated_at"])
            ds._dirty = False

            return ds

    except zipfile.BadZipFile as exc:
        raise ProjectError(
            f"Project file is not a valid ZIP archive: '{target}'.",
            details={"path": str(target)},
        ) from exc
    except KeyError as exc:
        raise ProjectError(
            f"Missing key in project data: {exc}.",
            details={"key": str(exc)},
        ) from exc
    except (OSError, pd.errors.EmptyDataError) as exc:
        raise ProjectError(
            f"Failed to load project from '{target}': {exc}",
            details={"path": str(target)},
        ) from exc

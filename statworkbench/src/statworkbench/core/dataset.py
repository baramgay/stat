"""Dataset model for StatWorkbench."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
import pandas as pd
from statworkbench.core.exceptions import DatasetError
from statworkbench.core.typing import StorageType, MeasureType, Role
from statworkbench.core.variable import VariableMeta


class Dataset:
    """A dataset combining a DataFrame with variable metadata."""

    def __init__(
        self,
        data: pd.DataFrame,
        name: str = "Untitled",
        variables: Optional[dict[str, VariableMeta]] = None,
        description: str = "",
        source_info: Optional[dict[str, Any]] = None,
    ) -> None:
        self._data: pd.DataFrame = data.copy()
        self.name: str = name
        self.description: str = description
        self.source_info: dict[str, Any] = source_info or {}
        self.created_at: datetime = datetime.now(timezone.utc)
        self.updated_at: datetime = datetime.now(timezone.utc)
        self._dirty: bool = False
        self._variables: dict[str, VariableMeta]
        if variables is not None:
            self._variables = dict(variables)
        else:
            self._variables = {}
            for col in list(self._data.columns):
                from statworkbench.core.validation import validate_variable_name
                safe_name = validate_variable_name(col)
                if safe_name != col:
                    self._data = self._data.rename(columns={col: safe_name})
                self._variables[safe_name] = _infer_variable_meta(safe_name, self._data[safe_name])

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise DatasetError("Data must be a pandas DataFrame.")
        # Sync variables for new columns
        for col in value.columns:
            if col not in self._variables:
                self._variables[col] = _infer_variable_meta(col, value[col])
        # Remove variables for dropped columns
        for col in list(self._variables.keys()):
            if col not in value.columns:
                del self._variables[col]
        self._data = value
        self._dirty = True

    @property
    def variables(self) -> dict[str, VariableMeta]:
        return self._variables

    @variables.setter
    def variables(self, value: dict[str, VariableMeta]) -> None:
        self._variables = value

    @property
    def var_names(self) -> list[str]:
        return list(self._data.columns)

    @property
    def is_empty(self) -> bool:
        return self._data.empty

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def n_rows(self) -> int:
        return len(self._data)

    @property
    def n_vars(self) -> int:
        return len(self._data.columns)

    @property
    def n_cols(self) -> int:
        return len(self._data.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    def get_column(self, name: str) -> pd.Series:
        return self._data[name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataset to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "data": {col: self._data[col].tolist() for col in self._data.columns},
            "variables": {name: var.to_dict() for name, var in self._variables.items()},
            "source_info": self.source_info,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        """Deserialize dataset from a dictionary."""
        df = pd.DataFrame(data.get("data", {}))
        variables = {
            name: VariableMeta.from_dict(var_dict)
            for name, var_dict in data.get("variables", {}).items()
        }
        ds = cls(
            data=df,
            name=data.get("name", "Untitled"),
            variables=variables if variables else None,
            description=data.get("description", ""),
            source_info=data.get("source_info", {}),
        )
        ds._dirty = False
        created_at_str = data.get("created_at")
        updated_at_str = data.get("updated_at")
        if created_at_str:
            ds.created_at = datetime.fromisoformat(created_at_str)
        if updated_at_str:
            ds.updated_at = datetime.fromisoformat(updated_at_str)
        return ds

    def copy(self) -> Dataset:
        import copy
        return Dataset(
            data=self._data.copy(),
            name=self.name,
            variables={k: copy.deepcopy(v) for k, v in self._variables.items()},
        )

    def rename_variable(self, old_name: str, new_name: str) -> None:
        from statworkbench.core.validation import validate_variable_name
        new_name = validate_variable_name(new_name)
        if old_name not in self._variables:
            raise DatasetError(f"Variable '{old_name}' does not exist.")
        if new_name in self._variables and new_name != old_name:
            raise DatasetError(f"Variable name '{new_name}' already exists.")
        if old_name == new_name:
            return
        self._data = self._data.rename(columns={old_name: new_name})
        meta = self._variables.pop(old_name)
        meta.name = new_name
        meta.source_column = new_name
        self._variables[new_name] = meta
        self._dirty = True

    def get_variable(self, name: str) -> VariableMeta:
        if name not in self._variables:
            raise DatasetError(f"Variable '{name}' does not exist.")
        return self._variables[name]

    def update_variable_meta(self, name: str, **kwargs: Any) -> None:
        if name not in self._variables:
            raise DatasetError(f"Variable '{name}' does not exist.")
        meta = self._variables[name]
        for key, value in kwargs.items():
            if not hasattr(meta, key):
                raise DatasetError(f"Invalid field '{key}' for VariableMeta.")
            setattr(meta, key, value)
        self._dirty = True

    def add_variable(self, name: str, data: Optional[pd.Series] = None, meta: Optional[VariableMeta] = None) -> None:
        from statworkbench.core.validation import validate_variable_name
        name = validate_variable_name(name)
        if name in self._variables:
            raise DatasetError(f"Variable '{name}' already exists.")
        if data is None:
            data = pd.Series([pd.NA] * len(self._data), index=self._data.index, name=name)
        self._data[name] = data
        if meta is not None:
            self._variables[name] = meta
        else:
            self._variables[name] = _infer_variable_meta(name, self._data[name])
        self._dirty = True

    def remove_variable(self, name: str) -> None:
        if name not in self._variables:
            raise DatasetError(f"Variable '{name}' does not exist.")
        self._data = self._data.drop(columns=[name])
        del self._variables[name]
        self._dirty = True

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, n_rows={self.n_rows}, n_vars={self.n_vars})"

    def __len__(self) -> int:
        return self.n_rows


def _infer_variable_meta(name: str, series: pd.Series) -> VariableMeta:
    """Infer variable metadata from a pandas Series."""
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        storage = StorageType.DATETIME
        measure = MeasureType.DATE_TIME
    elif pd.api.types.is_integer_dtype(dtype):
        storage = StorageType.INTEGER
        measure = MeasureType.SCALE
    elif pd.api.types.is_float_dtype(dtype):
        storage = StorageType.FLOAT
        measure = MeasureType.SCALE
    elif pd.api.types.is_bool_dtype(dtype):
        storage = StorageType.BOOLEAN
        measure = MeasureType.BINARY
    else:
        storage = StorageType.STRING
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2:
            measure = MeasureType.BINARY
        else:
            measure = MeasureType.NOMINAL

    return VariableMeta(
        name=name,
        label=name,
        storage_type=storage,
        measure=measure,
    )

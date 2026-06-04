"""Core domain models for NuriStat.

Provides the foundational data structures: variable metadata, dataset management,
type definitions, validation, and project persistence.
"""

from nuristat.core.audit import AuditLog
from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import (
    AnalysisError,
    DatasetError,
    ImportError,
    NuriStatError,
    ProjectError,
    ValidationError,
    VariableError,
)
from nuristat.core.project import Project
from nuristat.core.typing import (
    MeasureType,
    MissingPolicy,
    Role,
    StorageType,
)
from nuristat.core.validation import (
    validate_measure_storage_compatibility,
    validate_missing_rules,
    validate_variable_name,
)
from nuristat.core.variable import VariableMeta

__all__ = [
    "StorageType",
    "MeasureType",
    "Role",
    "MissingPolicy",
    "VariableMeta",
    "Dataset",
    "Project",
    "validate_variable_name",
    "validate_measure_storage_compatibility",
    "validate_missing_rules",
    "NuriStatError",
    "VariableError",
    "DatasetError",
    "AnalysisError",
    "ValidationError",
    "ImportError",
    "ProjectError",
    "AuditLog",
]

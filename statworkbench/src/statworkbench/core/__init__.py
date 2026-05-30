"""Core domain models for StatWorkbench.

Provides the foundational data structures: variable metadata, dataset management,
type definitions, validation, and project persistence.
"""

from statworkbench.core.audit import AuditLog
from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import (
    AnalysisError,
    DatasetError,
    ImportError,
    ProjectError,
    StatWorkbenchError,
    ValidationError,
    VariableError,
)
from statworkbench.core.project import Project
from statworkbench.core.typing import (
    MeasureType,
    MissingPolicy,
    Role,
    StorageType,
)
from statworkbench.core.validation import (
    validate_measure_storage_compatibility,
    validate_missing_rules,
    validate_variable_name,
)
from statworkbench.core.variable import VariableMeta

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
    "StatWorkbenchError",
    "VariableError",
    "DatasetError",
    "AnalysisError",
    "ValidationError",
    "ImportError",
    "ProjectError",
    "AuditLog",
]

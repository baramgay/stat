"""Core domain models for StatWorkbench.

Provides the foundational data structures: variable metadata, dataset management,
type definitions, validation, and project persistence.
"""

from statworkbench.core.typing import (
    StorageType,
    MeasureType,
    Role,
    MissingPolicy,
)
from statworkbench.core.variable import VariableMeta
from statworkbench.core.dataset import Dataset
from statworkbench.core.project import Project
from statworkbench.core.validation import (
    validate_variable_name,
    validate_measure_storage_compatibility,
    validate_missing_rules,
)
from statworkbench.core.exceptions import (
    StatWorkbenchError,
    VariableError,
    DatasetError,
    AnalysisError,
    ValidationError,
    ImportError,
    ProjectError,
)
from statworkbench.core.audit import AuditLog

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

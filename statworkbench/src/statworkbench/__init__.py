"""StatWorkbench — Menu-based desktop statistical package.

A desktop statistics package that lets researchers, clinicians, and data analysts
load data, define variable properties, and run statistical analyses through a
menu-driven interface without writing code.
"""

__version__ = "3.3.1"
__author__ = "Hermes Agent"

# Re-export key types for convenience
from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import StatWorkbenchError
from statworkbench.core.typing import (
    MeasureType,
    MissingPolicy,
    Role,
    StorageType,
)
from statworkbench.core.variable import VariableMeta

__all__ = [
    "__version__",
    "StorageType",
    "MeasureType",
    "Role",
    "MissingPolicy",
    "VariableMeta",
    "Dataset",
    "StatWorkbenchError",
]

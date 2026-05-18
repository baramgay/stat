"""StatWorkbench — Menu-based desktop statistical package.

A desktop statistics package that lets researchers, clinicians, and data analysts
load data, define variable properties, and run statistical analyses through a
menu-driven interface without writing code.
"""

__version__ = "0.1.0"
__author__ = "Hermes Agent"

# Re-export key types for convenience
from statworkbench.core.typing import (
    StorageType,
    MeasureType,
    Role,
    MissingPolicy,
)
from statworkbench.core.variable import VariableMeta
from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import StatWorkbenchError

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

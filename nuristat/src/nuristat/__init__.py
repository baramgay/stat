"""NuriStat — Menu-based desktop statistical package.

A desktop statistics package that lets researchers, clinicians, and data analysts
load data, define variable properties, and run statistical analyses through a
menu-driven interface without writing code.
"""

__version__ = "3.6.0"
__author__ = "Hermes Agent"

# Re-export key types for convenience
from nuristat.core.dataset import Dataset
from nuristat.core.exceptions import NuriStatError
from nuristat.core.typing import (
    MeasureType,
    MissingPolicy,
    Role,
    StorageType,
)
from nuristat.core.variable import VariableMeta

__all__ = [
    "__version__",
    "StorageType",
    "MeasureType",
    "Role",
    "MissingPolicy",
    "VariableMeta",
    "Dataset",
    "NuriStatError",
]

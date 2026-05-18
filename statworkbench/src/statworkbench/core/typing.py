"""Type definitions and enums for StatWorkbench."""

from enum import Enum, auto


class StorageType(Enum):
    """Storage type for variable data."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


class MeasureType(Enum):
    """Measurement scale type for statistical analysis."""

    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    SCALE = "scale"
    BINARY = "binary"
    DATE_TIME = "date_time"
    TEXT = "text"


class Role(Enum):
    """Variable role in analysis."""

    INPUT = "input"
    TARGET = "target"
    WEIGHT = "weight"
    ID = "id"
    SPLIT = "split"
    FREQUENCY = "frequency"
    NONE = "none"


class MissingPolicy(Enum):
    """Missing data handling policy for analysis."""

    LISTWISE = "listwise"
    PAIRWISE = "pairwise"
    ANALYSIS_DEFAULT = "analysis_default"
    INCLUDE_AS_CATEGORY = "include_as_category"
    EXCLUDE_USER_MISSING_ONLY = "exclude_user_missing_only"
    EXCLUDE_SYSTEM_MISSING_ONLY = "exclude_system_missing_only"

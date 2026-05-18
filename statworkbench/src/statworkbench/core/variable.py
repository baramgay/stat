"""Variable metadata model for StatWorkbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from statworkbench.core.typing import MeasureType, Role, StorageType


@dataclass
class VariableMeta:
    """Metadata for a single variable (column) in a dataset.

    This corresponds to a row in the Variable View of SPSS.
    Each variable carries both storage information and statistical
    measurement metadata needed for analysis validation.
    """

    # Identity
    name: str
    """Unique variable name (column name)."""

    label: str = ""
    """Human-readable description."""

    # Type information
    storage_type: StorageType = StorageType.FLOAT
    """Physical storage type."""

    measure: MeasureType = MeasureType.SCALE
    """Statistical measurement scale."""

    role: Role = Role.NONE
    """Analysis role."""

    # Display
    width: int = 8
    """Display width."""

    decimals: int = 2
    """Number of decimal places to display."""

    align: str = "left"
    """Alignment: left, right, center."""

    # Value metadata
    value_labels: dict[str | int | float, str] = field(default_factory=dict)
    """Mapping from values to labels."""

    missing_values: list[Any] = field(default_factory=list)
    """User-defined missing value rules."""

    unit: str = ""
    """Measurement unit."""

    allowed_min: Optional[float] = None
    """Minimum allowed value."""

    allowed_max: Optional[float] = None
    """Maximum allowed value."""

    format_pattern: str = ""
    """Display format pattern."""

    datetime_format: str = ""
    """Date parsing format."""

    description: str = ""
    """Long-form description or notes."""

    source_column: str = ""
    """Original column name from the source file."""

    derived: bool = False
    """Whether this is a derived (computed) variable."""

    formula: Optional[str] = None
    """Formula expression for derived variables."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Creation timestamp."""

    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Last modification timestamp."""

    def __post_init__(self) -> None:
        """Post-initialization: validate and sanitize name."""
        from statworkbench.core.validation import validate_variable_name, validate_measure_storage_compatibility, validate_missing_rules
        self.name = validate_variable_name(self.name)
        # Coerce None missing_values to empty list
        if self.missing_values is None:
            self.missing_values = []
        # Coerce scalar missing_values to list
        elif not isinstance(self.missing_values, list):
            self.missing_values = [self.missing_values]
        # Validate measure/storage compatibility
        validate_measure_storage_compatibility(self.measure, self.storage_type)
        # Validate missing rules
        validate_missing_rules(self.missing_values, self.allowed_min, self.allowed_max)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VariableMeta):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def set_value_label(self, value: str | int | float, label: str) -> None:
        """Add or update a value label."""
        self.value_labels[value] = label
        self.touch()

    def add_missing_value(self, value: Any) -> None:
        """Add a missing value if not already present."""
        if value not in self.missing_values:
            self.missing_values.append(value)
            self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "label": self.label,
            "storage_type": self.storage_type.value,
            "measure": self.measure.value,
            "role": self.role.value,
            "width": self.width,
            "decimals": self.decimals,
            "align": self.align,
            "value_labels": {
                str(k): v for k, v in self.value_labels.items()
            },
            "missing_values": self.missing_values,
            "unit": self.unit,
            "allowed_min": self.allowed_min,
            "allowed_max": self.allowed_max,
            "format_pattern": self.format_pattern,
            "datetime_format": self.datetime_format,
            "description": self.description,
            "source_column": self.source_column,
            "derived": self.derived,
            "formula": self.formula,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariableMeta:
        """Deserialize from dictionary."""
        from datetime import datetime as _dt
        created_at_str = data.get("created_at")
        updated_at_str = data.get("updated_at")
        created_at = _dt.fromisoformat(created_at_str) if created_at_str else _dt.now()
        updated_at = _dt.fromisoformat(updated_at_str) if updated_at_str else _dt.now()
        return cls(
            name=data["name"],
            label=data.get("label", ""),
            storage_type=StorageType(data.get("storage_type", "float")),
            measure=MeasureType(data.get("measure", "scale")),
            role=Role(data.get("role", "none")),
            width=data.get("width", 8),
            decimals=data.get("decimals", 2),
            align=data.get("align", "left"),
            value_labels=data.get("value_labels", {}),
            missing_values=data.get("missing_values", []),
            unit=data.get("unit", ""),
            allowed_min=data.get("allowed_min"),
            allowed_max=data.get("allowed_max"),
            format_pattern=data.get("format_pattern", ""),
            datetime_format=data.get("datetime_format", ""),
            description=data.get("description", ""),
            source_column=data.get("source_column", ""),
            derived=data.get("derived", False),
            formula=data.get("formula"),
            created_at=created_at,
            updated_at=updated_at,
        )

    @property
    def has_value_labels(self) -> bool:
        """Return True if value labels are defined."""
        return bool(self.value_labels)

    @property
    def is_numeric(self) -> bool:
        """Return True if the variable stores numeric data."""
        return self.storage_type in (
            StorageType.INTEGER,
            StorageType.FLOAT,
        )

    @property
    def is_categorical(self) -> bool:
        """Return True if the variable is categorical."""
        return self.measure in (
            MeasureType.NOMINAL,
            MeasureType.ORDINAL,
            MeasureType.BINARY,
        )

    def validate_value(self, value: Any) -> list[str]:
        """Validate a single value against this variable's rules.

        Returns a list of warning/error messages.
        """
        warnings: list[str] = []
        if value is None or (isinstance(value, float) and value != value):
            return warnings
        if self.allowed_min is not None:
            try:
                if float(value) < self.allowed_min:  # type: ignore[arg-type]
                    warnings.append(
                        f"Value {value} below minimum {self.allowed_min}"
                    )
            except (ValueError, TypeError):
                pass
        if self.allowed_max is not None:
            try:
                if float(value) > self.allowed_max:  # type: ignore[arg-type]
                    warnings.append(
                        f"Value {value} above maximum {self.allowed_max}"
                    )
            except (ValueError, TypeError):
                pass
        return warnings

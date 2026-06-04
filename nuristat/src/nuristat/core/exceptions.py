"""Custom exception hierarchy for the NuriStat package.

All exceptions raised by NuriStat inherit from the base
:class:`NuriStatError` to enable unified error handling across
the application.
"""


class NuriStatError(Exception):
    """Base exception for all NuriStat errors.

    Args:
        message: Human-readable description of the error.
        details: Optional dictionary with additional structured error
            information (e.g., field names, invalid values).
    """

    def __init__(
        self,
        message: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class VariableError(NuriStatError):
    """Exception raised for errors related to variable metadata operations.

    Examples include invalid variable names, incompatible type assignments,
    or missing required metadata fields.
    """


class DatasetError(NuriStatError):
    """Exception raised for errors related to dataset operations.

    Examples include shape mismatches, column synchronization failures,
    or data corruption detected during validation.
    """


class AnalysisError(NuriStatError):
    """Exception raised when a statistical analysis cannot be performed.

    This may occur due to insufficient sample size, violated assumptions,
    or incompatible variable roles for the requested procedure.
    """


class ImportError(NuriStatError):
    """Exception raised for errors during data import operations.

    Examples include unsupported file formats, encoding detection failures,
    or structural problems in the source data.
    """


class ValidationError(NuriStatError):
    """Exception raised when data or metadata fails validation checks.

    This covers schema violations, range errors, format mismatches,
    and any other constraint violations detected by the validation layer.
    """


class ProjectError(NuriStatError):
    """Exception raised for errors related to project save/load operations.

    Examples include corrupted project files, version mismatches,
    I/O failures, or missing required archive members.
    """


# ---------------------------------------------------------------------------
# I/O exceptions
# ---------------------------------------------------------------------------


class IORError(NuriStatError):
    """Base exception for I/O related errors."""

    def __init__(self, message: str = "", details: dict | None = None) -> None:
        super().__init__(message, details)


class FileReadError(IORError):
    """Raised when file reading fails."""

    def __init__(
        self, filepath: str = "", reason: str = "", details: dict | None = None
    ) -> None:
        d = details or {}
        d["filepath"] = filepath
        if reason:
            d["reason"] = reason
        msg = f"파일을 읽을 수 없습니다: {filepath}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, d)


class FileWriteError(IORError):
    """Raised when file writing fails."""

    def __init__(
        self, filepath: str = "", reason: str = "", details: dict | None = None
    ) -> None:
        d = details or {}
        d["filepath"] = filepath
        if reason:
            d["reason"] = reason
        msg = f"파일을 쓸 수 없습니다: {filepath}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, d)


class EncodingDetectionError(IORError):
    """Raised when encoding detection fails."""

    def __init__(
        self, filepath: str = "", reason: str = "", details: dict | None = None
    ) -> None:
        d = details or {}
        d["filepath"] = filepath
        if reason:
            d["reason"] = reason
        msg = f"인코딩을 감지할 수 없습니다: {filepath}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, d)


class DelimiterDetectionError(IORError):
    """Raised when delimiter detection fails."""

    def __init__(
        self, filepath: str = "", reason: str = "", details: dict | None = None
    ) -> None:
        d = details or {}
        d["filepath"] = filepath
        if reason:
            d["reason"] = reason
        msg = f"구분자를 감지할 수 없습니다: {filepath}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, d)


class ImportValidationError(IORError):
    """Raised when import validation fails."""

    def __init__(self, message: str = "", details: dict | None = None) -> None:
        super().__init__(message, details)


class ProjectStoreError(IORError):
    """Raised when project store operation fails."""

    def __init__(self, message: str = "", details: dict | None = None) -> None:
        super().__init__(message, details)

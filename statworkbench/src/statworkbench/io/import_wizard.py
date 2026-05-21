"""Import Wizard logic for StatWorkbench.

The ImportWizard guides users through a multi-step data import process:
file selection → encoding → delimiter → header → type preview → confirm.

Each step produces a dict of results, and ``validate_step`` checks for
warnings and errors at each stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from statworkbench.core.dataset import Dataset
from statworkbench.core.exceptions import (
    DelimiterDetectionError,
    EncodingDetectionError,
    FileReadError,
    ImportValidationError,
)
from statworkbench.io.csv_reader import _detect_delimiter, _detect_encoding


class ImportWizard:
    """Multi-step import wizard.

    The wizard maintains internal state across steps so that later
    stages can access information gathered in earlier stages.
    """

    # Supported encodings for the wizard
    ENCODINGS = [
        "auto",
        "utf-8",
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "latin-1",
        "ascii",
    ]

    # Supported delimiters
    DELIMITERS = [
        ("auto", "자동 감지"),
        (",", "쉼표 (Comma)"),
        ("\\t", "탭 (Tab)"),
        (";", "세미콜론 (Semicolon)"),
        ("|", "파이프 (Pipe)"),
        (" ", "공백 (Space)"),
    ]

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._errors: list[str] = []

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """Return the current wizard state."""
        return self._state.copy()

    @property
    def warnings(self) -> list[str]:
        """Return accumulated warnings."""
        return self._warnings.copy()

    @property
    def errors(self) -> list[str]:
        """Return accumulated errors."""
        return self._errors.copy()

    def reset(self) -> None:
        """Reset wizard state."""
        self._state = {}
        self._warnings = []
        self._errors = []

    # ------------------------------------------------------------------
    # Step 1: File selection
    # ------------------------------------------------------------------

    def step_file_select(self, filepath: str) -> dict[str, Any]:
        """Step 1 — Select file and gather basic info.

        Returns
        -------
        dict
            Keys: ``filepath``, ``filename``, ``file_size``,
            ``file_format``, ``exists``.
        """
        path = Path(filepath)
        result: dict[str, Any] = {
            "filepath": str(path.absolute()),
            "filename": path.name,
            "exists": path.exists() and path.is_file(),
            "file_size": path.stat().st_size if path.exists() else 0,
            "file_format": path.suffix.lower().lstrip("."),
        }
        self._state["file"] = result
        return result

    # ------------------------------------------------------------------
    # Step 2: Encoding
    # ------------------------------------------------------------------

    def step_encoding(
        self, filepath: str, encoding: str = "auto"
    ) -> dict[str, Any]:
        """Step 2 — Detect or confirm file encoding.

        Returns
        -------
        dict
            Keys: ``encoding``, ``detected``, ``preview_ok``.
        """
        detected = encoding
        if encoding == "auto":
            try:
                detected = _detect_encoding(filepath)
            except EncodingDetectionError as exc:
                self._errors.append(str(exc))
                detected = "utf-8"  # fallback

        # Quick preview check
        preview_ok = True
        try:
            with open(filepath, "r", encoding=detected) as f:
                sample = f.read(1024)
                if "\ufffd" in sample:
                    preview_ok = False
                    self._warnings.append(
                        "미리보기에 깨진 문자가 발견되었습니다. "
                        "다른 인코딩을 시도해 보세요."
                    )
        except (UnicodeDecodeError, OSError) as exc:
            preview_ok = False
            self._errors.append(f"미리보기 실패: {exc}")

        result = {
            "encoding": detected,
            "user_selected": encoding != "auto",
            "preview_ok": preview_ok,
        }
        self._state["encoding"] = result
        return result

    # ------------------------------------------------------------------
    # Step 3: Delimiter
    # ------------------------------------------------------------------

    def step_delimiter(
        self, filepath: str, encoding: str, delimiter: str = "auto"
    ) -> dict[str, Any]:
        """Step 3 — Detect or confirm delimiter.

        Returns
        -------
        dict
            Keys: ``delimiter``, ``detected``, ``n_columns``.
        """
        detected = delimiter
        n_columns = 0

        if delimiter == "auto":
            try:
                detected = _detect_delimiter(filepath, encoding)
            except DelimiterDetectionError as exc:
                self._warnings.append(str(exc))
                detected = ","  # fallback

        # Count columns from first line
        try:
            with open(filepath, "r", encoding=encoding) as f:
                first_line = f.readline()
                if first_line:
                    n_columns = first_line.count(detected) + 1
        except OSError:
            pass

        result = {
            "delimiter": detected,
            "user_selected": delimiter != "auto",
            "n_columns": n_columns,
        }
        self._state["delimiter"] = result
        return result

    # ------------------------------------------------------------------
    # Step 4: Header and row settings
    # ------------------------------------------------------------------

    def step_header(
        self,
        filepath: str,
        encoding: str,
        delimiter: str,
        header: int = 0,
        skip_rows: int = 0,
        max_rows: Optional[int] = None,
    ) -> dict[str, Any]:
        """Step 4 — Configure header and row options.

        Returns
        -------
        dict
            Keys: ``header``, ``skip_rows``, ``max_rows``, ``column_names``.
        """
        import csv as csv_mod

        column_names: list[str] = []
        try:
            with open(filepath, "r", encoding=encoding) as f:
                # Skip user-specified rows
                for _ in range(skip_rows):
                    next(f, None)
                # Read header row
                if header is not None:
                    for i, line in enumerate(f):
                        if i == header - skip_rows:
                            reader = csv_mod.reader([line], delimiter=delimiter)
                            rows = list(reader)
                            if rows:
                                column_names = rows[0]
                            break
        except (OSError, StopIteration):
            pass

        result = {
            "header": header,
            "skip_rows": skip_rows,
            "max_rows": max_rows,
            "column_names": column_names,
            "n_columns": len(column_names),
        }
        self._state["header"] = result
        return result

    # ------------------------------------------------------------------
    # Step 5: Type preview
    # ------------------------------------------------------------------

    def step_type_preview(
        self, dataset: Dataset
    ) -> dict[str, Any]:
        """Step 5 — Show inferred types for each column.

        Parameters
        ----------
        dataset : Dataset
            A preview Dataset (typically first N rows).

        Returns
        -------
        dict
            Keys: ``column_info`` (list of per-column type dicts),
            ``warnings``.
        """
        column_info: list[dict[str, Any]] = []
        warnings: list[str] = []

        for var_name, var_meta in dataset.variables.items():
            series = dataset.data[var_name]
            n_unique = series.nunique(dropna=True)
            n_total = len(series)
            missing_count = series.isna().sum()
            missing_rate = missing_count / n_total if n_total > 0 else 0.0

            col_info = {
                "name": var_name,
                "label": var_meta.label,
                "storage_type": var_meta.storage_type.value,
                "measure_type": var_meta.measure.value,
                "n_unique": int(n_unique),
                "missing_count": int(missing_count),
                "missing_rate": round(missing_rate, 4),
                "sample_values": series.dropna().head(5).tolist(),
            }
            column_info.append(col_info)

            # Warning checks
            if missing_rate > 0.5:
                warnings.append(
                    f"'{var_name}': 결측률이 {missing_rate*100:.1f}%입니다"
                )
            if var_meta.measure.value == "nominal" and n_unique > 50:
                warnings.append(
                    f"'{var_name}': 범주 수가 너무 많습니다 ({n_unique})"
                )
            if series.dtype == "object":
                # Check for mixed types
                try:
                    pd.to_numeric(series, errors="raise")
                except (ValueError, TypeError):
                    pass  # Expected for string columns

        result = {
            "column_info": column_info,
            "warnings": warnings,
        }
        self._state["type_preview"] = result
        return result

    # ------------------------------------------------------------------
    # Step 6: Final confirmation
    # ------------------------------------------------------------------

    def step_confirm(self, dataset: Dataset) -> dict[str, Any]:
        """Step 6 — Final confirmation with summary.

        Returns
        -------
        dict
            Keys: ``n_rows``, ``n_columns``, ``warnings``, ``ready``.
        """
        warnings = self._warnings.copy()

        # Check for duplicate variable names
        var_names = list(dataset.data.columns)
        seen: set[str] = set()
        for name in var_names:
            if name in seen:
                warnings.append(f"중복된 변수명: '{name}'")
            if name == "" or pd.isna(name):
                warnings.append("빈 변수명이 있습니다")
            seen.add(name)

        result = {
            "n_rows": dataset.n_rows,
            "n_columns": dataset.n_vars,
            "warnings": warnings,
            "errors": self._errors,
            "ready": len(self._errors) == 0,
        }
        self._state["confirm"] = result
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_step(self, step: str, data: dict[str, Any]) -> list[str]:
        """Validate a wizard step and return warnings/errors.

        Parameters
        ----------
        step : str
            Step name: ``"file_select"``, ``"encoding"``, ``"delimiter"``,
            ``"header"``, ``"type_preview"``, ``"confirm"``.
        data : dict
            The step result dictionary.

        Returns
        -------
        list[str]
            List of warning/error messages.  Empty list means no issues.
        """
        messages: list[str] = []

        if step == "file_select":
            if not data.get("exists", False):
                messages.append("파일이 존재하지 않습니다")
            if data.get("file_size", 0) == 0:
                messages.append("파일이 비어 있습니다")
            supported = {"csv", "txt", "tsv", "xlsx", "xls"}
            fmt = data.get("file_format", "").lower()
            if fmt not in supported:
                messages.append(f"지원하지 않는 파일 형식입니다: .{fmt}")

        elif step == "encoding":
            if not data.get("preview_ok", True):
                messages.append(
                    "인코딩 미리보기에 문제가 있습니다. "
                    "다른 인코딩을 선택해 보세요."
                )

        elif step == "delimiter":
            if data.get("n_columns", 0) < 2:
                messages.append(
                    "감지된 열 수가 1개입니다. 구분자를 확인하세요."
                )

        elif step == "header":
            col_names = data.get("column_names", [])
            if not col_names:
                messages.append("헤더(열 이름)를 읽을 수 없습니다")
            else:
                empty_names = [c for c in col_names if c == ""]
                if empty_names:
                    messages.append(
                        f"빈 열 이름이 {len(empty_names)}개 있습니다"
                    )

        elif step == "type_preview":
            for col in data.get("column_info", []):
                if col["missing_rate"] > 0.5:
                    messages.append(
                        f"'{col['name']}': 결측률 {col['missing_rate']*100:.1f}%"
                    )

        elif step == "confirm":
            if data.get("n_rows", 0) == 0:
                messages.append("데이터가 없습니다")
            if data.get("n_columns", 0) == 0:
                messages.append("열(변수)이 없습니다")

        return messages

    # ------------------------------------------------------------------
    # Convenience: run full wizard
    # ------------------------------------------------------------------

    def run_full_wizard(
        self,
        filepath: str,
        encoding: str = "auto",
        delimiter: str = "auto",
        header: int = 0,
        skip_rows: int = 0,
        max_rows: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run all wizard steps and return the combined state.

        This is a convenience method for non-interactive (programmatic)
        imports where user confirmation is not needed.

        Returns
        -------
        dict
            Combined wizard state with all step results.
        """
        self.reset()

        # Step 1
        file_info = self.step_file_select(filepath)

        # Step 2
        enc_info = self.step_encoding(filepath, encoding)

        # Step 3
        delim_info = self.step_delimiter(
            filepath, enc_info["encoding"], delimiter
        )

        # Step 4
        header_info = self.step_header(
            filepath,
            enc_info["encoding"],
            delim_info["delimiter"],
            header=header,
            skip_rows=skip_rows,
            max_rows=max_rows,
        )

        # Build a preview Dataset by reading the actual file
        fmt = file_info.get("file_format", "").lower()
        dataset: Optional[Dataset] = None

        try:
            if fmt in ("csv", "txt", "tsv"):
                from statworkbench.io.csv_reader import read_csv

                dataset = read_csv(
                    filepath,
                    encoding=enc_info["encoding"],
                    delimiter=delim_info["delimiter"],
                    header=header,
                    skiprows=skip_rows if skip_rows > 0 else None,
                    nrows=max_rows,
                )
            elif fmt in ("xlsx", "xls"):
                from statworkbench.io.excel_reader import read_excel

                dataset = read_excel(
                    filepath,
                    header=header,
                    skiprows=skip_rows if skip_rows > 0 else None,
                    nrows=max_rows,
                )
        except FileReadError as exc:
            self._errors.append(str(exc))

        # Step 5
        if dataset is not None:
            self.step_type_preview(dataset)
            # Step 6
            self.step_confirm(dataset)

        return self.state

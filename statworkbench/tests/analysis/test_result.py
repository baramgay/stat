"""Tests for AnalysisResult and ResultTable models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from statworkbench.analysis.result import AnalysisResult, ResultTable


class TestResultTable:
    """Tests for ResultTable construction and methods."""

    def test_create_minimal(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        rt = ResultTable(title="Test Table", dataframe=df)
        assert rt.title == "Test Table"
        assert rt.dataframe.shape == (2, 2)
        assert rt.footnotes == []
        assert rt.format_rules == {}
        assert rt.export_options == {}

    def test_create_full(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        rt = ResultTable(
            title="Full Table",
            dataframe=df,
            footnotes=["Note 1", "Note 2"],
            format_rules={"A": ".2f"},
            export_options={"index": True},
        )
        assert rt.footnotes == ["Note 1", "Note 2"]
        assert rt.format_rules == {"A": ".2f"}

    def test_serialize_dataframe(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        rt = ResultTable(title="Test", dataframe=df)
        # Trigger serialization via model_dump
        data = rt.model_dump()
        assert "dataframe" in data
        assert data["dataframe"]["columns"] == ["A", "B"]
        assert len(data["dataframe"]["data"]) == 2
        assert data["dataframe"]["shape"] == [2, 2]

    def test_to_html(self) -> None:
        df = pd.DataFrame({"A": [1, 2]})
        rt = ResultTable(title="HTML Test", dataframe=df)
        html = rt.to_html()
        assert "HTML Test" in html
        assert "<table" in html
        assert "</table>" in html

    def test_to_html_with_footnotes(self) -> None:
        df = pd.DataFrame({"A": [1, 2]})
        rt = ResultTable(
            title="HTML Test",
            dataframe=df,
            footnotes=["p < .05 is significant."],
        )
        html = rt.to_html()
        assert "p < .05 is significant." in html

    def test_to_markdown(self) -> None:
        df = pd.DataFrame({"A": [1, 2]})
        rt = ResultTable(title="MD Test", dataframe=df)
        md = rt.to_markdown()
        assert "MD Test" in md
        assert "|" in md  # markdown table

    def test_to_csv(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        rt = ResultTable(title="CSV Test", dataframe=df)
        csv_text = rt.to_csv()
        lines = csv_text.strip().split("\n")
        assert lines[0] == "A,B"
        assert len(lines) == 3  # header + 2 rows


class TestAnalysisResult:
    """Tests for AnalysisResult construction and methods."""

    def test_create_minimal(self) -> None:
        result = AnalysisResult(
            id="test-001",
            title="Test Analysis",
            created_at=datetime(2025, 1, 1, 12, 0, 0),
        )
        assert result.id == "test-001"
        assert result.title == "Test Analysis"
        assert result.created_at == datetime(2025, 1, 1, 12, 0, 0)
        assert result.spec == {}
        assert result.notes == []
        assert result.warnings == []
        assert result.tables == []
        assert result.text_blocks == []
        assert result.assumptions == []
        assert result.diagnostics == []
        assert result.figures == []
        assert result.syntax == ""
        assert result.metadata == {}

    def test_create_full(self) -> None:
        df = pd.DataFrame({"M": [1.0]})
        table = ResultTable(title="Summary", dataframe=df)
        result = AnalysisResult(
            id="test-002",
            title="Full Analysis",
            created_at=datetime.now(),
            spec={"analysis_id": "t_test"},
            notes=["Note 1"],
            warnings=["Warning 1"],
            tables=[table],
            text_blocks=["Interpretation text"],
            syntax="TTEST ...",
            metadata={"author": "test"},
        )
        assert result.spec == {"analysis_id": "t_test"}
        assert len(result.tables) == 1
        assert result.syntax == "TTEST ..."

    def test_add_table_chaining(self) -> None:
        result = AnalysisResult(
            id="test-003",
            title="Chaining Test",
            created_at=datetime.now(),
        )
        df = pd.DataFrame({"X": [1]})
        table = ResultTable(title="T1", dataframe=df)
        result.add_table(table).add_table(
            ResultTable(title="T2", dataframe=df)
        )
        assert len(result.tables) == 2
        assert result.tables[0].title == "T1"
        assert result.tables[1].title == "T2"

    def test_add_warning(self) -> None:
        result = AnalysisResult(
            id="test-004",
            title="Warning Test",
            created_at=datetime.now(),
        )
        result.add_warning("Small sample size.")
        assert result.warnings == ["Small sample size."]

    def test_add_note(self) -> None:
        result = AnalysisResult(
            id="test-005",
            title="Note Test",
            created_at=datetime.now(),
        )
        result.add_note("Analysis completed successfully.")
        assert result.notes == ["Analysis completed successfully."]

    def test_add_assumption(self) -> None:
        result = AnalysisResult(
            id="test-006",
            title="Assumption Test",
            created_at=datetime.now(),
        )
        df = pd.DataFrame({"W": [0.95], "p": [0.30]})
        result.add_assumption(ResultTable(title="Normality", dataframe=df))
        assert len(result.assumptions) == 1

    def test_add_diagnostic(self) -> None:
        result = AnalysisResult(
            id="test-007",
            title="Diagnostic Test",
            created_at=datetime.now(),
        )
        df = pd.DataFrame({"VIF": [1.2]})
        result.add_diagnostic(ResultTable(title="VIF", dataframe=df))
        assert len(result.diagnostics) == 1

    def test_summary(self) -> None:
        result = AnalysisResult(
            id="test-008",
            title="Summary Test",
            created_at=datetime.now(),
        )
        result.add_warning("W1").add_warning("W2")
        summary = result.summary()
        assert "Summary Test" in summary
        assert "W1" in summary
        assert "W2" in summary

    def test_model_dump(self) -> None:
        """AnalysisResult should serialize via model_dump."""
        df = pd.DataFrame({"A": [1]})
        result = AnalysisResult(
            id="test-009",
            title="Dump Test",
            created_at=datetime(2025, 6, 1, 10, 0, 0),
            tables=[ResultTable(title="T", dataframe=df)],
        )
        data = result.model_dump()
        assert data["id"] == "test-009"
        assert data["title"] == "Dump Test"
        assert len(data["tables"]) == 1
        assert data["tables"][0]["title"] == "T"

    def test_model_dump_json(self) -> None:
        """AnalysisResult should serialize to JSON-compatible dict."""
        df = pd.DataFrame({"A": [1]})
        result = AnalysisResult(
            id="test-010",
            title="JSON Test",
            created_at=datetime(2025, 6, 1, 10, 0, 0),
            tables=[ResultTable(title="T", dataframe=df)],
        )
        data = result.model_dump(mode="json")
        assert isinstance(data["created_at"], str)

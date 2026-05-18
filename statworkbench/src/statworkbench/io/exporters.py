"""Result exporters for StatWorkbench.

Provides export of analysis results and tables to HTML, Markdown, and CSV
formats for reporting and downstream use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from statworkbench.core.exceptions import FileWriteError


def export_html(
    result: dict[str, Any],
    path: str,
    title: str = "StatWorkbench Results",
) -> None:
    """Export analysis results to an HTML file.

    Parameters
    ----------
    result : dict
        Analysis result dictionary.  Expected to contain a ``"tables"``
        key with a list of table dicts, each having ``"title"`` and
        ``"dataframe"`` (or ``"data"``) keys.
    path : str
        Destination file path.
    title : str
        HTML document title.

    Raises
    ------
    FileWriteError
        If the file cannot be written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tables_html = ""
    tables = result.get("tables", [])

    for table in tables:
        table_title = table.get("title", "")
        df = table.get("dataframe") or table.get("data")

        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except (ValueError, TypeError):
                continue

        tables_html += f"<h2>{table_title}</h2>\n" if table_title else ""
        tables_html += df.to_html(
            index=False,
            border=0,
            classes="stat-table",
            table_id=f"table-{table_title.replace(' ', '-').lower()}",
        )
        tables_html += "\n<br/>\n"

    # Also include text blocks
    text_blocks = result.get("text_blocks", [])
    text_html = ""
    for block in text_blocks:
        if isinstance(block, str):
            text_html += f"<p>{block}</p>\n"
        elif isinstance(block, dict):
            text_html += f"<p>{block.get('text', '')}</p>\n"

    # Warnings / notes
    notes = result.get("notes", [])
    notes_html = ""
    if notes:
        notes_html += "<h2>Notes</h2>\n<ul>\n"
        for note in notes:
            notes_html += f"<li>{note}</li>\n"
        notes_html += "</ul>\n"

    warnings = result.get("warnings", [])
    warnings_html = ""
    if warnings:
        warnings_html += '<h2 style="color: #c00;">Warnings</h2>\n<ul>\n'
        for w in warnings:
            warnings_html += f'<li style="color: #c00;">{w}</li>\n'
        warnings_html += "</ul>\n"

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2em; }}
  h1 {{ color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: .3em; }}
  h2 {{ color: #555; margin-top: 1.5em; }}
  .stat-table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  .stat-table th {{ background: #f0f4f8; padding: 8px 12px; text-align: left;
                     border-bottom: 2px solid #4a90d9; }}
  .stat-table td {{ padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }}
  .stat-table tr:nth-child(even) {{ background: #fafbfc; }}
</style>
</head>
<body>
<h1>{title}</h1>
{text_html}
{tables_html}
{notes_html}
{warnings_html}
</body>
</html>"""

    try:
        out_path.write_text(html_content, encoding="utf-8")
    except OSError as exc:
        raise FileWriteError(path, str(exc)) from exc


def export_markdown(
    result: dict[str, Any],
    path: str,
) -> None:
    """Export analysis results to a Markdown file.

    Parameters
    ----------
    result : dict
        Analysis result dictionary with ``"tables"``, ``"text_blocks"``,
        ``"notes"``, and ``"warnings"`` keys.
    path : str
        Destination file path.

    Raises
    ------
    FileWriteError
        If the file cannot be written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Text blocks
    for block in result.get("text_blocks", []):
        if isinstance(block, str):
            lines.append(block)
        elif isinstance(block, dict):
            lines.append(block.get("text", ""))
        lines.append("")

    # Tables
    for table in result.get("tables", []):
        title = table.get("title", "")
        df = table.get("dataframe") or table.get("data")

        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except (ValueError, TypeError):
                continue

        if title:
            lines.append(f"### {title}")
        lines.append(df.to_markdown(index=False))
        lines.append("")

    # Notes
    notes = result.get("notes", [])
    if notes:
        lines.append("### Notes")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    # Warnings
    warnings = result.get("warnings", [])
    if warnings:
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    content = "\n".join(lines)

    try:
        out_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise FileWriteError(path, str(exc)) from exc


def export_csv_table(
    table: dict[str, Any] | pd.DataFrame,
    path: str,
    **kwargs: Any,
) -> None:
    """Export a single table to a CSV file.

    Parameters
    ----------
    table : dict or pd.DataFrame
        The table to export.  If a dict, the ``"dataframe"`` or
        ``"data"`` key is used.
    path : str
        Destination file path.
    **kwargs
        Additional keyword arguments forwarded to
        ``DataFrame.to_csv``.

    Raises
    ------
    FileWriteError
        If the file cannot be written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve DataFrame
    if isinstance(table, dict):
        df = table.get("dataframe") or table.get("data")
    else:
        df = table

    if df is None:
        raise FileWriteError(path, "납품할 테이블 데이터가 없습니다")

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except (ValueError, TypeError) as exc:
            raise FileWriteError(
                path, f"테이블을 DataFrame으로 변환 실패: {exc}"
            ) from exc

    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig", **kwargs)
    except OSError as exc:
        raise FileWriteError(path, str(exc)) from exc

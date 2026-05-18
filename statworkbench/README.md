# StatWorkbench

> Menu-based desktop statistical package — an alternative to SPSS/MedCalc

## Overview

**StatWorkbench** is a desktop statistics package that lets researchers, clinicians, and data analysts load data, define variable properties, and run statistical analyses through a menu-driven interface — without writing code.

### Key Differentiator

Unlike spreadsheet tools, StatWorkbench is **variable-centric**. Each column carries rich metadata (measurement scale, value labels, missing rules, role) that enables the system to recommend appropriate analyses and prevent statistical mistakes.

## Features (MVP)

- **Data Import**: CSV, TXT, TSV, Excel (.xlsx), Clipboard
- **Variable Management**: SPSS-style Variable View with full metadata
- **Spreadsheet UI**: Data View with cell editing, sort, filter
- **Statistical Analysis**:
  - Descriptive Statistics & Frequencies
  - Normality Tests (Shapiro-Wilk)
  - Crosstabulation with Chi-square
  - Independent & Paired t-tests
  - One-way ANOVA
  - Non-parametric: Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
  - Correlation: Pearson, Spearman, Kendall
  - Linear Regression
- **Result Output**: Structured tables with p-values, CIs, effect sizes
- **Syntax Log**: Reproducible command log for every analysis
- **Project Save/Load**: `.swb` bundle format (ZIP + Parquet + JSON)

## Tech Stack

| Area | Technology |
|------|------------|
| GUI | PySide6 |
| Data | pandas, numpy |
| Statistics | scipy, statsmodels |
| Excel | openpyxl |
| Validation | pydantic |
| Testing | pytest, pytest-qt |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run application
python -m statworkbench

# Run tests
pytest
```

## Project Structure

```
statworkbench/
├── HERMES.md           # Project specification (authoritative)
├── src/statworkbench/
│   ├── core/           # Dataset, VariableMeta, enums
│   ├── io/             # Import/export, project storage
│   ├── analysis/       # Statistical analysis engine
│   ├── ui/             # PySide6 GUI components
│   ├── syntax/         # Syntax logging
│   └── viz_bridge/     # Visualization bridge
├── tests/              # Test suite
└── examples/           # Sample datasets
```

## License

MIT

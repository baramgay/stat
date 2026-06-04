# NuriStat

> Menu-based desktop statistical package — SPSS/MedCalc alternative

[![Tests](https://img.shields.io/badge/tests-4420%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-3.3.1-blue)](pyproject.toml)

## Overview

**NuriStat** is a desktop statistics package that lets researchers, clinicians, and data analysts load data, define variable properties, and run statistical analyses through a menu-driven interface — without writing code.

### Key Differentiator

Unlike spreadsheet tools, NuriStat is **variable-centric**. Each column carries rich metadata (measurement scale, value labels, missing rules, role) that enables the system to recommend appropriate analyses and prevent statistical mistakes.

## Features (v3.3.1 — 27 Analysis Modules)

### Data Management
- **Data Import**: CSV, TXT, TSV, Excel (.xlsx), SPSS (.sav), Clipboard
- **Variable Management**: SPSS-style Variable View with full metadata (scale, value labels, missing rules)
- **Spreadsheet UI**: Data View with cell editing, sort, filter, Formula Bar
- **Data Transformation**: Compute Variable, Recode, Visual Binning, Rank Cases
- **Data Operations**: Select Cases, Weight Cases, Sort, Merge Files, Pivot Tables

### Statistical Analysis

**Descriptive Statistics**
- Frequencies: frequency, percent, valid percent, cumulative percent
- Descriptives: mean, SD, min, max, skewness, kurtosis, SEM
- Explore: Shapiro-Wilk, Levene's test, stem-and-leaf, box plot, Q-Q plot
- Crosstabulation: chi-square, Fisher's exact, Phi, Cramer's V

**Mean Comparison**
- One-Sample T-Test
- Independent-Samples T-Test (Levene's test, Cohen's d)
- Paired-Samples T-Test
- One-Way ANOVA (Tukey HSD, Bonferroni, Scheffe post hoc)

**Correlation & Regression**
- Bivariate Correlation: Pearson, Spearman, Kendall
- Partial Correlation: controlling for covariates
- Linear Regression: R², VIF, standardized coefficients
- Logistic Regression: binary & multinomial, Odds Ratio, Hosmer-Lemeshow, AUC

**Nonparametric Tests**
- Mann-Whitney U, Wilcoxon Signed-Rank
- Kruskal-Wallis H, Friedman test
- Normality: Shapiro-Wilk, Kolmogorov-Smirnov
- Chi-Square Goodness of Fit

**Advanced Analysis**
- Factor Analysis (EFA/PCA): Varimax/Oblimin rotation, KMO, Bartlett's test
- Cluster Analysis: K-Means (elbow, silhouette), Hierarchical (dendrogram)
- Discriminant Analysis (LDA): Wilks' Lambda, canonical coefficients
- Survival Analysis: Kaplan-Meier, Log-rank, Cox proportional hazards

**Diagnostic & Agreement**
- ROC Analysis: AUC, optimal cutoff, 95% CI
- Sensitivity & Specificity: confusion matrix, PPV, NPV, likelihood ratios
- Cohen's Kappa: inter-rater agreement for categorical data
- ICC (Intraclass Correlation Coefficient): inter-rater reliability for continuous data
- Bland-Altman Plot: limits of agreement, method comparison

**Scale & Reliability**
- Reliability Analysis: Cronbach's α, item-total correlation, alpha-if-deleted

**Machine Learning**
- Logistic Regression, Decision Tree, Random Forest, SVM
- Train/test split, k-fold cross-validation, variable importance, confusion matrix

### Output & Utilities
- **Result Output**: Structured tables with p-values, CIs, effect sizes, footnotes
- **Chart Builder**: 7 chart types, real-time preview, 300 DPI PNG export
- **Advanced Visualization**: heatmap, scatter matrix, violin plot, forest plot
- **Syntax Log**: reproducible command log for every analysis
- **Project Save/Load**: `.swb` bundle format (ZIP + Parquet + JSON)
- **HTML Export**: full output to single HTML file
- **Data Quality Diagnosis**: missing patterns, outlier detection, duplicate records

## Tech Stack

| Area | Technology |
|------|------------|
| GUI | PySide6 6.6+ |
| Data | pandas, numpy |
| Statistics | scipy, statsmodels |
| ML | scikit-learn |
| Survival | lifelines |
| Excel | openpyxl |
| Encoding detection | chardet |
| Testing | pytest 9.0, pytest-cov |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run application
python -m nuristat

# Run tests (2988 tests, ~98% coverage)
pytest
```

## Project Structure

```
nuristat/
├── src/nuristat/
│   ├── core/           # Dataset, VariableMeta, enums, validation
│   ├── io/             # CSV/Excel/SPSS import, project storage
│   ├── analysis/       # statistical analysis modules
│   ├── ui/             # PySide6 GUI (dialogs, main window)
│   ├── syntax/         # Syntax logging and execution
│   └── viz_bridge/     # Visualization bridge
├── tests/
│   ├── analysis/       # 70+ unit test files
│   └── integration/    # Data entry & analysis integration tests
└── docs/
    └── user_manual.md  # Full Korean user manual (v3.0.0)
```

## License

MIT

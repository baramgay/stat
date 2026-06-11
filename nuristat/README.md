<div align="center">

<h1>NuriStat</h1>

<p><strong>Free, open-source SPSS alternative for researchers and clinicians</strong></p>

<p>
  <a href="https://github.com/baramgay/stat/releases"><img src="https://img.shields.io/github/v/release/baramgay/stat?style=flat-square&label=version&color=blue" alt="Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square" alt="Platform">
  <a href="https://github.com/baramgay/stat/actions"><img src="https://img.shields.io/github/actions/workflow/status/baramgay/stat/publish-pypi.yml?style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/baramgay/stat/stargazers"><img src="https://img.shields.io/github/stars/baramgay/stat?style=flat-square&color=yellow" alt="Stars"></a>
</p>

<p>
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-analysis-modules">Analysis Modules</a> •
  <a href="#-screenshots">Screenshots</a> •
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

**NuriStat** is a GUI-based statistical analysis package built with PySide6. It gives researchers, clinicians, and data analysts a familiar menu-driven workflow — define variables, manage data, and run analyses — without writing code. Think SPSS or MedCalc, but **free, open-source, and Python-powered**.

```
pip install nuristat      # coming soon to PyPI
python -m nuristat        # launch the application
```

---

## Why NuriStat?

| | SPSS | Excel | NuriStat |
|---|:---:|:---:|:---:|
| Free | ✗ | ✗ | **✓** |
| Variable metadata (labels, missing rules, scales) | ✓ | ✗ | **✓** |
| SPSS .sav import | ✓ | ✗ | **✓** |
| 27+ analysis modules | ✓ | limited | **✓** |
| Cohen's d, effect sizes | ✓ | ✗ | **✓** |
| Survival analysis (K-M, Cox) | ✓ | ✗ | **✓** |
| ROC / AUC | ✓ | ✗ | **✓** |
| ICC / Bland-Altman | ✓ | ✗ | **✓** |
| Open source | ✗ | ✗ | **✓** |

---

## Features

### Data Management
- **Import**: CSV, TSV, Excel (.xlsx), SPSS (.sav), Clipboard paste
- **Variable View**: SPSS-style metadata editor — measurement scale, value labels, missing value rules, variable roles
- **Data View**: Spreadsheet grid with cell editing, undo/redo, copy-paste, sort, filter, formula bar
- **Transformations**: Compute Variable, Recode, Visual Binning, Rank Cases, Select Cases, Weight Cases

### Statistical Analysis

<details>
<summary><b>Descriptive Statistics</b></summary>

- **Frequencies** — frequency, percent, valid percent, cumulative percent, bar chart
- **Descriptives** — N, mean, SD, median, IQR, min/max, skewness, kurtosis, 95% CI
- **Explore** — Shapiro-Wilk, Levene's test, stem-and-leaf plot, box plot, Q-Q plot
- **Crosstabulation** — chi-square, Fisher's exact test, Phi, Cramer's V, row/column percents

</details>

<details>
<summary><b>Mean Comparison</b></summary>

- **One-Sample T-Test** — test against a hypothesized mean
- **Independent-Samples T-Test** — Levene's test, equal/unequal variance variants, Cohen's d
- **Paired-Samples T-Test** — mean difference, SE, 95% CI, Cohen's dz
- **One-Way ANOVA** — Tukey HSD, Bonferroni, Scheffé post hoc; effect size η²
- **Two-Way ANOVA** — main effects, interaction, partial η²
- **ANCOVA** — adjusted means, homogeneity of regression slopes
- **Repeated Measures ANOVA** — sphericity test (Mauchly's W), Greenhouse-Geisser correction

</details>

<details>
<summary><b>Correlation & Regression</b></summary>

- **Bivariate Correlation** — Pearson, Spearman, Kendall; significance matrix
- **Partial Correlation** — controlling for one or more covariates
- **Linear Regression** — R², adjusted R², VIF, standardized β, ANOVA table, residual diagnostics
- **Binary Logistic Regression** — Odds Ratio, Hosmer-Lemeshow test, Nagelkerke R², ROC AUC
- **Multinomial Logistic Regression** — reference category, class-level OR

</details>

<details>
<summary><b>Nonparametric Tests</b></summary>

- Mann-Whitney U, Wilcoxon Signed-Rank
- Kruskal-Wallis H (with Dunn's post hoc), Friedman test
- Normality: Shapiro-Wilk, Kolmogorov-Smirnov
- Chi-Square Goodness of Fit

</details>

<details>
<summary><b>Advanced & Multivariate</b></summary>

- **Factor Analysis** (EFA/PCA) — Varimax/Oblimin rotation, KMO, Bartlett's test, scree plot
- **Cluster Analysis** — K-Means (elbow, silhouette), Hierarchical (Ward, complete linkage, dendrogram)
- **Discriminant Analysis** — Wilks' Lambda, canonical coefficients, classification table
- **MANOVA** — Pillai's trace, Wilks' Lambda, Hotelling's T², Roy's largest root

</details>

<details>
<summary><b>Survival & Diagnostic</b></summary>

- **Survival Analysis** — Kaplan-Meier curves, log-rank test, Cox proportional hazards (HR, 95% CI)
- **ROC Analysis** — AUC, optimal cutoff (Youden's J), 95% CI bootstrap
- **Sensitivity & Specificity** — confusion matrix, PPV, NPV, LR+/LR−
- **Cohen's Kappa** — inter-rater agreement with weighted Kappa option
- **ICC** — two-way mixed/random/fixed; absolute agreement vs consistency
- **Bland-Altman Plot** — limits of agreement, proportional bias test

</details>

<details>
<summary><b>Scale & Machine Learning</b></summary>

- **Reliability Analysis** — Cronbach's α, item-total correlation, α-if-item-deleted
- **Machine Learning** — Logistic Regression, Decision Tree, Random Forest, SVM; train/test split, k-fold CV, confusion matrix, feature importance

</details>

### Output & Visualization
- **Structured output pane** — tables with p-values, CIs, effect sizes, footnotes
- **Chart Builder** — 7 chart types, real-time preview, 300 DPI PNG/SVG export
- **Advanced visualization** — heatmap, scatter matrix, violin plot, forest plot
- **HTML export** — full output to a single portable HTML file
- **Syntax log** — reproducible command log auto-generated for every analysis
- **Project files** — `.swb` bundle (ZIP + Parquet + JSON metadata)

---

## Quick Start

### Requirements
- Python 3.10+
- Windows / macOS / Linux

### Install from source

```bash
git clone https://github.com/baramgay/stat.git
cd stat
pip install -e ".[dev]"
python -m nuristat
```

### Standalone installer (Windows)

Download the latest `.exe` installer from the [Releases](https://github.com/baramgay/stat/releases) page — no Python required.

---

## Screenshots

> Screenshots coming soon. See [Releases](https://github.com/baramgay/stat/releases) for demo data.

---

## Tech Stack

| Component | Technology |
|---|---|
| GUI framework | PySide6 6.6+ |
| Data layer | pandas 2.x, numpy |
| Statistics | scipy, statsmodels |
| Machine learning | scikit-learn |
| Survival analysis | lifelines |
| Visualization | matplotlib |
| Excel I/O | openpyxl |
| Testing | pytest 9.x, pytest-cov |

---

## Project Structure

```
nuristat/
├── src/nuristat/
│   ├── core/          # Dataset, VariableMeta, enums, validation
│   ├── io/            # CSV / Excel / SPSS import, project storage
│   ├── analysis/      # 27 statistical analysis modules
│   ├── ui/            # PySide6 GUI — main window, dialogs, views
│   └── i18n/          # Translations (English default, Korean available)
├── tests/
│   ├── analysis/      # Unit tests for every analysis module
│   └── integration/   # End-to-end data + analysis tests
└── docs/
    └── user_manual.md
```

---

## Language / 언어 설정

NuriStat defaults to **English**. Korean is available via **Settings → Language → 한국어**.

한국어로 변경하려면 **설정(Settings) → 언어(Language) → 한국어**를 선택하세요.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-analysis`
3. Run tests: `pytest`
4. Submit a pull request

**Good first issues:** analysis output formatting, new chart types, import format support.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with Python + PySide6 · Free alternative to SPSS, MedCalc, and similar commercial packages</sub>
</div>

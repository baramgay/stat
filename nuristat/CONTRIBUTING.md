# Contributing to NuriStat

Thank you for your interest in contributing! NuriStat is a free, open-source SPSS alternative and community contributions are what make it better.

## Ways to Contribute

- **Bug reports** — open a GitHub issue with steps to reproduce
- **Feature requests** — open an issue describing the statistical method or workflow
- **Code** — see below
- **Documentation** — improve the user manual, add examples
- **Translations** — help translate the UI to more languages

## Development Setup

```bash
git clone https://github.com/baramgay/stat.git
cd stat
pip install -e ".[dev]"
pytest                  # run all tests
python -m nuristat      # launch the app
```

## Code Guidelines

- Python 3.10+, PySide6 for UI
- Every new analysis module needs a corresponding test in `tests/analysis/`
- Statistical outputs must include effect sizes and confidence intervals where applicable
- Follow existing patterns in `src/nuristat/analysis/` for new analysis modules

## Adding a New Analysis Module

1. Create `src/nuristat/analysis/my_analysis.py` — implement `run_my_analysis(spec, dataset)`
2. Return an `AnalysisResult` with `ResultTable` objects
3. Register in `src/nuristat/analysis/registry.py`
4. Add a dialog in `src/nuristat/ui/dialogs/my_analysis_dialog.py`
5. Wire up in `main_window.py`
6. Write tests in `tests/analysis/test_my_analysis.py`

## Pull Request Checklist

- [ ] Tests pass: `pytest`
- [ ] New analysis module has at least one test
- [ ] No hardcoded Korean strings — use `t("key")` from `nuristat.i18n`
- [ ] Effect sizes and CIs included in statistical output

## Questions?

Open a GitHub issue or start a Discussion.

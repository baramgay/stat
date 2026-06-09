"""Assumption checks and missing-data utilities for statistical analyses.

Provides:
* Missing-data handling (``prepare_analysis_frame``)
* Case-processing summary table generation
* Normality tests (Shapiro-Wilk)
* Homogeneity-of-variance tests (Levene)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from nuristat.analysis.result import ResultTable
from nuristat.core.dataset import Dataset
from nuristat.core.typing import MissingPolicy

# ---------------------------------------------------------------------------
# Prepared analysis frame
# ---------------------------------------------------------------------------

@dataclass
class PreparedAnalysisFrame:
    """Result of ``prepare_analysis_frame``.

    Attributes
    ----------
    data : pd.DataFrame
        Cleaned data frame ready for analysis.
    n_total : int
        Total number of rows in the original dataset (before any filter).
    n_valid : int
        Number of rows after filter + missing-data removal.
    n_excluded : int
        Number of rows excluded (filtered out + missing).
    excluded_pct : float
        Percentage of rows excluded.
    n_filtered : int
        Rows remaining after applying the active case filter.
        Equals *n_total* when no filter is active.
    weight_var : str | None
        Name of the active weight variable, or ``None`` if no weight is set.
        Weight-aware analyses (frequencies, crosstabs, descriptives) should
        use ``data[weight_var]`` as frequency weights.
    """

    data: pd.DataFrame
    n_total: int
    n_valid: int
    n_excluded: int
    excluded_pct: float
    n_filtered: int = -1
    weight_var: str | None = None

    def __post_init__(self) -> None:
        # -1 is the "not set" sentinel; back-fill for callers that omit n_filtered
        if self.n_filtered < 0:
            self.n_filtered = self.n_total


# ---------------------------------------------------------------------------
# Missing-data helpers
# ---------------------------------------------------------------------------

def _apply_user_missing(
    df: pd.DataFrame,
    dataset: Dataset,
    variables: list[str],
) -> pd.DataFrame:
    """Replace user-defined missing values with NaN in a copy of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to modify.
    dataset : Dataset
        Source dataset containing variable metadata with missing-value rules.
    variables : list[str]
        Columns to process.

    Returns
    -------
    pd.DataFrame
        Modified data frame with user-defined missing values set to NaN.
    """
    df = df.copy()
    for var in variables:
        if var not in dataset.variables:
            continue
        meta = dataset.variables[var]
        if not meta.missing_values:
            continue
        for mv in meta.missing_values:
            if isinstance(mv, (list, tuple)) and len(mv) == 2:
                # Range missing: [min, max]
                lo, hi = mv
                mask = df[var].between(lo, hi, inclusive="both")
                df.loc[mask, var] = np.nan
            else:
                # Single-value missing
                df[var] = df[var].replace(mv, np.nan)
    return df


_FILTER_COLUMN = "filter_$"
# validate_variable_name() sanitises '$' → '_', so handle both column names
_FILTER_COLUMN_SAFE = "filter__"


def prepare_analysis_frame(
    dataset: Dataset,
    variables: list[str],
    missing_policy: MissingPolicy = MissingPolicy.LISTWISE,
    include_user_missing: bool = True,
    weight_var: str | None = None,
) -> PreparedAnalysisFrame:
    """Prepare a clean analysis frame after applying missing-data policy.

    Automatically honours the active case filter (``filter_$`` column) and
    the supplied weight variable so that downstream analysis functions see
    only the selected subset.

    Parameters
    ----------
    dataset : Dataset
        The source dataset.
    variables : list[str]
        Variable names to include in the analysis.
    missing_policy : MissingPolicy, default MissingPolicy.LISTWISE
        How to handle missing values.
    include_user_missing : bool, default True
        If True, also treat user-defined missing values as missing.
    weight_var : str | None, default None
        Name of the active weight variable.  When set, the weight column is
        appended to ``data`` so callers can apply frequency weights.

    Returns
    -------
    PreparedAnalysisFrame
        Cleaned data and case-count summary.

    Raises
    ------
    ValueError
        If *variables* contains names not in the dataset.
    """
    df = dataset.data
    n_total = len(df)

    # ------------------------------------------------------------------
    # 1. Apply active case filter (select_cases_dialog writes filter_$ = 0/1)
    # Note: validate_variable_name() may rename filter_$ → filter__ on init;
    # select_cases_dialog writes directly after init so the column stays filter_$.
    # Check both names for robustness.
    # ------------------------------------------------------------------
    n_filtered = n_total
    _filter_col = (
        _FILTER_COLUMN if _FILTER_COLUMN in df.columns
        else (_FILTER_COLUMN_SAFE if _FILTER_COLUMN_SAFE in df.columns else None)
    )
    if _filter_col is not None:
        filter_mask = df[_filter_col] == 1
        df = df[filter_mask]
        n_filtered = len(df)

    # Validate variable names (after filter so variable check is meaningful)
    all_needed = list(variables)
    if weight_var and weight_var not in all_needed:
        all_needed.append(weight_var)

    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        raise ValueError(
            f"Variable(s) not found in dataset: {missing_vars}"
        )

    # ------------------------------------------------------------------
    # 2. Subset to requested columns (+ weight if needed)
    # ------------------------------------------------------------------
    cols_to_keep = [c for c in all_needed if c in df.columns]
    subset = df[cols_to_keep].copy()

    # Apply user-defined missing rules if requested
    if include_user_missing:
        subset = _apply_user_missing(subset, dataset, variables)

    # ------------------------------------------------------------------
    # 3. Apply missing policy (on analysis variables only, not weight col)
    # ------------------------------------------------------------------
    analysis_cols = [c for c in variables if c in subset.columns]
    if missing_policy == MissingPolicy.LISTWISE:
        valid_mask = subset[analysis_cols].notna().all(axis=1)
        clean = subset[valid_mask]
    elif missing_policy == MissingPolicy.PAIRWISE:
        # For pairwise, return the full subset; analysis functions handle
        # missing values per-pair themselves.  We still report total N.
        clean = subset
    elif missing_policy == MissingPolicy.ANALYSIS_DEFAULT:
        # Default to listwise for most analyses
        valid_mask = subset[analysis_cols].notna().all(axis=1)
        clean = subset[valid_mask]
    elif missing_policy == MissingPolicy.INCLUDE_AS_CATEGORY:
        # Keep all rows; missing values will be treated as a category
        clean = subset
    elif missing_policy == MissingPolicy.EXCLUDE_USER_MISSING_ONLY:
        # Only exclude user-defined missing, keep system missing
        clean = subset
    else:
        valid_mask = subset[analysis_cols].notna().all(axis=1)
        clean = subset[valid_mask]

    n_valid = len(clean)
    n_excluded = n_total - n_valid
    excluded_pct = (n_excluded / n_total * 100) if n_total > 0 else 0.0

    # Resolve actual weight var (only if present in clean)
    resolved_weight = weight_var if (weight_var and weight_var in clean.columns) else None

    return PreparedAnalysisFrame(
        data=clean,
        n_total=n_total,
        n_valid=n_valid,
        n_excluded=n_excluded,
        excluded_pct=excluded_pct,
        n_filtered=n_filtered,
        weight_var=resolved_weight,
    )


# ---------------------------------------------------------------------------
# Case Processing Summary
# ---------------------------------------------------------------------------

def get_case_processing_summary(
    n_total: int,
    n_valid: int,
    n_excluded: int,
    excluded_pct: float | None = None,
) -> ResultTable:
    """Build a *Case Processing Summary* result table.

    Parameters
    ----------
    n_total : int
        Total cases.
    n_valid : int
        Valid (included) cases.
    n_excluded : int
        Excluded cases.
    excluded_pct : float, optional
        Percentage excluded.  Computed automatically if omitted.

    Returns
    -------
    ResultTable
        The summary table.
    """
    if excluded_pct is None:
        excluded_pct = (n_excluded / n_total * 100) if n_total > 0 else 0.0

    df = pd.DataFrame({
        "Total Cases": [n_total],
        "Valid Cases": [n_valid],
        "Excluded Cases": [n_excluded],
        "Excluded %": [f"{excluded_pct:.1f}%"],
    })

    return ResultTable(
        title="Case Processing Summary",
        dataframe=df,
        footnotes=[
            "Missing values were excluded listwise."
        ],
    )


def get_cps_table_kr(
    n_total: int,
    n_valid: int,
    n_excluded: int,
) -> ResultTable:
    """한글 Case Processing Summary 테이블 생성.

    Parameters
    ----------
    n_total : int
        전체 케이스 수.
    n_valid : int
        유효 케이스 수.
    n_excluded : int
        제외된 케이스 수.
    """
    valid_pct = round(n_valid / n_total * 100, 1) if n_total > 0 else 0.0
    excl_pct = round(n_excluded / n_total * 100, 1) if n_total > 0 else 0.0

    df = pd.DataFrame({
        "구분": ["유효", "제외됨", "합계"],
        "N": [n_valid, n_excluded, n_total],
        "%": [valid_pct, excl_pct, 100.0],
    })
    return ResultTable(
        title="Case Processing Summary",
        dataframe=df,
        footnotes=["결측값은 listwise 방식으로 제외됩니다."],
    )


# ---------------------------------------------------------------------------
# Normality check
# ---------------------------------------------------------------------------

def check_normality(
    data: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Test normality using the Shapiro-Wilk test.

    For samples with ``n > 5000`` a warning is issued because the
    Shapiro-Wilk test may not be reliable; the method still returns
    the statistic and p-value from scipy.

    Parameters
    ----------
    data : pd.Series
        Numeric data (missing values are dropped automatically).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        * ``test`` — test name (``"Shapiro-Wilk"``)
        * ``statistic`` — test statistic
        * ``p_value`` — p-value
        * ``n`` — sample size used
        * ``normal`` — ``True`` if *p* >= *alpha* (assumption met)
        * ``warnings`` — list of interpretive warnings
    """
    clean = data.dropna()
    n = len(clean)
    warnings: list[str] = []

    if n < 3:
        return {
            "test": "Shapiro-Wilk",
            "statistic": np.nan,
            "p_value": np.nan,
            "n": n,
            "normal": False,
            "warnings": ["Sample size < 3. Normality test cannot be performed."],
        }

    if n > 5000:
        warnings.append(
            "Sample size > 5,000. Shapiro-Wilk test may be overly sensitive "
            "to minor departures from normality. Consider visual inspection."
        )

    statistic, p_value = stats.shapiro(clean)
    normal = p_value >= alpha

    if n < 20:
        warnings.append(
            "Sample size is small (< 20). The test has low power to detect "
            "non-normality."
        )

    return {
        "test": "Shapiro-Wilk",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": n,
        "normal": bool(normal),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Homogeneity of variance
# ---------------------------------------------------------------------------

def check_homogeneity_of_variance(
    *groups: pd.Series | np.ndarray | list,
    alpha: float = 0.05,
    center: str = "median",
) -> dict:
    """Test homogeneity of variance using Levene's test.

    Parameters
    ----------
    *groups : array-like
        Two or more sample arrays.
    alpha : float, default 0.05
        Significance level.
    center : {"median", "mean", "trimmed"}, default "median"
        The center statistic for Levene's test.
        ``"median"`` → Brown-Forsythe variant (robust to skewness).

    Returns
    -------
    dict
        * ``test`` — ``"Levene"`` (or ``"Brown-Forsythe"`` when center="median")
        * ``statistic`` — test statistic
        * ``p_value`` — p-value
        * ``alpha`` — significance level
        * ``homogeneous`` — ``True`` if *p* >= *alpha*
        * ``warnings`` — list of interpretive warnings
    """
    # Convert inputs to arrays and drop NaN
    clean_groups = []
    for g in groups:
        arr = np.asarray(g, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        return {
            "test": "Levene",
            "statistic": np.nan,
            "p_value": np.nan,
            "alpha": alpha,
            "homogeneous": False,
            "warnings": ["At least 2 non-empty groups are required."],
        }

    statistic, p_value = stats.levene(*clean_groups, center=center)

    test_name = "Brown-Forsythe" if center == "median" else "Levene"
    homogeneous = p_value >= alpha

    warnings: list[str] = []
    total_n = sum(len(g) for g in clean_groups)
    if total_n < 20:
        warnings.append(
            "Small total sample size. Levene's test has limited power."
        )

    return {
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "homogeneous": bool(homogeneous),
        "warnings": warnings,
    }


def check_homogeneity_of_variance_from_groups(
    data: pd.Series,
    group: pd.Series,
    alpha: float = 0.05,
    center: str = "median",
) -> dict:
    """Convenience wrapper: split *data* by *group* and run Levene's test.

    Parameters
    ----------
    data : pd.Series
        Numeric dependent variable.
    group : pd.Series
        Grouping variable.
    alpha : float, default 0.05
        Significance level.
    center : str, default "median"
        Center statistic for Levene's test.

    Returns
    -------
    dict
        Same structure as :func:`check_homogeneity_of_variance`.
    """
    combined = pd.DataFrame({"data": data, "group": group}).dropna()
    groups = [
        combined.loc[combined["group"] == level, "data"].values
        for level in combined["group"].unique()
    ]
    return check_homogeneity_of_variance(*groups, alpha=alpha, center=center)


# ---------------------------------------------------------------------------
# Convenience wrappers with simpler signatures
# ---------------------------------------------------------------------------

def levene_test(
    *groups: np.ndarray | list | pd.Series,
    center: str = "median",
) -> tuple[float, float]:
    """Simple Levene test returning (statistic, p_value).

    Parameters
    ----------
    *groups : array-like
        Two or more sample arrays.
    center : str, default "median"
        Center statistic for Levene's test.

    Returns
    -------
    tuple[float, float]
        (statistic, p_value)
    """
    result = check_homogeneity_of_variance(*groups, center=center)
    return result["statistic"], result["p_value"]


def shapiro_test(data: np.ndarray | list | pd.Series) -> tuple[float, float]:
    """Simple Shapiro-Wilk test returning (statistic, p_value).

    Parameters
    ----------
    data : array-like
        Numeric data.

    Returns
    -------
    tuple[float, float]
        (statistic, p_value)
    """
    if isinstance(data, (np.ndarray, list)):
        data = pd.Series(data)
    result = check_normality(data)
    return result["statistic"], result["p_value"]

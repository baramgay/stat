"""Shared test fixtures for NuriStat."""

import sys
from pathlib import Path

# Ensure src/ is on the path before any nuristat imports.
_PROJECT_ROOT = Path(__file__).parent.parent
src_path = str(_PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from nuristat.core.dataset import Dataset
    from nuristat.core.variable import VariableMeta
    from nuristat.core.typing import StorageType, MeasureType, Role
except ModuleNotFoundError:
    # If the package is not found, add src/ and retry
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from nuristat.core.dataset import Dataset
    from nuristat.core.variable import VariableMeta
    from nuristat.core.typing import StorageType, MeasureType, Role

import pytest
import pandas as pd
import numpy as np


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_output_language():
    """매 테스트마다 분석 결과 출력 언어를 영어로 리셋 — 테스트 격리.

    앱은 기본 한국어지만, MainWindow 생성 등으로 i18n 전역이 'ko'로 바뀌어
    다른 테스트(영문 출력 단정)에 누수되지 않도록 고정한다.
    """
    from nuristat.core import i18n
    i18n.set_language("en")
    yield
    i18n.set_language("en")


@pytest.fixture
def empty_dataset():
    """Return an empty dataset."""
    df = pd.DataFrame()
    return Dataset(df, name="Empty")


@pytest.fixture
def sample_dataset():
    """Return a sample numeric dataset for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "score": [78.5, 82.3, 75.1, 88.7, 91.2, 85.6, 72.4, 76.8, 79.3],
        "age": [25, 30, 35, 40, 45, 50, 28, 33, 38],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M"],
    })
    ds = Dataset(df, name="Sample")
    ds.variables["group"].measure = MeasureType.NOMINAL
    ds.variables["score"].measure = MeasureType.SCALE
    ds.variables["age"].measure = MeasureType.SCALE
    ds.variables["gender"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def clinical_dataset():
    """Return the clinical dataset from fixtures."""
    path = FIXTURES_DIR / "sample_clinical.csv"
    try:
        from nuristat.io.csv_reader import read_csv
        return read_csv(str(path))
    except Exception:
        pytest.skip("csv_reader not available")


@pytest.fixture
def small_numeric_dataset():
    """Return a small numeric dataset for simple tests."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "y": [2.1, 4.0, 6.1, 7.8, 10.2, 12.0, 14.1, 15.9, 18.0, 20.2],
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    ds = Dataset(df, name="SmallNumeric")
    ds.variables["x"].measure = MeasureType.SCALE
    ds.variables["y"].measure = MeasureType.SCALE
    ds.variables["group"].measure = MeasureType.BINARY
    return ds


@pytest.fixture
def paired_dataset():
    """Return a dataset suitable for paired tests."""
    np.random.seed(123)
    df = pd.DataFrame({
        "pre_test": [72, 68, 75, 80, 65, 70, 78, 74, 69, 73],
        "post_test": [78, 72, 80, 85, 70, 75, 82, 79, 74, 77],
    })
    ds = Dataset(df, name="Paired")
    ds.variables["pre_test"].measure = MeasureType.SCALE
    ds.variables["post_test"].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def categorical_dataset():
    """Return a dataset with categorical variables for crosstab."""
    df = pd.DataFrame({
        "treatment": ["Drug", "Drug", "Drug", "Drug", "Drug",
                      "Placebo", "Placebo", "Placebo", "Placebo", "Placebo"],
        "response": ["Yes", "Yes", "No", "Yes", "No",
                     "No", "No", "Yes", "No", "No"],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    })
    ds = Dataset(df, name="Categorical")
    ds.variables["treatment"].measure = MeasureType.NOMINAL
    ds.variables["response"].measure = MeasureType.BINARY
    ds.variables["sex"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def anova_dataset():
    """Return a dataset with multiple groups for ANOVA."""
    np.random.seed(42)
    df = pd.DataFrame({
        "group": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
        "score": (
            np.random.normal(75, 8, 12).tolist() +
            np.random.normal(82, 10, 12).tolist() +
            np.random.normal(70, 7, 12).tolist()
        ),
    })
    ds = Dataset(df, name="ANOVA")
    ds.variables["group"].measure = MeasureType.NOMINAL
    ds.variables["score"].measure = MeasureType.SCALE
    return ds


@pytest.fixture
def nonparametric_dataset():
    """Return a dataset for nonparametric tests."""
    df = pd.DataFrame({
        "group": ["A"] * 8 + ["B"] * 8,
        "score": [12, 15, 18, 14, 16, 19, 13, 17, 25, 28, 22, 27, 24, 26, 23, 29],
        "time1": [5, 7, 6, 8, 4, 6, 7, 5, 6, 8, 5, 7, 6, 8, 7, 5],
        "time2": [7, 9, 8, 10, 6, 8, 9, 7, 8, 10, 7, 9, 8, 10, 9, 7],
        "time3": [9, 11, 10, 12, 8, 10, 11, 9, 10, 12, 9, 11, 10, 12, 11, 9],
    })
    ds = Dataset(df, name="Nonparametric")
    ds.variables["group"].measure = MeasureType.NOMINAL
    ds.variables["score"].measure = MeasureType.ORDINAL
    ds.variables["time1"].measure = MeasureType.ORDINAL
    ds.variables["time2"].measure = MeasureType.ORDINAL
    ds.variables["time3"].measure = MeasureType.ORDINAL
    return ds

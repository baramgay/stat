"""Local conftest for core tests - overrides parent conftest."""

import sys
from pathlib import Path

# Ensure src/ is on the path
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
import pandas as pd
import numpy as np

from statworkbench.core.dataset import Dataset
from statworkbench.core.variable import VariableMeta
from statworkbench.core.typing import StorageType, MeasureType, Role

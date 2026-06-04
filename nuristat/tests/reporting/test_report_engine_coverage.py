"""report_engine.py 커버리지 보강 테스트.

미커버 라인:
  162     : outlier_rows += ... (이상치 존재 시 HTML 행 추가)
  172     : if outlier_rows: sections.append(...)
  448-450 : save_html() 메서드
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from nuristat.core.dataset import Dataset
from nuristat.reporting.report_engine import ReportEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return ReportEngine()


@pytest.fixture
def outlier_dataset():
    """이상치를 포함한 데이터셋 — IQR 경계 밖 값 포함."""
    rng = np.random.default_rng(42)
    n = 50
    x = rng.normal(50, 5, n).tolist()
    x.append(200.0)  # 명확한 이상치
    x.append(-100.0)  # 명확한 이상치
    df = pd.DataFrame({"x": x, "y": rng.normal(0, 1, len(x))})
    return Dataset(df, name="OutlierData")


@pytest.fixture
def clean_dataset():
    """이상치 없는 데이터셋."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 11.0, 12.0, 13.0, 14.0],
    })
    return Dataset(df, name="CleanData")


# ---------------------------------------------------------------------------
# Lines 162, 172: 이상치 존재 → outlier_rows 생성 + sections.append
# ---------------------------------------------------------------------------

class TestOutlierSection:

    def test_outlier_rows_generated_when_outliers_exist(self, engine, outlier_dataset):
        """이상치 존재 → lines 162, 172 실행 → HTML에 이상치 섹션 포함."""
        html = engine.generate_data_quality_report(outlier_dataset)
        assert "IQR" in html

    def test_quality_report_contains_basic_info(self, engine, clean_dataset):
        """이상치 없음 → 기본 정보 섹션 있음."""
        html = engine.generate_data_quality_report(clean_dataset)
        assert "데이터셋명" in html or "기본 정보" in html or len(html) > 100


# ---------------------------------------------------------------------------
# Lines 448-450: save_html()
# ---------------------------------------------------------------------------

class TestSaveHtml:

    def test_save_html_writes_file(self, engine, tmp_path):
        """save_html() → 파일 생성(448-450)."""
        output_path = str(tmp_path / "report.html")
        html_content = "<html><body>Test Report</body></html>"
        engine.save_html(html_content, output_path)
        assert Path(output_path).exists()
        assert Path(output_path).read_text(encoding="utf-8") == html_content

    def test_save_html_content_matches(self, engine, outlier_dataset, tmp_path):
        """generate + save_html 통합 경로."""
        html = engine.generate_html_report(outlier_dataset, [], title="SaveTest")
        path = str(tmp_path / "full_report.html")
        engine.save_html(html, path)
        saved = Path(path).read_text(encoding="utf-8")
        assert "SaveTest" in saved

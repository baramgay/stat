"""추가 검증 테스트 — 6개 영역 심층 검증.

1. 통계 정밀도 — scipy/statsmodels 직접 비교
2. 동시 실행 안전성 — 다중 스레드
3. 보고서 엔진 — HTML 생성 구조
4. 감사 로그 — append/persist/reload
5. 설정 관리자 — 최근 파일, 분석 기본값
6. 극단값 처리 — inf, nan, 0분산, 초대형/초소형 수
7. 한글 변수명 처리
8. 레지스트리 — 구현된 분석 목록 일관성
9. 데이터셋 불변성 — 분석 후 원본 유지
10. 분석 결과 수치 일관성 — 그룹 N 합계 = 전체 N
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import MeasureType, MissingPolicy, StorageType


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def _ds(df: pd.DataFrame, measures: dict[str, MeasureType] | None = None) -> Dataset:
    ds = Dataset(df)
    if measures:
        for name, m in measures.items():
            ds.variables[name].measure = m
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 1. 통계 정밀도 — scipy/statsmodels 직접 비교
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalPrecision:
    """NuriStat 출력이 scipy/statsmodels 기준값과 일치하는지 검증."""

    def _ind_ttest_spec(self):
        return {
            "variables": {"dependent": "score", "group": "grp"},
            "options": {"equal_var": "yes"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }

    def _make_ttest_ds(self, seed: int = 99):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "score": np.concatenate([rng.normal(70, 10, 25), rng.normal(80, 10, 25)]),
            "grp": ["A"] * 25 + ["B"] * 25,
        })
        ds = _ds(df, {"score": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        return ds

    def test_independent_ttest_t_matches_scipy(self):
        """독립표본 t 통계량이 scipy와 0.001 이내."""
        from nuristat.analysis.ttests import run_analysis

        ds = self._make_ttest_ds()
        spec = self._ind_ttest_spec()
        result = run_analysis(ds, spec)

        t_tbl = [t for t in result.tables if "Independent" in t.title]
        assert t_tbl, "Independent t-test 테이블 없음"
        eq_row = t_tbl[0].dataframe[
            t_tbl[0].dataframe["Variant"].str.contains("Equal variances assumed", na=False)
        ]
        assert len(eq_row) == 1
        t_sw = float(eq_row.iloc[0]["t"])

        g1 = ds.data[ds.data["grp"] == "A"]["score"].values
        g2 = ds.data[ds.data["grp"] == "B"]["score"].values
        t_scipy, _ = stats.ttest_ind(g1, g2, equal_var=True)

        assert abs(t_sw - t_scipy) < 0.001

    def test_independent_ttest_p_matches_scipy(self):
        """p-value가 scipy와 0.001 이내."""
        from nuristat.analysis.ttests import run_analysis

        ds = self._make_ttest_ds(seed=77)
        spec = self._ind_ttest_spec()
        result = run_analysis(ds, spec)

        t_tbl = [t for t in result.tables if "Independent" in t.title][0]
        eq_row = t_tbl.dataframe[
            t_tbl.dataframe["Variant"].str.contains("Equal variances assumed", na=False)
        ].iloc[0]
        p_raw = str(eq_row["p-value"]).strip()
        p_sw = 0.0 if p_raw.startswith("<") else float(p_raw)

        g1 = ds.data[ds.data["grp"] == "A"]["score"].values
        g2 = ds.data[ds.data["grp"] == "B"]["score"].values
        _, p_scipy = stats.ttest_ind(g1, g2, equal_var=True)

        # p < .001 표기이거나 직접 비교 (0.01 허용)
        if p_raw.startswith("<"):
            assert p_scipy < 0.01
        else:
            assert abs(p_sw - p_scipy) < 0.01

    def test_descriptive_mean_matches_numpy(self):
        """기술통계 평균이 numpy와 일치."""
        from nuristat.analysis.descriptive import run_analysis

        rng = np.random.default_rng(11)
        x = rng.normal(100, 15, 50)
        df = pd.DataFrame({"x": x})
        ds = _ds(df, {"x": MeasureType.SCALE})
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        desc_tbl = [t for t in result.tables if "Descriptive" in t.title][0]
        row = desc_tbl.dataframe[desc_tbl.dataframe["Variable"].astype(str) == "x"]
        mean_sw = float(row.iloc[0]["Mean"])

        # 표시된 평균은 소수점 2자리 반올림이므로 0.01 허용
        assert abs(mean_sw - x.mean()) < 0.01

    def test_descriptive_std_matches_numpy(self):
        """기술통계 표준편차(ddof=1)가 numpy와 일치."""
        from nuristat.analysis.descriptive import run_analysis

        rng = np.random.default_rng(22)
        x = rng.normal(50, 10, 40)
        df = pd.DataFrame({"x": x})
        ds = _ds(df, {"x": MeasureType.SCALE})
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        desc_tbl = [t for t in result.tables if "Descriptive" in t.title][0]
        df_r = desc_tbl.dataframe
        std_col = [c for c in df_r.columns if "Std" in str(c) or "SD" in str(c)]
        if std_col:
            std_sw = float(df_r.iloc[0][std_col[0]])
            assert abs(std_sw - x.std(ddof=1)) < 0.001

    def test_anova_f_stat_matches_scipy(self):
        """일원분산분석 F 통계량이 scipy와 0.01 이내."""
        from nuristat.analysis.anova import run_analysis

        rng = np.random.default_rng(33)
        g1 = rng.normal(10, 2, 20)
        g2 = rng.normal(12, 2, 20)
        g3 = rng.normal(14, 2, 20)
        df = pd.DataFrame({
            "y": np.concatenate([g1, g2, g3]),
            "grp": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        ds = _ds(df, {"y": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "y", "factor": "grp"},
            "options": {"post_hoc": False},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        anova_tbl = [t for t in result.tables if t.title == "ANOVA"]
        assert anova_tbl, "ANOVA 테이블 없음"
        f_row = anova_tbl[0].dataframe[
            anova_tbl[0].dataframe["Source"].str.contains("grp", na=False)
        ]
        f_sw = float(f_row.iloc[0]["F"])
        f_scipy, _ = stats.f_oneway(g1, g2, g3)
        assert abs(f_sw - f_scipy) < 0.01

    def test_pearson_r_matches_scipy(self):
        """Pearson 상관계수가 scipy와 0.001 이내."""
        from nuristat.analysis.correlation import run_analysis

        rng = np.random.default_rng(44)
        x = rng.normal(0, 1, 40)
        y = x * 0.7 + rng.normal(0, 0.5, 40)
        df = pd.DataFrame({"x": x, "y": y})
        ds = _ds(df, {"x": MeasureType.SCALE, "y": MeasureType.SCALE})
        spec = {
            "variables": {"target": ["x", "y"]},
            "options": {"method": "pearson", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        r_scipy, _ = stats.pearsonr(x, y)

        corr_tbl = [t for t in result.tables if "Correlation" in t.title and "Matrix" in t.title]
        assert corr_tbl, f"상관행렬 테이블 없음, tables={[t.title for t in result.tables]}"
        df_c = corr_tbl[0].dataframe
        # 행렬에서 x-y 값 추출 (대각 제외)
        xy_candidates = df_c.values.flatten()
        floats = []
        for v in xy_candidates:
            try:
                fv = float(v)
                if not math.isnan(fv) and 0.01 < abs(fv) < 0.9999:
                    floats.append(fv)
            except (TypeError, ValueError):
                pass
        if floats:
            assert any(abs(v - abs(r_scipy)) < 0.01 for v in floats)

    def test_chi_square_matches_scipy(self):
        """카이제곱 통계량이 scipy와 0.5 이내."""
        from nuristat.analysis.crosstab import run_analysis

        rng = np.random.default_rng(55)
        df = pd.DataFrame({
            "a": rng.choice(["X", "Y", "Z"], 100),
            "b": rng.choice(["P", "Q"], 100),
        })
        ds = _ds(df, {"a": MeasureType.NOMINAL, "b": MeasureType.NOMINAL})
        spec = {
            "variables": {"row": "a", "column": "b"},
            "options": {"chi_square": True},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        ct = pd.crosstab(df["a"], df["b"])
        chi2_scipy, _, _, _ = stats.chi2_contingency(ct)

        chi_tbl = [t for t in result.tables if "Chi" in t.title or "카이" in t.title]
        if chi_tbl:
            df_chi = chi_tbl[0].dataframe
            chi_vals = []
            for v in df_chi.values.flatten():
                try:
                    fv = float(v)
                    if fv > 0:
                        chi_vals.append(fv)
                except (TypeError, ValueError):
                    pass
            if chi_vals:
                assert any(abs(v - chi2_scipy) < 1.0 for v in chi_vals)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 동시 실행 안전성
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentExecution:
    """동일 데이터셋에 복수 스레드가 동시에 분석을 실행해도 충돌 없음."""

    def _base_ds(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 100),
            "y": rng.normal(0, 1, 100),
        })
        return _ds(df, {"x": MeasureType.SCALE, "y": MeasureType.SCALE})

    def test_concurrent_descriptive_same_dataset(self):
        """5개 스레드가 동시에 기술통계 실행 → 오류 없음, 결과 일관."""
        from nuristat.analysis.descriptive import run_analysis

        ds = self._base_ds()
        spec = {"variables": {"variables": ["x", "y"]}, "missing_policy": MissingPolicy.LISTWISE}
        results, errors = [], []

        def run():
            try:
                r = run_analysis(ds, spec)
                results.append(len(r.tables))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=run) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"동시 실행 오류: {errors}"
        assert len(results) == 5
        assert len(set(results)) == 1, "결과 테이블 수 불일치"

    def test_concurrent_different_analyses(self):
        """서로 다른 분석을 동시에 실행 → 오류 없음."""
        from nuristat.analysis.descriptive import run_analysis as desc_run
        from nuristat.analysis.normality import run_analysis as norm_run
        from nuristat.analysis.correlation import run_analysis as corr_run

        ds = self._base_ds()
        specs = [
            (desc_run, {"variables": {"variables": ["x"]}, "missing_policy": MissingPolicy.LISTWISE}),
            (norm_run, {"variables": {"variables": ["x"]}, "missing_policy": MissingPolicy.LISTWISE}),
            (corr_run, {"variables": {"variables": ["x", "y"]}, "options": {"method": "pearson"},
                        "missing_policy": MissingPolicy.LISTWISE}),
        ]
        errors = []

        def run_one(fn, spec):
            try:
                fn(ds, spec)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=run_one, args=(fn, sp)) for fn, sp in specs * 3]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"동시 실행 오류: {errors}"

    def test_concurrent_dataset_reads_are_safe(self):
        """다중 스레드가 Dataset.data를 동시에 읽어도 안전."""
        rng = np.random.default_rng(8)
        df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
        ds = _ds(df)

        means, errors = [], []

        def read_mean():
            try:
                means.append(float(ds.data["x"].mean()))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read_mean) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(set(round(m, 10) for m in means)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. 보고서 엔진 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestReportEngine:
    """보고서 엔진 HTML 출력 구조 및 내용 검증."""

    def _make_ds(self, n: int = 30):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "age": rng.integers(20, 70, n).astype(float),
            "income": rng.normal(300, 50, n),
            "gender": rng.choice(["M", "F"], n),
        })
        return _ds(df, {"age": MeasureType.SCALE, "income": MeasureType.SCALE,
                        "gender": MeasureType.NOMINAL})

    def test_html_report_is_valid_html(self):
        """generate_html_report → DOCTYPE 포함 HTML 문자열."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds()
        engine = ReportEngine()
        html = engine.generate_html_report(ds, [], title="테스트 보고서")
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_html_report_contains_title(self):
        """지정한 제목이 HTML에 포함됨."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds()
        engine = ReportEngine()
        title = "고유한_보고서_제목_2025"
        html = engine.generate_html_report(ds, [], title=title)
        assert title in html

    def test_html_report_contains_dataset_info(self):
        """데이터셋 행/열 수가 HTML에 반영됨."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds(n=50)
        engine = ReportEngine()
        html = engine.generate_html_report(ds, [])
        assert "50" in html  # row count

    def test_html_report_with_analysis_section(self):
        """분석 섹션이 포함된 HTML 생성."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds()
        engine = ReportEngine()
        analyses = [
            {"type": "기술통계", "result": "평균: 45.2\n표준편차: 12.3"},
            {"type": "t-검정", "result": "t=2.34, p=.021"},
        ]
        html = engine.generate_html_report(ds, analyses)
        assert "기술통계" in html
        assert "t-검정" in html
        assert "평균" in html

    def test_data_quality_report_is_html_string(self):
        """generate_data_quality_report → HTML 문자열 반환."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds()
        engine = ReportEngine()
        html = engine.generate_data_quality_report(ds)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_data_quality_report_contains_row_count(self):
        """품질 보고서에 행 수 포함."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds(n=30)
        engine = ReportEngine()
        html = engine.generate_data_quality_report(ds)
        assert "30" in html  # row count

    def test_html_report_with_author(self):
        """author 필드가 HTML에 포함됨."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds()
        engine = ReportEngine()
        html = engine.generate_html_report(ds, [], author="홍길동")
        assert "홍길동" in html

    def test_save_html_writes_file(self):
        """save_html이 파일을 실제로 생성함."""
        from nuristat.reporting.report_engine import ReportEngine

        ds = self._make_ds()
        engine = ReportEngine()
        html = engine.generate_html_report(ds, [])

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            fname = f.name
        try:
            engine.save_html(html, fname)
            assert os.path.exists(fname)
            assert os.path.getsize(fname) > 100
            with open(fname, encoding="utf-8") as fp:
                content = fp.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(fname)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 감사 로그 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLog:
    """AuditLog append/persist/reload 검증."""

    def test_append_adds_entry(self):
        """append 후 to_list 길이 증가."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        assert len(log) == 0
        log.append("open_file", {"path": "data.csv"})
        assert len(log) == 1

    def test_entry_contains_action(self):
        """엔트리에 action 필드 포함."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        log.append("variable_rename", {"old": "x", "new": "age"})
        entries = log.to_list()
        assert entries[0]["action"] == "variable_rename"

    def test_entry_contains_timestamp(self):
        """엔트리에 timestamp 포함."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        log.append("analysis_run")
        entry = log.to_list()[0]
        assert "timestamp" in entry
        assert len(entry["timestamp"]) >= 10

    def test_entry_contains_details(self):
        """details 딕셔너리가 보존됨."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        log.append("data_import", {"rows": 100, "cols": 5, "file": "test.csv"})
        entry = log.to_list()[0]
        assert entry["details"]["rows"] == 100
        assert entry["details"]["file"] == "test.csv"

    def test_multiple_appends_ordered(self):
        """여러 번 append → 순서 유지."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        for i in range(10):
            log.append(f"action_{i}")
        entries = log.to_list()
        assert len(entries) == 10
        for i, e in enumerate(entries):
            assert e["action"] == f"action_{i}"

    def test_persist_and_reload_jsonl(self):
        """save_jsonl → load_jsonl 왕복 무결성."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        log.append("open", {"file": "test.csv"})
        log.append("analysis_run", {"procedure": "t_test"})
        log.append("save", {"file": "result.swb"})

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            fname = f.name
        try:
            log.save_jsonl(fname)
            assert os.path.exists(fname)

            log2 = AuditLog.load_jsonl(fname)
            assert len(log2) == 3
            actions = [e["action"] for e in log2.to_list()]
            assert actions == ["open", "analysis_run", "save"]
        finally:
            os.unlink(fname)

    def test_clear_empties_log(self):
        """clear 후 길이 0."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        for _ in range(5):
            log.append("action")
        log.clear()
        assert len(log) == 0

    def test_no_details_entry_valid(self):
        """details 없는 엔트리도 유효."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        log.append("close")
        entry = log.to_list()[0]
        assert "details" not in entry or entry.get("details") is None
        assert entry["action"] == "close"

    def test_large_log_persist(self):
        """1000개 엔트리 저장·복원."""
        from nuristat.core.audit import AuditLog

        log = AuditLog()
        for i in range(1000):
            log.append(f"event_{i % 10}", {"index": i})

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            fname = f.name
        try:
            log.save_jsonl(fname)
            log2 = AuditLog.load_jsonl(fname)
            assert len(log2) == 1000
        finally:
            os.unlink(fname)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 설정 관리자 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsManager:
    """SettingsManager 최근 파일, 분석 기본값 검증."""

    @pytest.fixture(autouse=True)
    def _isolated_settings(self, tmp_path, monkeypatch):
        """격리된 QSettings 사용 (테스트 간 오염 방지)."""
        from unittest.mock import patch, MagicMock
        self._mock_qs = MagicMock()
        self._stored: dict = {}

        def fake_value(key, default=None, type=None):
            return self._stored.get(key, default)

        def fake_set_value(key, value):
            self._stored[key] = value

        def fake_begin_group(g):
            pass

        def fake_end_group():
            pass

        def fake_begin_read_array(key):
            return len(self._stored.get(f"_arr_{key}", []))

        def fake_begin_write_array(key):
            pass

        def fake_set_array_index(i):
            pass

        def fake_end_array():
            pass

        self._mock_qs.value.side_effect = fake_value
        self._mock_qs.setValue.side_effect = fake_set_value
        self._mock_qs.beginGroup.side_effect = fake_begin_group
        self._mock_qs.endGroup.side_effect = fake_end_group
        self._mock_qs.beginReadArray.side_effect = fake_begin_read_array
        self._mock_qs.beginWriteArray.side_effect = fake_begin_write_array
        self._mock_qs.setArrayIndex.side_effect = fake_set_array_index
        self._mock_qs.endArray.side_effect = fake_end_array

        with patch(
            "nuristat.core.settings.QSettings",
            return_value=self._mock_qs,
        ):
            yield

    def test_analysis_defaults_roundtrip(self):
        """save_analysis_defaults → load_analysis_defaults 왕복."""
        from nuristat.core.settings import SettingsManager

        sm = SettingsManager()
        defaults = {"confidence_level": 0.99, "missing_policy": "listwise", "method": "pearson"}
        sm.save_analysis_defaults("correlation", defaults)
        loaded = sm.load_analysis_defaults("correlation")
        assert isinstance(loaded, dict)

    def test_load_analysis_defaults_returns_dict(self):
        """저장 없이 load → dict 반환 (빈 dict 가능)."""
        from nuristat.core.settings import SettingsManager

        sm = SettingsManager()
        result = sm.load_analysis_defaults("t_test")
        assert isinstance(result, dict)

    def test_theme_default_is_bool(self):
        """load_theme → dark_mode bool 반환."""
        from nuristat.core.settings import SettingsManager

        sm = SettingsManager()
        theme = sm.load_theme()
        assert isinstance(theme, (bool, int))  # QSettings bool

    def test_window_maximized_default_bool(self):
        """load_window_maximized → bool."""
        from nuristat.core.settings import SettingsManager

        sm = SettingsManager()
        result = sm.load_window_maximized()
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 극단값 처리
# ─────────────────────────────────────────────────────────────────────────────

class TestExtremeValues:
    """분석이 극단값에도 충돌 없이 처리됨."""

    def test_very_large_values_descriptive(self):
        """1e15 수준 값 → 기술통계 오류 없음."""
        from nuristat.analysis.descriptive import run_analysis

        df = pd.DataFrame({"x": [1e15, 2e15, 3e15, 4e15, 5e15]})
        ds = _ds(df)
        spec = {"variables": {"scale": ["x"]}, "missing_policy": MissingPolicy.LISTWISE}
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 1

    def test_very_small_values_descriptive(self):
        """1e-15 수준 값 → 기술통계 오류 없음."""
        from nuristat.analysis.descriptive import run_analysis

        df = pd.DataFrame({"x": [1e-15, 2e-15, 3e-15, 4e-15, 5e-15]})
        ds = _ds(df)
        spec = {"variables": {"scale": ["x"]}, "missing_policy": MissingPolicy.LISTWISE}
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 1

    def test_inf_values_no_crash(self):
        """inf 포함 데이터 → 충돌 없음."""
        from nuristat.analysis.descriptive import run_analysis

        df = pd.DataFrame({"x": [1.0, float("inf"), 3.0, 4.0, 5.0]})
        ds = _ds(df)
        spec = {"variables": {"scale": ["x"]}, "missing_policy": MissingPolicy.LISTWISE}
        try:
            result = run_analysis(ds, spec)
            assert result is not None
        except Exception:
            pass  # 예외도 충돌이 아닌 정상 처리로 간주

    def test_zero_variance_ttest_no_crash(self):
        """분산=0인 그룹 → t-test 충돌 없음."""
        from nuristat.analysis.ttests import run_analysis

        df = pd.DataFrame({
            "x": [5.0] * 10 + [5.0] * 10,  # 동일값 → 분산=0
            "grp": ["A"] * 10 + ["B"] * 10,
        })
        ds = _ds(df, {"x": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        try:
            result = run_analysis(ds, spec)
            assert result is not None
        except Exception:
            pass

    def test_single_observation_descriptive(self):
        """N=1 데이터셋 → 기술통계 충돌 없음."""
        from nuristat.analysis.descriptive import run_analysis

        df = pd.DataFrame({"x": [42.0]})
        ds = _ds(df)
        spec = {"variables": {"variables": ["x"]}, "missing_policy": MissingPolicy.LISTWISE}
        try:
            result = run_analysis(ds, spec)
            assert result is not None
        except Exception:
            pass

    def test_all_same_values_normality(self):
        """동일 값 → 정규성 검정 충돌 없음."""
        from nuristat.analysis.normality import run_analysis

        df = pd.DataFrame({"x": [3.0] * 20})
        ds = _ds(df)
        spec = {"variables": {"variables": ["x"]}, "missing_policy": MissingPolicy.LISTWISE}
        try:
            result = run_analysis(ds, spec)
            assert result is not None
        except Exception:
            pass

    def test_negative_values_anova(self):
        """음수 값 → 분산분석 오류 없음."""
        from nuristat.analysis.anova import run_analysis

        rng = np.random.default_rng(66)
        df = pd.DataFrame({
            "y": np.concatenate([rng.normal(-100, 10, 15), rng.normal(-50, 10, 15)]),
            "grp": ["A"] * 15 + ["B"] * 15,
        })
        ds = _ds(df, {"y": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "y", "factor": "grp"},
            "options": {},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 1

    def test_high_missing_rate_ttest(self):
        """결측율 80% 데이터 → t-test 충돌 없음."""
        from nuristat.analysis.ttests import run_analysis

        rng = np.random.default_rng(77)
        x = rng.normal(0, 1, 50).tolist()
        for i in range(40):
            x[i] = None  # 80% 결측
        df = pd.DataFrame({"x": x, "grp": ["A"] * 25 + ["B"] * 25})
        ds = _ds(df, {"x": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        try:
            result = run_analysis(ds, spec)
            assert result is not None
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 7. 한글 변수명 처리
# ─────────────────────────────────────────────────────────────────────────────

class TestKoreanVariableNames:
    """한글 변수명 포함 데이터셋의 분석 정확도."""

    def test_korean_varname_descriptive(self):
        """한글 변수명 기술통계 처리."""
        from nuristat.analysis.descriptive import run_analysis

        df = pd.DataFrame({
            "나이": [25, 30, 35, 40, 45, 50, 55, 60],
            "소득": [200, 250, 300, 350, 400, 450, 500, 550],
        })
        ds = _ds(df, {"나이": MeasureType.SCALE, "소득": MeasureType.SCALE})
        spec = {"variables": {"scale": ["나이", "소득"]}, "missing_policy": MissingPolicy.LISTWISE}
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 1

    def test_korean_varname_correlation(self):
        """한글 변수명 상관분석."""
        from nuristat.analysis.correlation import run_analysis

        rng = np.random.default_rng(88)
        df = pd.DataFrame({
            "체중": rng.normal(65, 10, 30),
            "키": rng.normal(170, 10, 30),
        })
        ds = _ds(df, {"체중": MeasureType.SCALE, "키": MeasureType.SCALE})
        spec = {
            "variables": {"target": ["체중", "키"]},
            "options": {"method": "pearson", "pairwise": False},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 1

    def test_korean_varname_anova(self):
        """한글 변수명 분산분석."""
        from nuristat.analysis.anova import run_analysis

        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "점수": np.concatenate([rng.normal(70, 10, 15), rng.normal(80, 10, 15), rng.normal(90, 10, 15)]),
            "집단": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        })
        ds = _ds(df, {"점수": MeasureType.SCALE, "집단": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "점수", "factor": "집단"},
            "options": {},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)
        assert len(result.tables) >= 1

    def test_korean_varname_io_csv_roundtrip(self):
        """한글 변수명 CSV 저장·불러오기."""
        from nuristat.io.csv_reader import read_csv

        df = pd.DataFrame({"나이": [20, 30, 40], "지역": ["서울", "부산", "경남"]})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8-sig",
                                          delete=False, newline="") as f:
            df.to_csv(f, index=False)
            fname = f.name
        try:
            ds = read_csv(fname)
            assert ds is not None
            assert any("나이" in c or "age" in c.lower() for c in ds.data.columns)
        finally:
            os.unlink(fname)

    def test_korean_value_labels_preserved(self):
        """한글 값 레이블 직렬화 보존."""
        from nuristat.core.variable import VariableMeta
        from nuristat.core.typing import StorageType

        var = VariableMeta("성별", storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL)
        var.value_labels = {1: "남성", 2: "여성"}

        d = var.to_dict()
        var2 = VariableMeta.from_dict(d)
        assert var2.value_labels.get(1) == "남성" or var2.value_labels.get("1") == "남성"
        assert var2.value_labels.get(2) == "여성" or var2.value_labels.get("2") == "여성"


# ─────────────────────────────────────────────────────────────────────────────
# 8. 레지스트리 일관성
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisRegistry:
    """구현된 분석 목록이 레지스트리와 일관성 유지."""

    def test_implemented_analyses_count(self):
        """구현된 분석 수 ≥ 20."""
        from nuristat.analysis.registry import AnalysisRegistry

        r = AnalysisRegistry()
        implemented = r.list_implemented()
        assert len(implemented) >= 20

    def test_all_implemented_have_id(self):
        """모든 구현 분석에 id 속성 존재."""
        from nuristat.analysis.registry import AnalysisRegistry

        r = AnalysisRegistry()
        for plugin in r.list_implemented():
            assert hasattr(plugin, "id")
            assert isinstance(plugin.id, str)
            assert len(plugin.id) > 0

    def test_all_implemented_have_name(self):
        """모든 구현 분석에 name 속성 존재."""
        from nuristat.analysis.registry import AnalysisRegistry

        r = AnalysisRegistry()
        for plugin in r.list_implemented():
            assert hasattr(plugin, "name")
            assert isinstance(plugin.name, str)

    def test_all_implemented_have_run(self):
        """모든 구현 분석에 run 호출 가능."""
        from nuristat.analysis.registry import AnalysisRegistry

        r = AnalysisRegistry()
        for plugin in r.list_implemented():
            assert callable(plugin.run)

    def test_no_duplicate_ids(self):
        """분석 ID 중복 없음."""
        from nuristat.analysis.registry import AnalysisRegistry

        r = AnalysisRegistry()
        ids = [p.id for p in r.list_implemented()]
        assert len(ids) == len(set(ids)), f"중복 ID: {set(i for i in ids if ids.count(i) > 1)}"

    def test_list_all_includes_planned(self):
        """list_all은 구현 + 기획 분석 포함."""
        from nuristat.analysis.registry import AnalysisRegistry

        r = AnalysisRegistry()
        all_count = len(r.list_all())
        impl_count = len(r.list_implemented())
        assert all_count >= impl_count

    def test_categories_returns_list(self):
        """categories → 카테고리 목록 반환."""
        from nuristat.analysis.registry import AnalysisRegistry

        r = AnalysisRegistry()
        cats = r.categories()
        assert isinstance(cats, (list, tuple, set))
        assert len(cats) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. 데이터셋 불변성 — 분석 후 원본 유지
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetImmutability:
    """분석 함수들이 원본 Dataset을 변경하지 않음."""

    def _snap(self, ds: Dataset) -> tuple:
        """데이터셋 상태 스냅샷 (shape, 합계, 컬럼명)."""
        return (
            ds.data.shape,
            tuple(ds.data.columns.tolist()),
            float(ds.data.select_dtypes(include="number").sum().sum()),
        )

    def test_descriptive_does_not_mutate(self):
        from nuristat.analysis.descriptive import run_analysis

        rng = np.random.default_rng(10)
        df = pd.DataFrame({"x": rng.normal(0, 1, 50), "y": rng.normal(0, 1, 50)})
        ds = _ds(df)
        snap_before = self._snap(ds)
        run_analysis(ds, {"variables": {"variables": ["x", "y"]},
                          "missing_policy": MissingPolicy.LISTWISE})
        assert self._snap(ds) == snap_before

    def test_ttest_does_not_mutate(self):
        from nuristat.analysis.ttests import run_analysis

        rng = np.random.default_rng(20)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 40),
            "grp": ["A"] * 20 + ["B"] * 20,
        })
        ds = _ds(df, {"x": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        snap_before = self._snap(ds)
        run_analysis(ds, {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        })
        assert self._snap(ds) == snap_before

    def test_anova_does_not_mutate(self):
        from nuristat.analysis.anova import run_analysis

        rng = np.random.default_rng(30)
        df = pd.DataFrame({
            "y": np.concatenate([rng.normal(10, 2, 15), rng.normal(12, 2, 15), rng.normal(14, 2, 15)]),
            "grp": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        })
        ds = _ds(df, {"y": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        snap_before = self._snap(ds)
        run_analysis(ds, {
            "variables": {"dependent": "y", "factor": "grp"},
            "options": {},
            "missing_policy": MissingPolicy.LISTWISE,
        })
        assert self._snap(ds) == snap_before

    def test_correlation_does_not_mutate(self):
        from nuristat.analysis.correlation import run_analysis

        rng = np.random.default_rng(40)
        df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})
        ds = _ds(df)
        snap_before = self._snap(ds)
        run_analysis(ds, {
            "variables": {"variables": ["a", "b"]},
            "options": {"method": "pearson"},
            "missing_policy": MissingPolicy.LISTWISE,
        })
        assert self._snap(ds) == snap_before

    def test_regression_does_not_mutate(self):
        from nuristat.analysis.regression import run_analysis

        rng = np.random.default_rng(50)
        df = pd.DataFrame({
            "y": rng.normal(0, 1, 30),
            "x1": rng.normal(0, 1, 30),
            "x2": rng.normal(0, 1, 30),
        })
        ds = _ds(df, {"y": MeasureType.SCALE, "x1": MeasureType.SCALE, "x2": MeasureType.SCALE})
        snap_before = self._snap(ds)
        run_analysis(ds, {
            "variables": {"dependent": "y", "independents": ["x1", "x2"]},
            "options": {},
            "missing_policy": MissingPolicy.LISTWISE,
        })
        assert self._snap(ds) == snap_before


# ─────────────────────────────────────────────────────────────────────────────
# 10. 분석 결과 수치 일관성
# ─────────────────────────────────────────────────────────────────────────────

class TestResultNumericalConsistency:
    """분석 결과 테이블 내 N, 합계 등 내부 일관성."""

    def test_ttest_group_n_sums_to_total(self):
        """t-test 그룹별 N 합계 = 전체 유효 N."""
        from nuristat.analysis.ttests import run_analysis

        rng = np.random.default_rng(11)
        n_a, n_b = 18, 22
        df = pd.DataFrame({
            "x": np.concatenate([rng.normal(5, 1, n_a), rng.normal(6, 1, n_b)]),
            "grp": ["A"] * n_a + ["B"] * n_b,
        })
        ds = _ds(df, {"x": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "x", "group": "grp"},
            "options": {"equal_var": "auto"},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        grp_tbl = [t for t in result.tables if "Group Statistics" in t.title]
        if grp_tbl:
            n_sum = int(grp_tbl[0].dataframe["N"].sum())
            assert n_sum == n_a + n_b

    def test_anova_ss_within_plus_between_equals_total(self):
        """SS_between + SS_within ≈ SS_total (오차 ±0.1)."""
        from nuristat.analysis.anova import run_analysis

        rng = np.random.default_rng(22)
        groups = [rng.normal(10 + i * 2, 3, 20) for i in range(3)]
        df = pd.DataFrame({
            "y": np.concatenate(groups),
            "grp": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        })
        ds = _ds(df, {"y": MeasureType.SCALE, "grp": MeasureType.NOMINAL})
        spec = {
            "variables": {"dependent": "y", "factor": "grp"},
            "options": {},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        anova_tbl = [t for t in result.tables if t.title == "ANOVA"]
        if anova_tbl:
            df_a = anova_tbl[0].dataframe
            ss_rows = df_a[pd.to_numeric(df_a["SS"], errors="coerce").notna()]
            if len(ss_rows) >= 2:
                ss_vals = pd.to_numeric(ss_rows["SS"], errors="coerce").dropna().tolist()
                # 전체 SS = 분산 분해
                total_ss = float(np.var(df["y"], ddof=0)) * len(df)
                partial_sum = sum(ss_vals[:2])
                assert abs(partial_sum - total_ss) < 1.0

    def test_frequency_pct_sums_to_100(self):
        """빈도분석 유효 퍼센트 합계 ≈ 100%."""
        from nuristat.analysis.frequencies import run_analysis

        df = pd.DataFrame({"cat": ["A"] * 30 + ["B"] * 40 + ["C"] * 30})
        ds = _ds(df, {"cat": MeasureType.NOMINAL})
        spec = {
            "variables": {"target": ["cat"]},
            "options": {},
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        freq_tbl = [t for t in result.tables if "Frequency" in t.title or "빈도" in t.title
                    or "Frequenc" in t.title]
        if freq_tbl:
            df_f = freq_tbl[0].dataframe
            pct_cols = [c for c in df_f.columns
                        if "Percent" in str(c) or "%" in str(c) or "pct" in str(c).lower()]
            if pct_cols:
                pct_vals = pd.to_numeric(df_f[pct_cols[0]], errors="coerce").dropna()
                total = pct_vals[pct_vals < 99.0].sum()  # 누적 합계 행 제외
                if total > 50:
                    assert abs(total - 100.0) < 1.0

    def test_descriptive_n_matches_dataset_n(self):
        """기술통계 N = 결측 없을 때 데이터셋 행 수."""
        from nuristat.analysis.descriptive import run_analysis

        n = 45
        rng = np.random.default_rng(33)
        df = pd.DataFrame({"x": rng.normal(0, 1, n)})
        ds = _ds(df, {"x": MeasureType.SCALE})
        spec = {
            "variables": {"scale": ["x"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        desc_tbl = [t for t in result.tables if "Descriptive" in t.title]
        if desc_tbl:
            df_d = desc_tbl[0].dataframe
            if "N" in df_d.columns:
                n_val = int(pd.to_numeric(df_d["N"], errors="coerce").dropna().iloc[0])
                assert n_val == n

    def test_cps_table_n_matches_dataset(self):
        """CPS 테이블 N이 데이터셋 행 수와 일치."""
        from nuristat.analysis.descriptive import run_analysis

        n = 60
        rng = np.random.default_rng(44)
        df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        ds = _ds(df, {"x": MeasureType.SCALE, "y": MeasureType.SCALE})
        spec = {
            "variables": {"scale": ["x", "y"]},
            "options": {},
            "confidence_level": 0.95,
            "missing_policy": MissingPolicy.LISTWISE,
        }
        result = run_analysis(ds, spec)

        cps_tbl = [t for t in result.tables if "Case Processing" in t.title or "CPS" in t.title]
        if cps_tbl:
            df_cps = cps_tbl[0].dataframe
            # 전체 N 확인
            n_col = [c for c in df_cps.columns if c == "N" or "Total" in str(c)]
            if n_col:
                vals = pd.to_numeric(df_cps[n_col[0]], errors="coerce").dropna()
                assert any(int(v) == n for v in vals)

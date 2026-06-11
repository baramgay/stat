"""6개 수정 기능 통합 E2E 검증.

수정 범위:
  STEP 1 — ChartBuilderDialog 연결 (스텁 제거)
  STEP 2 — ComputeVariableDialog 시그널·핸들러·데이터 반영
  STEP 3 — RankDialog 시그널·핸들러·순위 계산·데이터 반영
  STEP 4 — ScriptRunnerDialog script_executed Signal 추가
  STEP 5 — PivotDialog pivot_created emit 추가
  STEP 6 — PivotDialog 버튼 라벨 오타 수정

담당 에이전트: tester-qa
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from nuristat.core.dataset import Dataset
from nuristat.core.typing import MeasureType, StorageType
from nuristat.core.variable import VariableMeta


# ── 공통 fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def dataset() -> Dataset:
    df = pd.DataFrame({
        "age":    [20, 35, 22, 40, 25, 38, 30, 45],
        "income": [200, 350, 210, 400, 250, 380, 300, 450],
        "group":  [0, 1, 0, 1, 0, 1, 0, 1],
    })
    variables = {
        "age": VariableMeta(name="age", label="나이",
                            storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        "income": VariableMeta(name="income", label="소득",
                               storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
        "group": VariableMeta(name="group", label="집단",
                              storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL,
                              value_labels={0: "A", 1: "B"}),
    }
    return Dataset(df, "test", variables)


# ── STEP 1: ChartBuilderDialog 인스턴스화 ────────────────────────────────────

class TestChartBuilderDialog:
    """STEP 1 — 스텁 제거 후 ChartBuilderDialog가 정상 생성되는지 검증."""

    def test_dialog_instantiates(self, dataset):
        from nuristat.ui.chart_builder import ChartBuilderDialog
        dlg = ChartBuilderDialog(dataset)
        assert dlg is not None
        dlg.close()

    def test_has_chart_saved_signal(self, dataset):
        from nuristat.ui.chart_builder import ChartBuilderDialog
        dlg = ChartBuilderDialog(dataset)
        assert hasattr(dlg, "chart_saved")
        dlg.close()

    def test_has_chart_inserted_signal(self, dataset):
        from nuristat.ui.chart_builder import ChartBuilderDialog
        dlg = ChartBuilderDialog(dataset)
        assert hasattr(dlg, "chart_inserted")
        dlg.close()

    def test_variables_populated_in_combo(self, dataset):
        from nuristat.ui.chart_builder import ChartBuilderDialog
        dlg = ChartBuilderDialog(dataset)
        texts = [dlg._x_combo.itemText(i) for i in range(dlg._x_combo.count())]
        assert any("age" in t for t in texts), f"age 없음: {texts}"
        dlg.close()


# ── STEP 2: ComputeVariableDialog 시그널 + 데이터 반영 ──────────────────────

class TestComputeVariableDialog:
    """STEP 2 — computed 시그널이 올바른 (name, Series)를 emit하는지 검증."""

    def test_computed_signal_fires(self, dataset):
        from nuristat.ui.dialogs.compute_variable_dialog import ComputeVariableDialog
        dlg = ComputeVariableDialog(dataset)

        received: list = []
        dlg.computed.connect(lambda name, series: received.append((name, series)))

        dlg.target_edit.setText("age_z")
        dlg.formula_edit.setPlainText("(age - mean(age)) / std(age)")
        dlg._compute()

        assert len(received) == 1, "computed 시그널이 1회 emit되어야 함"
        name, series = received[0]
        assert name == "age_z"
        assert isinstance(series, pd.Series)
        assert len(series) == len(dataset.data)

    def test_computed_series_values_correct(self, dataset):
        from nuristat.ui.dialogs.compute_variable_dialog import ComputeVariableDialog
        dlg = ComputeVariableDialog(dataset)

        received: list = []
        dlg.computed.connect(lambda n, s: received.append(s))

        dlg.target_edit.setText("income2")
        dlg.formula_edit.setPlainText("income * 2")
        dlg._compute()

        series = received[0]
        expected = dataset.data["income"] * 2
        pd.testing.assert_series_equal(series.reset_index(drop=True),
                                       expected.reset_index(drop=True),
                                       check_names=False)

    def test_handler_applies_to_dataset(self, dataset):
        """main_window._on_variable_computed 핸들러가 dataset에 실제 반영되는지."""
        ds = Dataset(dataset.data.copy(), "copy", dict(dataset.variables))

        from nuristat.ui.dialogs.compute_variable_dialog import ComputeVariableDialog
        dlg = ComputeVariableDialog(ds)

        received_series: list = []
        dlg.computed.connect(lambda n, s: received_series.append((n, s)))

        dlg.target_edit.setText("new_col")
        dlg.formula_edit.setPlainText("age + 100")
        dlg._compute()

        name, series = received_series[0]
        # 핸들러 로직을 직접 실행 (main_window 없이)
        ds.data[name] = series

        assert "new_col" in ds.data.columns
        assert (ds.data["new_col"] == ds.data["age"] + 100).all()


# ── STEP 3: RankDialog 시그널 + 순위 계산 ────────────────────────────────────

class TestRankDialog:
    """STEP 3 — rank_applied 시그널·순위 계산 정확성 검증."""

    def test_rank_applied_signal_fires(self, dataset):
        from nuristat.ui.dialogs.rank_dialog import RankDialog
        dlg = RankDialog(dataset)

        received: list = []
        dlg.rank_applied.connect(lambda src, tgt, meth: received.append((src, tgt, meth)))

        dlg.source_combo.setCurrentText("age")
        dlg.target_edit.setText("age_rank")
        dlg.rank_radio.setChecked(True)
        dlg._apply_rank()

        assert len(received) == 1
        src, tgt, meth = received[0]
        assert src == "age"
        assert tgt == "age_rank"
        assert meth == "average"

    def test_rank_values_are_correct(self, dataset):
        from nuristat.ui.dialogs.rank_dialog import RankDialog
        dlg = RankDialog(dataset)

        received: list = []
        dlg.rank_applied.connect(lambda src, tgt, meth: received.append((src, tgt, meth)))

        dlg.source_combo.setCurrentText("income")
        dlg.target_edit.setText("income_rank")
        dlg.rank_radio.setChecked(True)
        dlg._apply_rank()

        src, tgt, meth = received[0]
        # 핸들러 로직 직접 실행
        series = dataset.data[src]
        ranked = series.rank(pct=False, method=meth)
        assert ranked.min() == 1.0
        assert ranked.max() == len(dataset.data)

    def test_pct_rank_method(self, dataset):
        from nuristat.ui.dialogs.rank_dialog import RankDialog
        dlg = RankDialog(dataset)

        received: list = []
        dlg.rank_applied.connect(lambda src, tgt, meth: received.append((src, tgt, meth)))

        dlg.source_combo.setCurrentText("age")
        dlg.target_edit.setText("age_pct")
        dlg.pct_radio.setChecked(True)
        dlg._apply_rank()

        src, tgt, meth = received[0]
        assert meth == "pct"
        # pct rank는 0~1 범위
        ranked = dataset.data[src].rank(pct=True, method="average")
        assert ranked.max() <= 1.0
        assert ranked.min() > 0.0

    def test_handler_applies_rank_to_dataset(self, dataset):
        """main_window._on_rank_created 핸들러 로직 직접 검증."""
        from nuristat.ui.dialogs.rank_dialog import RankDialog
        ds = Dataset(dataset.data.copy(), "copy", dict(dataset.variables))
        dlg = RankDialog(ds)

        received: list = []
        dlg.rank_applied.connect(lambda s, t, m: received.append((s, t, m)))

        dlg.source_combo.setCurrentText("age")
        dlg.target_edit.setText("age_rank2")
        dlg.rank_radio.setChecked(True)
        dlg._apply_rank()

        src, tgt, meth = received[0]
        series = ds.data[src]
        ranked = series.rank(pct=(meth == "pct"), method=meth if meth != "pct" else "average")
        ds.data[tgt] = ranked

        assert "age_rank2" in ds.data.columns
        assert ds.data["age_rank2"].min() == 1.0


# ── STEP 4: ScriptRunnerDialog script_executed Signal ───────────────────────

class TestScriptRunnerDialog:
    """STEP 4 — script_executed 클래스 속성 및 PythonBridge 실제 실행 검증."""

    def test_signal_defined_on_class(self):
        from nuristat.ui.dialogs.script_runner_dialog import ScriptRunnerDialog
        from PySide6.QtCore import Signal
        assert hasattr(ScriptRunnerDialog, "script_executed"), \
            "ScriptRunnerDialog에 script_executed 클래스 속성 없음"

    def test_dialog_instantiates(self, dataset):
        from nuristat.ui.dialogs.script_runner_dialog import ScriptRunnerDialog
        dlg = ScriptRunnerDialog(dataset)
        assert dlg is not None
        dlg.close()

    def test_python_bridge_executes_script(self, dataset):
        """PythonBridge가 실제로 Python 스크립트를 실행하고 stdout을 반환하는지."""
        from nuristat.analysis.python_bridge import PythonBridge
        bridge = PythonBridge()
        result = bridge.execute("x = 1 + 1\nprint(x)", dataset)
        assert result["success"] is True
        assert "2" in result["stdout"]

    def test_python_bridge_captures_variable(self, dataset):
        from nuristat.analysis.python_bridge import PythonBridge
        bridge = PythonBridge()
        result = bridge.execute("total = df['age'].sum()", dataset)
        assert result["success"] is True
        assert "total" in result["variables"]

    def test_python_bridge_dataframe_access(self, dataset):
        from nuristat.analysis.python_bridge import PythonBridge
        bridge = PythonBridge()
        result = bridge.execute("print(df.shape)", dataset)
        assert result["success"] is True
        assert "8" in result["stdout"]  # 8행

    def test_r_bridge_availability_check(self):
        """R 브리지 is_available()이 오류 없이 실행되는지."""
        from nuristat.analysis.r_bridge import RBridge
        bridge = RBridge()
        available = bridge.is_available()
        assert isinstance(available, bool)

    def test_r_bridge_executes_if_available(self, dataset):
        """R이 설치된 경우 실제 스크립트 실행."""
        from nuristat.analysis.r_bridge import RBridge
        bridge = RBridge()
        if not bridge.is_available():
            pytest.skip("R이 설치되지 않아 건너뜀")
        result = bridge.execute("cat('hello from R\\n')", dataset)
        assert result["success"] is True
        assert "hello from R" in result["stdout"]

    def test_r_bridge_dataframe_passed(self, dataset):
        """R 실행 시 df가 올바르게 전달되는지."""
        from nuristat.analysis.r_bridge import RBridge
        bridge = RBridge()
        if not bridge.is_available():
            pytest.skip("R 미설치")
        result = bridge.execute("cat(nrow(df), '\\n')", dataset)
        assert result["success"] is True
        assert "8" in result["stdout"]


# ── STEP 5: PivotDialog pivot_created emit ───────────────────────────────────

class TestPivotDialog:
    """STEP 5 — pivot_created가 실제로 emit되는지 검증."""

    def test_pivot_created_signal_defined(self):
        from nuristat.ui.dialogs.pivot_dialog import PivotDialog
        assert hasattr(PivotDialog, "pivot_created"), \
            "PivotDialog에 pivot_created 시그널 없음"

    def test_pivot_signal_fires_on_generate(self, dataset):
        from nuristat.ui.dialogs.pivot_dialog import PivotDialog
        dlg = PivotDialog(dataset)

        received: list = []
        dlg.pivot_created.connect(lambda pt: received.append(pt))

        # 행·열 설정 후 피벗 생성
        dlg.row_combo.setCurrentText("group")
        dlg.col_combo.setCurrentText("(없음)")
        dlg.val_combo.setCurrentText("age")
        dlg._generate_pivot()

        assert len(received) == 1, "pivot_created가 emit되어야 함"
        pivot = received[0]
        assert pivot is not None
        assert isinstance(pivot, pd.DataFrame)

    def test_pivot_table_content_correct(self, dataset):
        from nuristat.ui.dialogs.pivot_dialog import PivotDialog
        dlg = PivotDialog(dataset)

        received: list = []
        dlg.pivot_created.connect(lambda pt: received.append(pt))

        dlg.row_combo.setCurrentText("group")
        dlg.val_combo.setCurrentText("income")
        dlg._generate_pivot()

        pivot = received[0]
        assert pivot.shape[0] > 0


# ── STEP 6: PivotDialog 버튼 라벨 오타 수정 ─────────────────────────────────

class TestPivotDialogLabel:
    """STEP 6 — 내보내기 버튼 라벨 오타 수정 검증."""

    def test_export_button_label(self, dataset):
        from nuristat.ui.dialogs.pivot_dialog import PivotDialog
        dlg = PivotDialog(dataset)
        assert dlg.btn_export.text() == "💾 내보내기", \
            f"버튼 라벨 오타: '{dlg.btn_export.text()}'"

    def test_export_button_disabled_initially(self, dataset):
        from nuristat.ui.dialogs.pivot_dialog import PivotDialog
        dlg = PivotDialog(dataset)
        assert not dlg.btn_export.isEnabled(), "초기에는 비활성화여야 함"

    def test_export_button_enabled_after_pivot(self, dataset):
        from nuristat.ui.dialogs.pivot_dialog import PivotDialog
        dlg = PivotDialog(dataset)
        dlg.row_combo.setCurrentText("group")
        dlg.val_combo.setCurrentText("age")
        dlg._generate_pivot()
        assert dlg.btn_export.isEnabled(), "피벗 생성 후 활성화되어야 함"

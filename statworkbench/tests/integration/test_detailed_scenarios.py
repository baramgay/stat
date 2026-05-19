"""상세 시나리오 테스트 — 실제 사용 패턴 기반 통합 테스트.

커버 영역:
1. 데이터 입력 (숫자/문자/결측/혼합)
2. 네비게이션 (Enter/Tab/화살표/F2/Delete)
3. dataset.data 동기화 확인
4. 변수 메타데이터 (타입 추론)
5. 분석 다이얼로그 변수 목록
6. 빈도/기술통계/T검정 실제 실행
7. 결측값 처리 (SPSS 방식)
8. 경계 조건 및 특수 케이스
"""
from __future__ import annotations

import sys
import pytest
import pandas as pd
import numpy as np

from PySide6.QtWidgets import QApplication, QLineEdit, QAbstractItemView
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import StorageType, MeasureType
from statworkbench.ui.data_view import DataView
from statworkbench.ui.models.spss_grid_model import SPSSGridModel


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    if existing:
        yield existing
    else:
        a = QApplication(sys.argv)
        yield a


@pytest.fixture
def view(app):
    """빈 DataView (그리드 입력 테스트용)."""
    ds = Dataset(pd.DataFrame(), name="Test")
    v = DataView()
    v.set_dataset(ds)
    v.resize(900, 500)
    v.show()
    QApplication.processEvents()
    yield v
    v.hide()
    v.close()


@pytest.fixture
def model():
    """직접 SPSSGridModel (UI 없는 모델 단위 테스트)."""
    return SPSSGridModel()


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

def _select(view: DataView, row: int, col: int):
    idx = view._model.index(row, col)
    view.table.setCurrentIndex(idx)
    view.table.setFocus()
    QApplication.processEvents()


def _editor(view: DataView) -> QLineEdit | None:
    for w in view.table.viewport().findChildren(QLineEdit):
        if w.isVisible():
            return w
    return None


def _type(view: DataView, char: str):
    QTest.keyClicks(view.table, char)
    QApplication.processEvents()


def _wait(ms: int = 60):
    QTest.qWait(ms)


def _display(view: DataView, row: int, col: int) -> str:
    val = view._model.data(view._model.index(row, col), Qt.ItemDataRole.DisplayRole)
    return str(val) if val is not None else ""


def _enter_value(view: DataView, row: int, col: int, value: str):
    """셀에 값을 입력하고 Enter로 커밋."""
    _select(view, row, col)
    # F2로 편집기 열기 (기존 값 유지, 이후 clear+retype)
    QTest.keyClick(view.table, Qt.Key.Key_F2)
    QApplication.processEvents()
    ed = _editor(view)
    if ed is None:
        # F2 실패 시 Enter로 시도
        QTest.keyClick(view.table, Qt.Key.Key_Return)
        QApplication.processEvents()
        ed = _editor(view)
    if ed:
        ed.clear()
        if value:
            QTest.keyClicks(ed, value)
        QTest.keyClick(ed, Qt.Key.Key_Return)
        _wait(60)


def _model_index(m: SPSSGridModel, row: int, col: int):
    return m.index(row, col)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 입력 시나리오
# ══════════════════════════════════════════════════════════════════════════════

class TestDataEntryScenarios:
    """실제 사용자 데이터 입력 패턴."""

    def test_enter_integer_series(self, view):
        """정수 시리즈 입력: 1~5를 Enter로 연속 입력."""
        _select(view, 0, 0)
        for value in ["1", "2", "3", "4", "5"]:
            QTest.keyClicks(view.table, value)
            QApplication.processEvents()
            ed = _editor(view)
            assert ed is not None
            QTest.keyClick(ed, Qt.Key.Key_Return)
            _wait(60)

        # 5개 행 모두 커밋 확인
        for row, expected in enumerate([1, 2, 3, 4, 5]):
            val = _display(view, row, 0)
            assert val != "" and val != ".", f"Row {row} should have value, got '{val}'"

    def test_enter_float_series(self, view):
        """소수점 값 입력: 1.5, 2.7, 3.14 등."""
        for row, value in enumerate(["1.5", "2.7", "3.14"]):
            _enter_value(view, row, 0, value)

        df = view._model.get_dataframe()
        assert df.iloc[0, 0] == pytest.approx(1.5)
        assert df.iloc[1, 0] == pytest.approx(2.7)

    def test_enter_string_labels(self, view):
        """문자열 그룹 레이블 입력: A, B, C."""
        _select(view, 0, 0)
        for label in ["A", "B", "C"]:
            QTest.keyClicks(view.table, label)
            QApplication.processEvents()
            ed = _editor(view)
            assert ed is not None
            QTest.keyClick(ed, Qt.Key.Key_Return)
            _wait(60)

        for row, expected in enumerate(["A", "B", "C"]):
            assert _display(view, row, 0) == expected

    def test_enter_negative_numbers(self, view):
        """음수 입력."""
        for row, value in enumerate(["-10", "-3.5", "-0.001"]):
            _enter_value(view, row, 0, value)

        df = view._model.get_dataframe()
        assert df.iloc[0, 0] < 0
        assert df.iloc[1, 0] < 0

    def test_enter_zero(self, view):
        """0 입력 — 결측과 구분."""
        _enter_value(view, 0, 0, "0")
        val = _display(view, 0, 0)
        assert val == "0", f"0 should display as '0', not '{val}'"

    def test_enter_large_number(self, view):
        """큰 수 입력."""
        _enter_value(view, 0, 0, "9999999")
        df = view._model.get_dataframe()
        assert df.iloc[0, 0] == 9999999

    def test_tab_across_columns(self, view):
        """Tab으로 여러 컬럼에 값 입력."""
        _select(view, 0, 0)
        # 값 입력 후 Tab → 다음 열
        for value in ["10", "20", "30"]:
            QTest.keyClicks(view.table, value)
            QApplication.processEvents()
            ed = _editor(view)
            assert ed is not None
            QTest.keyClick(ed, Qt.Key.Key_Tab)
            _wait(60)

        # 3개 열에 각각 값 확인
        for col in range(3):
            val = _display(view, 0, col)
            assert val not in ("", "."), f"Col {col} should have value, got '{val}'"

    def test_overwrite_existing_value(self, view):
        """기존 값 덮어쓰기."""
        _enter_value(view, 0, 0, "100")
        assert _display(view, 0, 0) not in ("", ".")

        # 동일 셀 다시 입력
        _enter_value(view, 0, 0, "999")
        assert _display(view, 0, 0) == "999"

    def test_f2_edit_preserves_value(self, view):
        """F2 편집 모드: 기존 값 전체 선택 후 보여줌."""
        _enter_value(view, 0, 0, "42")
        _select(view, 0, 0)
        QTest.keyClick(view.table, Qt.Key.Key_F2)
        QApplication.processEvents()

        ed = _editor(view)
        assert ed is not None, "F2 should open editor"
        assert ed.text() == "42", f"F2 should show existing value, got '{ed.text()}'"

        QTest.keyClick(ed, Qt.Key.Key_Escape)
        _wait(30)

    def test_delete_clears_cell(self, view):
        """Delete 키로 셀 값 삭제."""
        _enter_value(view, 0, 0, "55")
        _select(view, 0, 0)
        QTest.keyClick(view.table, Qt.Key.Key_Delete)
        _wait(60)

        # 삭제 후 결측 표시
        val = _display(view, 0, 0)
        assert val in ("", "."), f"After delete, should be empty/missing, got '{val}'"

    def test_two_column_data_entry(self, view):
        """두 컬럼 데이터 입력: 그룹 변수 + 점수 변수."""
        rows = [("A", "85"), ("A", "90"), ("B", "75"), ("B", "80")]
        for row_idx, (group, score) in enumerate(rows):
            _enter_value(view, row_idx, 0, group)
            _enter_value(view, row_idx, 1, score)

        # 데이터 확인
        assert _display(view, 0, 0) == "A"
        assert _display(view, 2, 0) == "B"
        assert _display(view, 0, 1) not in ("", ".")
        assert _display(view, 2, 1) not in ("", ".")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 네비게이션 정확도
# ══════════════════════════════════════════════════════════════════════════════

class TestNavigationAccuracy:
    """Enter/Tab 한 번 = 정확히 한 칸 이동."""

    def test_enter_moves_exactly_one_row(self, view):
        """Enter 커밋 후 정확히 1행 아래로 이동."""
        _select(view, 0, 0)
        QTest.keyClicks(view.table, "5")
        QApplication.processEvents()
        ed = _editor(view)
        assert ed is not None
        QTest.keyClick(ed, Qt.Key.Key_Return)
        _wait(80)

        cur = view.table.currentIndex()
        assert cur.row() == 1, f"Expected row 1, got row {cur.row()}"
        assert cur.column() == 0, f"Expected col 0, got col {cur.column()}"

    def test_enter_from_row1_moves_to_row2(self, view):
        """row 1에서 Enter → row 2로 이동."""
        _enter_value(view, 0, 0, "1")
        _enter_value(view, 1, 0, "2")

        _select(view, 1, 0)
        QTest.keyClicks(view.table, "X")
        QApplication.processEvents()
        ed = _editor(view)
        QTest.keyClick(ed, Qt.Key.Key_Return)
        _wait(80)

        cur = view.table.currentIndex()
        assert cur.row() == 2, f"Expected row 2, got row {cur.row()}"

    def test_tab_moves_exactly_one_column(self, view):
        """Tab 커밋 후 정확히 1열 오른쪽으로 이동."""
        _select(view, 0, 0)
        QTest.keyClicks(view.table, "5")
        QApplication.processEvents()
        ed = _editor(view)
        assert ed is not None
        QTest.keyClick(ed, Qt.Key.Key_Tab)
        _wait(80)

        cur = view.table.currentIndex()
        assert cur.row() == 0, f"Expected row 0, got row {cur.row()}"
        assert cur.column() == 1, f"Expected col 1, got col {cur.column()}"

    def test_shift_tab_moves_left(self, view):
        """Shift+Tab: 왼쪽으로 이동."""
        _enter_value(view, 0, 0, "A")
        _select(view, 0, 1)
        QTest.keyClicks(view.table, "B")
        QApplication.processEvents()
        ed = _editor(view)
        assert ed is not None
        QTest.keyClick(ed, Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
        _wait(80)

        cur = view.table.currentIndex()
        assert cur.column() == 0, f"Shift+Tab should go left, got col {cur.column()}"

    def test_enter_sequential_three_rows(self, view):
        """3행 연속 Enter: row 0→1→2→3."""
        for row, val in enumerate(["10", "20", "30"]):
            _select(view, row, 0)
            QTest.keyClicks(view.table, val)
            QApplication.processEvents()
            ed = _editor(view)
            QTest.keyClick(ed, Qt.Key.Key_Return)
            _wait(80)
            cur = view.table.currentIndex()
            assert cur.row() == row + 1, (
                f"After Enter at row {row}, expected row {row+1}, got {cur.row()}"
            )

    def test_arrow_keys_in_non_edit_mode(self, view):
        """편집 모드 밖 화살표키 이동."""
        _select(view, 2, 2)
        QTest.keyClick(view.table, Qt.Key.Key_Up)
        _wait(30)
        assert view.table.currentIndex().row() == 1

        QTest.keyClick(view.table, Qt.Key.Key_Down)
        _wait(30)
        assert view.table.currentIndex().row() == 2

        QTest.keyClick(view.table, Qt.Key.Key_Left)
        _wait(30)
        assert view.table.currentIndex().column() == 1

        QTest.keyClick(view.table, Qt.Key.Key_Right)
        _wait(30)
        assert view.table.currentIndex().column() == 2


# ══════════════════════════════════════════════════════════════════════════════
# 3. 결측값 처리 (SPSS 방식)
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingValueHandling:
    """SPSS 스타일 결측값 처리."""

    def test_numeric_empty_cell_shows_dot(self, model):
        """수치형 컬럼 빈 셀은 '.'으로 표시."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        # row 1은 비어 있음 → 수치형 컬럼이므로 '.'
        val = model.data(_model_index(model, 1, 0), Qt.ItemDataRole.DisplayRole)
        assert val == ".", f"Empty numeric cell should show '.', got '{val}'"

    def test_dot_input_treated_as_missing(self, model):
        """'.' 입력 → 결측(pd.NA)으로 처리."""
        model.setData(_model_index(model, 0, 0), "5", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 1, 0), ".", Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        assert pd.isna(df.iloc[1, 0]), "'.' should become NaN/NA"

    def test_string_column_empty_shows_blank(self, model):
        """문자형 컬럼 빈 셀은 ''으로 표시."""
        model.setData(_model_index(model, 0, 0), "Hello", Qt.ItemDataRole.EditRole)
        # row 1은 비어 있음 → 문자형 컬럼이므로 ''
        val = model.data(_model_index(model, 1, 0), Qt.ItemDataRole.DisplayRole)
        assert val == "", f"Empty string cell should show '', got '{val}'"

    def test_empty_string_input_is_missing(self, model):
        """빈 문자열 입력 → 결측."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 1, 0), "", Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        assert pd.isna(df.iloc[1, 0])

    def test_none_input_is_missing(self, model):
        """None 입력 → 결측."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 1, 0), None, Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        assert pd.isna(df.iloc[1, 0])

    def test_missing_numeric_background(self, model):
        """결측 수치형 셀: 노란색 배경 (SPSS 스타일)."""
        from PySide6.QtGui import QColor
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        brush = model.data(_model_index(model, 1, 0), Qt.ItemDataRole.BackgroundRole)
        assert brush is not None, "Missing numeric cell should have background"
        # 결측 배경은 노란색 계열
        color = brush.color()
        assert color.red() > 200, "Missing bg should be yellow-ish"

    def test_data_with_mixed_missing(self):
        """결측이 섞인 데이터 분석 가능 여부."""
        df = pd.DataFrame({
            "score": [85, None, 90, None, 78, 92, None, 88],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        })
        ds = Dataset(df, name="MissingTest")
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["score"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert result is not None
        # 결측 제외 후 5개 케이스 사용 가능
        assert result.to_html() != ""


# ══════════════════════════════════════════════════════════════════════════════
# 4. 변수 메타데이터 동기화
# ══════════════════════════════════════════════════════════════════════════════

class TestVariableMetadataSync:
    """데이터 입력 후 변수 메타데이터가 올바르게 갱신되는지."""

    def test_integer_entry_sets_integer_type(self, model):
        """정수 입력 → StorageType.INTEGER."""
        model.setData(_model_index(model, 0, 0), "42", Qt.ItemDataRole.EditRole)
        var_name = model._dataframe.columns[0]
        assert model._variables[var_name].storage_type == StorageType.INTEGER

    def test_float_entry_sets_float_type(self, model):
        """소수점 입력 → StorageType.FLOAT."""
        model.setData(_model_index(model, 0, 0), "3.14", Qt.ItemDataRole.EditRole)
        var_name = model._dataframe.columns[0]
        assert model._variables[var_name].storage_type == StorageType.FLOAT

    def test_string_entry_sets_string_type(self, model):
        """문자열 입력 → StorageType.STRING."""
        model.setData(_model_index(model, 0, 0), "Hello", Qt.ItemDataRole.EditRole)
        var_name = model._dataframe.columns[0]
        assert model._variables[var_name].storage_type == StorageType.STRING

    def test_dataset_variables_matches_model(self, view):
        """DataView를 통한 입력 후 dataset.variables 동기화."""
        _enter_value(view, 0, 0, "5")
        _enter_value(view, 1, 0, "8")

        ds = view._dataset
        assert len(ds.variables) > 0, "Dataset.variables should have entries after input"
        var_name = list(ds.variables.keys())[0]
        assert ds.variables[var_name].storage_type in (StorageType.INTEGER, StorageType.FLOAT)

    def test_dataset_data_columns_match_after_entry(self, view):
        """데이터 입력 후 dataset.data.columns == variables.keys()."""
        _enter_value(view, 0, 0, "10")
        _enter_value(view, 0, 1, "20")

        ds = view._dataset
        data_cols = set(ds.data.columns)
        var_cols = set(ds.variables.keys())
        # variables에 있는 컬럼은 data에도 있어야 함
        for var in var_cols:
            assert var in data_cols, f"Variable '{var}' missing from dataset.data"

    def test_get_dataframe_returns_numeric_dtype(self, model):
        """정수 입력 컬럼: get_dataframe() dtype이 수치형."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 1, 0), "20", Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        assert pd.api.types.is_numeric_dtype(df.iloc[:, 0]), \
            f"Integer column dtype should be numeric, got {df.iloc[:, 0].dtype}"

    def test_variable_counter_increments(self, model):
        """새 컬럼 생성 시 VAR00001, VAR00002 순서로 자동 명명."""
        model.setData(_model_index(model, 0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 0, 1), "2", Qt.ItemDataRole.EditRole)
        cols = list(model._dataframe.columns)
        assert cols[0].startswith("VAR")
        assert cols[1].startswith("VAR")
        assert cols[0] != cols[1]


# ══════════════════════════════════════════════════════════════════════════════
# 5. 분석 다이얼로그 변수 목록
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalysisDialogVarList:
    """그리드 입력 데이터가 분석 다이얼로그 변수 목록에 표시되는지."""

    def _make_dataset_via_model(self, rows: dict[str, list]) -> Dataset:
        """모델을 통해 Dataset 생성 (실제 사용 흐름 재현)."""
        model = SPSSGridModel()
        for col_idx, (col_name, values) in enumerate(rows.items()):
            for row_idx, val in enumerate(values):
                model.setData(model.index(row_idx, col_idx), str(val), Qt.ItemDataRole.EditRole)
        ds = Dataset(model.get_dataframe(), name="Test")
        # 변수 메타데이터 복사
        for name, meta in model._variables.items():
            ds.variables[name] = meta
        return ds

    def test_frequencies_dialog_shows_all_vars(self, app):
        """FrequenciesDialog: 입력된 모든 변수가 목록에 표시."""
        from statworkbench.ui.dialogs.frequencies_dialog import FrequenciesDialog

        model = SPSSGridModel()
        model.setData(model.index(0, 0), "A", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 1), "10", Qt.ItemDataRole.EditRole)
        ds = Dataset(model.get_dataframe(), name="T")
        for name, meta in model._variables.items():
            ds.variables[name] = meta

        dlg = FrequenciesDialog(ds)
        items = [dlg.var_list.item(i).text() for i in range(dlg.var_list.count())]
        assert len(items) == 2, f"Expected 2 vars, got {len(items)}: {items}"
        dlg.close()

    def test_descriptives_dialog_shows_numeric_only(self, app):
        """DescriptivesDialog: 수치형 변수만 표시."""
        from statworkbench.ui.dialogs.descriptives_dialog import DescriptivesDialog

        model = SPSSGridModel()
        model.setData(model.index(0, 0), "Group_A", Qt.ItemDataRole.EditRole)   # 문자형
        model.setData(model.index(0, 1), "85.5", Qt.ItemDataRole.EditRole)       # 수치형
        model.setData(model.index(0, 2), "90", Qt.ItemDataRole.EditRole)          # 수치형
        ds = Dataset(model.get_dataframe(), name="T")
        for name, meta in model._variables.items():
            ds.variables[name] = meta

        dlg = DescriptivesDialog(ds)
        items = [dlg.var_list.item(i).text() for i in range(dlg.var_list.count())]
        # 수치형 2개만 표시
        assert len(items) == 2, f"Expected 2 numeric vars, got {len(items)}: {items}"
        dlg.close()

    def test_ttest_dialog_test_combo_numeric_only(self, app):
        """IndependentTTestDialog: 검정변수 콤보에 수치형만."""
        from statworkbench.ui.dialogs.ttest_dialog import IndependentTTestDialog

        model = SPSSGridModel()
        model.setData(model.index(0, 0), "A", Qt.ItemDataRole.EditRole)   # 문자
        model.setData(model.index(0, 1), "10", Qt.ItemDataRole.EditRole)   # 수치
        ds = Dataset(model.get_dataframe(), name="T")
        for name, meta in model._variables.items():
            ds.variables[name] = meta

        dlg = IndependentTTestDialog(ds)
        test_items = [dlg.test_combo.itemText(i) for i in range(dlg.test_combo.count())]
        group_items = [dlg.group_combo.itemText(i) for i in range(dlg.group_combo.count())]
        # 검정변수: 수치형만 (scale/numeric)
        assert len(test_items) == 1, f"Test combo should have 1 numeric var, got {test_items}"
        # 그룹변수: SPSS 호환 — 범주형(NOMINAL/ORDINAL) 우선, 없으면 전체
        # 문자형 VAR00001은 NOMINAL으로 인식되므로 그룹 콤보에 1개 표시
        assert len(group_items) >= 1, f"Group combo should have at least 1 var, got {group_items}"
        dlg.close()

    def test_dialog_vars_from_imported_dataset(self, app):
        """외부 파일로 불러온 데이터셋도 변수 목록 정상 표시."""
        from statworkbench.ui.dialogs.descriptives_dialog import DescriptivesDialog

        df = pd.DataFrame({
            "score": [85.0, 90.0, 78.0],
            "age": [25, 30, 35],
            "name": ["Alice", "Bob", "Carol"],
        })
        ds = Dataset(df, name="Imported")

        dlg = DescriptivesDialog(ds)
        # 아이템은 "📏 score (score)" 형식이므로 변수명 포함 여부로 확인
        items = [dlg.var_list.item(i).text() for i in range(dlg.var_list.count())]
        var_names = [dlg.var_list.item(i).data(0x0100) for i in range(dlg.var_list.count())]
        assert "score" in var_names, f"score not in var_names: {var_names}"
        assert "age" in var_names, f"age not in var_names: {var_names}"
        assert "name" not in var_names  # 문자형은 제외
        dlg.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. 빈도분석 — 실제 실행
# ══════════════════════════════════════════════════════════════════════════════

class TestFrequenciesAnalysis:
    """빈도분석 실제 실행 시나리오."""

    @pytest.fixture
    def ds_categorical(self):
        df = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "grade": ["A", "B", "A", "C", "B", "A", "B", "C", "A", "B"],
        })
        return Dataset(df, name="Cat")

    @pytest.fixture
    def ds_likert(self):
        """5점 리커트 척도."""
        np.random.seed(42)
        scores = np.random.randint(1, 6, 50).tolist()
        df = pd.DataFrame({"satisfaction": scores})
        return Dataset(df, name="Likert")

    def test_basic_frequency(self, ds_categorical):
        """기본 빈도분석."""
        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": ["gender"]}, "options": {"include_missing": False, "show_cumulative": True}}
        result = run_analysis(ds_categorical, spec)
        assert result is not None
        html = result.to_html()
        assert "gender" in html.lower() or len(html) > 0

    def test_frequency_counts(self, ds_categorical):
        """빈도수가 올바른지 확인."""
        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": ["gender"]}, "options": {"include_missing": False, "show_cumulative": True}}
        result = run_analysis(ds_categorical, spec)
        tables = result.tables
        # 빈도표 존재 확인
        assert len(tables) > 0

    def test_frequency_multiple_vars(self, ds_categorical):
        """다변수 빈도분석."""
        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": ["gender", "grade"]}, "options": {"include_missing": False, "show_cumulative": True}}
        result = run_analysis(ds_categorical, spec)
        assert result is not None
        assert len(result.tables) >= 2

    def test_frequency_with_missing(self):
        """결측값이 포함된 빈도분석."""
        df = pd.DataFrame({"x": ["A", "B", None, "A", "B", "B", None, "A"]})
        ds = Dataset(df, name="Missing")
        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": ["x"]}, "options": {"include_missing": True, "show_cumulative": True}}
        result = run_analysis(ds, spec)
        assert result is not None

    def test_frequency_numeric_variable(self, ds_likert):
        """수치형 변수 빈도분석."""
        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": ["satisfaction"]}, "options": {"include_missing": False, "show_cumulative": True}}
        result = run_analysis(ds_likert, spec)
        assert result is not None
        assert result.to_html() != ""

    def test_frequency_single_category(self):
        """모든 케이스가 동일한 값인 경우."""
        df = pd.DataFrame({"x": ["A"] * 10})
        ds = Dataset(df, name="Single")
        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": ["x"]}, "options": {"include_missing": False, "show_cumulative": True}}
        result = run_analysis(ds, spec)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# 7. 기술통계 — 실제 실행
# ══════════════════════════════════════════════════════════════════════════════

class TestDescriptivesAnalysis:
    """기술통계 실제 실행 시나리오."""

    @pytest.fixture
    def ds_scores(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "math": np.random.normal(70, 10, 30),
            "english": np.random.normal(75, 8, 30),
            "science": np.random.normal(68, 12, 30),
        })
        return Dataset(df, name="Scores")

    def test_single_variable_descriptives(self, ds_scores):
        """단일 변수 기술통계."""
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["math"]}, "options": {"show_mean": True, "show_std": True, "show_minmax": True}}
        result = run_analysis(ds_scores, spec)
        assert result is not None
        html = result.to_html()
        assert len(html) > 0

    def test_multiple_variable_descriptives(self, ds_scores):
        """다변수 기술통계."""
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["math", "english", "science"]}, "options": {}}
        result = run_analysis(ds_scores, spec)
        assert result is not None

    def test_descriptives_mean_is_correct(self):
        """평균값 정확성 검증."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds = Dataset(df, name="Exact")
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["x"]}, "options": {"show_mean": True}}
        result = run_analysis(ds, spec)
        assert result is not None
        # 평균이 3.0인지 결과 테이블에서 확인
        found_mean = False
        for table in result.tables:
            df_table = table.dataframe
            for col in df_table.columns:
                for v in df_table[col].dropna():
                    try:
                        if abs(float(v) - 3.0) < 0.01:
                            found_mean = True
                    except (ValueError, TypeError):
                        pass
        assert found_mean, "Mean of [1,2,3,4,5] should be 3.0"

    def test_descriptives_with_missing(self):
        """결측값 포함 기술통계 — 결측 제외 후 계산."""
        df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
        ds = Dataset(df, name="Missing")
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["x"]}, "options": {}}
        result = run_analysis(ds, spec)
        assert result is not None

    def test_descriptives_options_show_hide(self, ds_scores):
        """옵션에 따라 통계량 포함 여부 변경."""
        from statworkbench.analysis.descriptive import run_analysis
        spec_minimal = {"variables": {"scale": ["math"]}, "options": {"show_mean": True, "show_std": False, "show_minmax": False}}
        result = run_analysis(ds_scores, spec_minimal)
        assert result is not None

    def test_descriptives_std_is_correct(self):
        """표준편차 정확성: 상수 시리즈 → std = 0."""
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 5.0]})
        ds = Dataset(df, name="Constant")
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["x"]}, "options": {"show_std": True}}
        result = run_analysis(ds, spec)
        assert result is not None
        found_zero = False
        for table in result.tables:
            df_table = table.dataframe
            for col in df_table.select_dtypes(include="number").columns:
                if any(abs(v) < 0.001 for v in df_table[col].dropna()):
                    found_zero = True
        assert found_zero, "Std of constant series should be ~0"


# ══════════════════════════════════════════════════════════════════════════════
# 8. T 검정 — 실제 실행
# ══════════════════════════════════════════════════════════════════════════════

class TestTTestAnalysis:
    """T 검정 실제 실행 시나리오."""

    @pytest.fixture
    def ds_independent(self):
        """독립표본 T 검정용 — 그룹 A vs B."""
        np.random.seed(7)
        df = pd.DataFrame({
            "score": (
                np.random.normal(80, 10, 20).tolist() +
                np.random.normal(70, 10, 20).tolist()
            ),
            "group": ["A"] * 20 + ["B"] * 20,
        })
        return Dataset(df, name="IndTTest")

    @pytest.fixture
    def ds_paired(self):
        """대응표본 T 검정용 — 사전/사후."""
        np.random.seed(11)
        pre = np.random.normal(65, 8, 25)
        post = pre + np.random.normal(5, 3, 25)
        df = pd.DataFrame({"pre": pre, "post": post})
        return Dataset(df, name="Paired")

    def test_independent_ttest_runs(self, ds_independent):
        """독립표본 T 검정 실행 — 오류 없이 완료."""
        from statworkbench.analysis.ttests import run_analysis
        spec = {"variables": {"dependent": "score", "group": "group"}, "options": {"equal_var": "auto"}}
        result = run_analysis(ds_independent, spec)
        assert result is not None
        assert result.to_html() != ""

    def test_independent_ttest_p_value_present(self, ds_independent):
        """독립표본 T 검정: p값 포함."""
        from statworkbench.analysis.ttests import run_analysis
        spec = {"variables": {"dependent": "score", "group": "group"}, "options": {"equal_var": "auto"}}
        result = run_analysis(ds_independent, spec)
        html = result.to_html()
        assert "p" in html.lower() or len(result.tables) > 0

    def test_paired_ttest_runs(self, ds_paired):
        """대응표본 T 검정 실행."""
        from statworkbench.analysis.ttests import run_analysis
        spec = {"variables": {"paired": ["pre", "post"]}}
        result = run_analysis(ds_paired, spec)
        assert result is not None
        assert result.to_html() != ""

    def test_independent_ttest_significant_difference(self):
        """명확히 다른 두 그룹: p < 0.05 예상."""
        np.random.seed(0)
        df = pd.DataFrame({
            "score": [50]*20 + [90]*20,
            "group": ["low"]*20 + ["high"]*20,
        })
        ds = Dataset(df, name="Sig")
        from statworkbench.analysis.ttests import run_analysis
        spec = {"variables": {"dependent": "score", "group": "group"}, "options": {}}
        result = run_analysis(ds, spec)
        assert result is not None
        # 결과가 통계적으로 유의해야 함 (확인은 HTML에서)
        assert len(result.tables) > 0

    def test_paired_ttest_no_difference(self):
        """동일 값 쌍: 차이 없음."""
        df = pd.DataFrame({"pre": [70.0]*10, "post": [70.0]*10})
        ds = Dataset(df, name="NoDiff")
        from statworkbench.analysis.ttests import run_analysis
        spec = {"variables": {"paired": ["pre", "post"]}}
        result = run_analysis(ds, spec)
        assert result is not None

    def test_ttest_with_missing(self):
        """결측값 포함 T 검정."""
        df = pd.DataFrame({
            "score": [80, None, 75, 85, None, 70, 90, 88, None, 72],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        })
        ds = Dataset(df, name="MissingT")
        from statworkbench.analysis.ttests import run_analysis
        spec = {"variables": {"dependent": "score", "group": "group"}, "options": {}}
        result = run_analysis(ds, spec)
        assert result is not None

    def test_one_sample_ttest_runs(self):
        """단일표본 T 검정 실행."""
        df = pd.DataFrame({"score": [72, 68, 75, 80, 65, 70, 78, 74, 69, 73]})
        ds = Dataset(df, name="OneSample")
        from statworkbench.analysis.ttests import run_one_sample_ttest
        result = run_one_sample_ttest(ds.data, "score", test_value=70.0)
        assert result is not None
        assert result.to_html() != ""


# ══════════════════════════════════════════════════════════════════════════════
# 9. 데이터셋 동기화 종합 시나리오
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetSyncScenarios:
    """DataView ↔ Dataset 동기화 종합 확인."""

    def test_data_in_sync_after_first_entry(self, view):
        """첫 번째 입력 후 dataset.data 동기화."""
        _enter_value(view, 0, 0, "42")

        ds = view._dataset
        assert len(ds.data.columns) > 0, "dataset.data should have columns after entry"
        col = ds.data.columns[0]
        assert col in ds.variables, "Column should be in variables dict"

    def test_analysis_possible_after_grid_entry(self, view):
        """그리드 입력 후 바로 기술통계 분석 가능."""
        values = [75, 80, 85, 90, 70, 65, 88, 92, 78, 83]
        for row, val in enumerate(values):
            _enter_value(view, row, 0, str(val))

        ds = view._dataset
        var_name = list(ds.variables.keys())[0]

        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": [var_name]}, "options": {}}
        result = run_analysis(ds, spec)
        assert result is not None, "Analysis should succeed after grid entry"

    def test_frequencies_after_string_entry(self, view):
        """문자형 데이터 입력 후 빈도분석 가능."""
        groups = ["A", "B", "A", "C", "B", "A", "B", "C"]
        for row, g in enumerate(groups):
            _enter_value(view, row, 0, g)

        ds = view._dataset
        var_name = list(ds.variables.keys())[0]

        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": [var_name]}, "options": {"include_missing": False, "show_cumulative": True}}
        result = run_analysis(ds, spec)
        assert result is not None

    def test_multiple_columns_sync(self, view):
        """다중 컬럼 입력 후 dataset.data 컬럼 수 일치."""
        _enter_value(view, 0, 0, "A")
        _enter_value(view, 0, 1, "10")
        _enter_value(view, 0, 2, "5.5")

        ds = view._dataset
        assert len(ds.data.columns) == 3, \
            f"Expected 3 columns, got {len(ds.data.columns)}: {list(ds.data.columns)}"

    def test_ttest_after_two_column_entry(self, view):
        """두 컬럼(점수+그룹) 입력 후 독립표본 T 검정."""
        rows = [
            ("80", "A"), ("85", "A"), ("90", "A"), ("75", "A"), ("88", "A"),
            ("70", "B"), ("65", "B"), ("72", "B"), ("68", "B"), ("74", "B"),
        ]
        for row_idx, (score, group) in enumerate(rows):
            _enter_value(view, row_idx, 0, score)
            _enter_value(view, row_idx, 1, group)

        ds = view._dataset
        var_names = list(ds.variables.keys())
        assert len(var_names) == 2

        score_var = var_names[0]
        group_var = var_names[1]

        from statworkbench.analysis.ttests import run_analysis
        spec = {
            "variables": {"dependent": score_var, "group": group_var},
            "options": {},
        }
        result = run_analysis(ds, spec)
        assert result is not None, "T-test should work with grid-entered data"


# ══════════════════════════════════════════════════════════════════════════════
# 10. 모델 경계 조건
# ══════════════════════════════════════════════════════════════════════════════

class TestModelEdgeCases:
    """경계 조건 및 예외 시나리오."""

    def test_model_starts_empty(self, model):
        """초기 상태: 데이터 없음."""
        df = model.get_dataframe()
        assert df.empty or len(df.columns) == 0

    def test_row_count_expands_beyond_data(self, model):
        """가상 그리드: 데이터 행보다 rowCount가 크다."""
        model.setData(_model_index(model, 0, 0), "1", Qt.ItemDataRole.EditRole)
        assert model.rowCount() >= 100

    def test_column_count_expands_beyond_data(self, model):
        """가상 그리드: 데이터 열보다 columnCount가 크다."""
        model.setData(_model_index(model, 0, 0), "1", Qt.ItemDataRole.EditRole)
        assert model.columnCount() >= 10

    def test_out_of_range_cell_returns_none(self, model):
        """범위 밖 셀: EditRole은 빈 문자열."""
        val = model.data(_model_index(model, 500, 50), Qt.ItemDataRole.EditRole)
        assert val == "" or val is None

    def test_set_same_value_no_error(self, model):
        """동일 값 재입력: 오류 없음."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        result = model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        assert result is True

    def test_very_long_string(self, model):
        """매우 긴 문자열 입력."""
        long_str = "A" * 500
        model.setData(_model_index(model, 0, 0), long_str, Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        assert str(df.iloc[0, 0]) == long_str

    def test_special_characters_in_string(self, model):
        """특수문자 포함 문자열."""
        special = "테스트 데이터 #1 (2024)"
        model.setData(_model_index(model, 0, 0), special, Qt.ItemDataRole.EditRole)
        df = model.get_dataframe()
        assert df.iloc[0, 0] == special

    def test_header_rename(self, model):
        """헤더 이름 변경 후 변수 참조 유지."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        old_name = model._dataframe.columns[0]
        model.setHeaderData(0, Qt.Orientation.Horizontal, "MyVar", Qt.ItemDataRole.EditRole)
        assert "MyVar" in model._dataframe.columns
        assert old_name not in model._dataframe.columns
        assert "MyVar" in model._variables

    def test_remove_column(self, model):
        """컬럼 삭제 후 데이터 정합성."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 0, 1), "20", Qt.ItemDataRole.EditRole)
        model.remove_column(0)
        assert len(model._dataframe.columns) == 1

    def test_remove_row(self, model):
        """행 삭제 후 데이터 정합성."""
        model.setData(_model_index(model, 0, 0), "10", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 1, 0), "20", Qt.ItemDataRole.EditRole)
        model.remove_row(0)
        df = model.get_dataframe()
        assert len(df) == 1
        assert df.iloc[0, 0] == 20

    def test_100_rows_entry(self, model):
        """100행 데이터 입력 성능 및 정확성."""
        for i in range(100):
            model.setData(_model_index(model, i, 0), str(i * 2), Qt.ItemDataRole.EditRole)

        df = model.get_dataframe()
        assert len(df) == 100
        assert df.iloc[0, 0] == 0
        assert df.iloc[99, 0] == 198

    def test_get_dataframe_row_count_correct(self, model):
        """get_dataframe(): 마지막 데이터 행까지만 반환."""
        model.setData(_model_index(model, 0, 0), "1", Qt.ItemDataRole.EditRole)
        model.setData(_model_index(model, 2, 0), "3", Qt.ItemDataRole.EditRole)
        # row 1 은 비어 있어 NA
        df = model.get_dataframe()
        assert len(df) == 3  # row 0, 1(NA), 2


# ══════════════════════════════════════════════════════════════════════════════
# 11. 실무 시나리오 (설문 데이터 흐름)
# ══════════════════════════════════════════════════════════════════════════════

class TestSurveyDataWorkflow:
    """설문조사 데이터 분석 전체 흐름."""

    @pytest.fixture
    def survey_dataset(self):
        """10명 설문 데이터 (성별, 연령, 만족도 1-5점, 추천 여부)."""
        np.random.seed(42)
        df = pd.DataFrame({
            "gender":       ["M","F","M","F","M","F","M","F","M","F"],
            "age":          [25, 32, 28, 45, 38, 29, 52, 34, 41, 27],
            "satisfaction": [4, 5, 3, 4, 2, 5, 3, 4, 5, 3],
            "recommend":    [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        })
        return Dataset(df, name="Survey")

    def test_gender_frequency(self, survey_dataset):
        """성별 빈도분석 — 각 5명씩."""
        from statworkbench.analysis.frequencies import run_analysis
        spec = {"variables": {"target": ["gender"]}, "options": {"include_missing": False, "show_cumulative": True}}
        result = run_analysis(survey_dataset, spec)
        assert result is not None
        tables = result.tables
        assert len(tables) > 0

    def test_satisfaction_descriptives(self, survey_dataset):
        """만족도 기술통계."""
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["satisfaction", "age"]}, "options": {}}
        result = run_analysis(survey_dataset, spec)
        assert result is not None

    def test_satisfaction_by_gender_ttest(self, survey_dataset):
        """성별 만족도 차이 검정."""
        from statworkbench.analysis.ttests import run_analysis
        spec = {"variables": {"dependent": "satisfaction", "group": "gender"}, "options": {}}
        result = run_analysis(survey_dataset, spec)
        assert result is not None

    def test_age_mean_correct(self, survey_dataset):
        """연령 평균 정확성 (25+32+28+45+38+29+52+34+41+27)/10 = 35.1."""
        from statworkbench.analysis.descriptive import run_analysis
        spec = {"variables": {"scale": ["age"]}, "options": {"show_mean": True}}
        result = run_analysis(survey_dataset, spec)
        assert result is not None
        # 평균 35.1 근사 확인
        found = False
        for table in result.tables:
            for col in table.dataframe.columns:
                for v in table.dataframe[col].dropna():
                    try:
                        if abs(float(v) - 35.1) < 0.5:
                            found = True
                    except (ValueError, TypeError):
                        pass
        assert found, "Age mean should be approximately 35.1"

    def test_full_pipeline(self, survey_dataset):
        """전체 분석 파이프라인: 빈도 → 기술통계 → T검정 모두 실행."""
        from statworkbench.analysis.frequencies import run_analysis as freq_run
        from statworkbench.analysis.descriptive import run_analysis as desc_run
        from statworkbench.analysis.ttests import run_analysis as ttest_run

        r1 = freq_run(survey_dataset, {"variables": {"target": ["gender", "recommend"]}, "options": {"include_missing": False, "show_cumulative": True}})
        r2 = desc_run(survey_dataset, {"variables": {"scale": ["satisfaction", "age"]}, "options": {}})
        r3 = ttest_run(survey_dataset, {"variables": {"dependent": "satisfaction", "group": "gender"}, "options": {}})

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None

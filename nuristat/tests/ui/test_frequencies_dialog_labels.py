"""FrequenciesDialog 변수 라벨 표시 테스트."""

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.ui.dialogs.frequencies_dialog import FrequenciesDialog


@pytest.fixture
def app(qapp):
    return qapp


def _make_dataset_with_labels():
    df = pd.DataFrame({"성별": [1, 2, 1], "나이": [30, 40, 50]})
    variables = {
        "성별": VariableMeta(name="성별", label="Sex / 성별"),
        "나이": VariableMeta(name="나이", label="Age / 나이"),
    }
    return Dataset(name="test", data=df, variables=variables)


def test_freq_dialog_shows_label_in_list(app):
    """변수 라벨이 리스트 항목 텍스트에 포함돼야 한다."""
    ds = _make_dataset_with_labels()
    dlg = FrequenciesDialog(ds)
    texts = [dlg.var_list.item(i).text() for i in range(dlg.var_list.count())]
    assert any("Sex / 성별" in t for t in texts)
    assert any("Age / 나이" in t for t in texts)


def test_freq_dialog_userrole_stores_varname(app):
    """UserRole에 실제 변수명(내부 ID)이 저장돼야 한다."""
    ds = _make_dataset_with_labels()
    dlg = FrequenciesDialog(ds)
    item0 = dlg.var_list.item(0)
    var_name = item0.data(Qt.ItemDataRole.UserRole)
    assert var_name == "성별"


def test_freq_dialog_no_labels_shows_varname(app):
    """라벨 없는 변수는 변수명 그대로 표시돼야 한다."""
    df = pd.DataFrame({"age": [10, 20]})
    ds = Dataset(name="test", data=df)
    dlg = FrequenciesDialog(ds)
    texts = [dlg.var_list.item(i).text() for i in range(dlg.var_list.count())]
    assert any("age" in t for t in texts)


def test_freq_dialog_run_uses_userrole(app, monkeypatch):
    """실행 시 UserRole 변수명으로 분석 스펙을 구성해야 한다."""
    ds = _make_dataset_with_labels()
    dlg = FrequenciesDialog(ds)
    dlg.var_list.setCurrentRow(0)

    captured = {}

    def fake_start_analysis(run_fn, dataset, spec):
        captured["spec"] = spec

    monkeypatch.setattr(dlg, "_start_analysis", fake_start_analysis)
    dlg._run()

    assert captured.get("spec") is not None
    assert "성별" in captured["spec"]["variables"]["target"]

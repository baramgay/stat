"""구문 편집기 실제 실행 테스트 — T-TEST, RECODE, SELECT IF."""

import pandas as pd
import pytest

from nuristat.core.dataset import Dataset
from nuristat.core.variable import VariableMeta
from nuristat.core.typing import StorageType, MeasureType
from nuristat.ui.syntax_editor import SyntaxEditor


@pytest.fixture
def dataset():
    df = pd.DataFrame({
        "score": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "sex":   [0, 0, 0, 1, 1, 1],
        "age":   [20, 25, 30, 35, 40, 45],
    })
    variables = {
        "score": VariableMeta(name="score", label="점수", storage_type=StorageType.FLOAT, measure=MeasureType.SCALE),
        "sex":   VariableMeta(name="sex",   label="성별", storage_type=StorageType.INTEGER, measure=MeasureType.NOMINAL),
        "age":   VariableMeta(name="age",   label="나이", storage_type=StorageType.INTEGER, measure=MeasureType.SCALE),
    }
    return Dataset(name="test", data=df, variables=variables)


@pytest.fixture
def editor(qapp, dataset):
    ed = SyntaxEditor()
    ed.set_dataset(dataset)
    return ed


def test_recode_simple(editor):
    """RECODE var (old=new) → INTO 없으면 원본 변수를 덮어써야 한다."""
    result = editor._execute_recode("RECODE sex (0=100) (1=200)")
    assert "완료" in result
    assert editor._dataset.data["sex"].iloc[0] == 100


def test_recode_into_new_var(editor):
    """RECODE var (old=new) INTO new_var → 새 변수 생성해야 한다."""
    result = editor._execute_recode("RECODE sex (0=10) (1=20) INTO sex_recoded")
    assert "완료" in result
    assert "sex_recoded" in editor._dataset.data.columns
    assert editor._dataset.data["sex_recoded"].iloc[3] == 20


def test_select_if_filters_rows(editor):
    """SELECT IF (score > 30) → 조건 충족 케이스만 남아야 한다."""
    before = len(editor._dataset.data)
    result = editor._execute_select_if("SELECT IF (score > 30)")
    after = len(editor._dataset.data)
    assert after < before
    assert "완료" in result or "→" in result


def test_ttest_emits_analysis_result(editor):
    """T-TEST GROUPS=sex(0 1) /VARIABLES=score → analysis_ready 시그널 방출 검증."""
    emitted = []
    editor.analysis_ready.connect(lambda r: emitted.append(r))
    syntax = "T-TEST GROUPS=sex(0 1) /VARIABLES=score"
    editor._parse_and_execute(syntax)
    assert len(emitted) >= 1


def test_pairs_ttest_emits_analysis_result(editor):
    """T-TEST PAIRS=score WITH age → 대응표본 T검정 analysis_ready 방출 검증."""
    emitted = []
    editor.analysis_ready.connect(lambda r: emitted.append(r))
    syntax = "T-TEST PAIRS=score WITH age"
    editor._parse_and_execute(syntax)
    assert len(emitted) >= 1


def test_pairs_ttest_bad_syntax_no_crash(editor):
    """T-TEST PAIRS without WITH → 오류 없이 안내 메시지만 반환해야 한다."""
    emitted = []
    editor.analysis_ready.connect(lambda r: emitted.append(r))
    syntax = "T-TEST PAIRS=score"  # missing WITH var2
    editor._parse_and_execute(syntax)
    # Should not emit analysis_ready; no crash
    assert len(emitted) == 0


def test_recode_invalid_var_returns_error(editor):
    """존재하지 않는 변수 RECODE → 오류 메시지 반환해야 한다."""
    result = editor._execute_recode("RECODE nonexistent (0=1)")
    assert "없음" in result or "오류" in result


# SE-09 회귀 테스트: COMPUTE 구문이 데이터셋에 실제로 반영되어야 한다
def test_compute_adds_new_variable(editor):
    """COMPUTE new = (score + age) / 2. → 데이터셋에 새 변수 생성."""
    editor._parse_and_execute("COMPUTE mean_val = (score + age) / 2.")
    assert "mean_val" in editor._dataset.data.columns
    expected = (editor._dataset.data["score"] + editor._dataset.data["age"]) / 2
    # 첫 행만 비교 (index 동일성)
    assert abs(editor._dataset.data["mean_val"].iloc[0] - expected.iloc[0]) < 1e-9


def test_compute_overwrites_existing_variable(editor):
    """COMPUTE score = score * 2. → 기존 변수 값 수정."""
    original_first = editor._dataset.data["score"].iloc[0]
    editor._parse_and_execute("COMPUTE score = score * 2.")
    assert abs(editor._dataset.data["score"].iloc[0] - original_first * 2) < 1e-9


def test_compute_scalar_constant(editor):
    """COMPUTE const = 99. → 모든 행에 상수 99 저장 (스칼라 broadcast)."""
    editor._parse_and_execute("COMPUTE const_var = 99.")
    assert "const_var" in editor._dataset.data.columns
    assert (editor._dataset.data["const_var"] == 99).all()

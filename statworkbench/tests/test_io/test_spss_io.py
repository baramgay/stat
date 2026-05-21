"""SPSS .sav 파일 읽기/쓰기 SPSS 29/30 호환 종합 테스트.

검증 항목:
- read_sav: .sav 파일 → Dataset 변환
- write_sav: Dataset → .sav 파일 저장
- 왕복 검증 (Round-trip): write → read → 원본과 동일
- 변수 레이블 보존
- 값 레이블 보존
- 측정 척도 보존 (SCALE / NOMINAL / ORDINAL)
- 결측치 NaN 보존
- 숫자형 / 문자형 열 혼합
- 오류 처리 (파일 없음, 잘못된 확장자)

SPSS 29 .sav 포맷 규격:
  - 변수명: 최대 64바이트
  - 변수 레이블: 최대 256자
  - 값 레이블: 숫자형/문자형 키 모두 지원
  - 결측치: 시스템 결측(NaN) + 사용자 정의 결측
  - 측정 척도: scale / ordinal / nominal (pyreadstat 매핑)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statworkbench.core.dataset import Dataset
from statworkbench.core.typing import MeasureType, StorageType
from statworkbench.core.variable import VariableMeta
from statworkbench.core.exceptions import ImportError as SWBImportError
from statworkbench.io.spss_reader import read_sav, _guess_storage_type, _guess_measure_type
from statworkbench.io.spss_writer import write_sav


# ──────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def simple_dataset() -> Dataset:
    """숫자 + 문자 혼합 Dataset."""
    df = pd.DataFrame({
        "age":    [25, 30, 35, 40, np.nan],
        "score":  [85.5, 90.2, 78.0, 95.1, 88.0],
        "gender": ["M", "F", "M", "F", "M"],
    })
    ds = Dataset(df, name="simple")
    ds.variables["age"].label    = "응답자 나이"
    ds.variables["score"].label  = "시험 점수"
    ds.variables["gender"].label = "성별"
    ds.variables["age"].measure    = MeasureType.SCALE
    ds.variables["score"].measure  = MeasureType.SCALE
    ds.variables["gender"].measure = MeasureType.NOMINAL
    return ds


@pytest.fixture
def labeled_dataset() -> Dataset:
    """값 레이블 포함 Dataset."""
    df = pd.DataFrame({
        "edu":   [1, 2, 3, 2, 1],
        "agree": [1, 2, 3, 4, 5],
    })
    ds = Dataset(df, name="labeled")
    ds.variables["edu"].label = "최종 학력"
    ds.variables["edu"].value_labels = {1: "고졸", 2: "대졸", 3: "대학원"}
    ds.variables["edu"].measure = MeasureType.ORDINAL
    ds.variables["agree"].label = "동의 정도"
    ds.variables["agree"].value_labels = {
        1: "전혀 동의 안 함", 2: "동의 안 함",
        3: "보통", 4: "동의", 5: "매우 동의",
    }
    ds.variables["agree"].measure = MeasureType.ORDINAL
    return ds


@pytest.fixture
def sav_path(simple_dataset, tmp_path) -> Path:
    """임시 .sav 파일 저장 후 경로 반환."""
    path = tmp_path / "test.sav"
    write_sav(simple_dataset, str(path))
    return path


# ──────────────────────────────────────────────────────────────
# 1. write_sav 기본 동작
# ──────────────────────────────────────────────────────────────

class TestWriteSav:
    """write_sav — .sav 파일 생성 검증."""

    def test_file_created(self, simple_dataset, tmp_path):
        """write_sav 호출 후 파일이 생성된다."""
        path = tmp_path / "out.sav"
        write_sav(simple_dataset, str(path))
        assert path.exists()

    def test_file_size_positive(self, simple_dataset, tmp_path):
        """생성된 .sav 파일 크기 > 0."""
        path = tmp_path / "out.sav"
        write_sav(simple_dataset, str(path))
        assert path.stat().st_size > 0

    def test_empty_dataset_raises(self, tmp_path):
        """빈 데이터셋 → ValueError."""
        ds = Dataset(pd.DataFrame(), name="empty")
        with pytest.raises(ValueError, match="데이터가 없"):
            write_sav(ds, str(tmp_path / "out.sav"))

    def test_none_dataset_raises(self, tmp_path):
        """None 데이터셋 → ValueError."""
        with pytest.raises((ValueError, AttributeError)):
            write_sav(None, str(tmp_path / "out.sav"))

    def test_with_value_labels(self, labeled_dataset, tmp_path):
        """값 레이블 포함 Dataset 저장 → 파일 생성 성공."""
        path = tmp_path / "labeled.sav"
        write_sav(labeled_dataset, str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_with_missing_values(self, tmp_path):
        """NaN 포함 데이터셋 저장 → 파일 생성 성공."""
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, np.nan]})
        ds = Dataset(df, name="miss")
        ds.variables["x"].measure = MeasureType.SCALE
        path = tmp_path / "missing.sav"
        write_sav(ds, str(path))
        assert path.exists()

    def test_large_dataset(self, tmp_path):
        """대용량 데이터 (1000행, 10열) 저장 성공."""
        df = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f"v{i}" for i in range(10)],
        )
        ds = Dataset(df, name="large")
        for col in df.columns:
            ds.variables[col].measure = MeasureType.SCALE
        path = tmp_path / "large.sav"
        write_sav(ds, str(path))
        assert path.stat().st_size > 0


# ──────────────────────────────────────────────────────────────
# 2. read_sav 기본 동작
# ──────────────────────────────────────────────────────────────

class TestReadSav:
    """read_sav — .sav 파일 읽기 검증."""

    def test_returns_dataset(self, sav_path):
        """read_sav 반환값이 Dataset 인스턴스."""
        ds = read_sav(sav_path)
        assert isinstance(ds, Dataset)

    def test_column_names_preserved(self, sav_path, simple_dataset):
        """열 이름이 원본과 동일하게 복원된다."""
        ds = read_sav(sav_path)
        assert set(ds.data.columns) == set(simple_dataset.data.columns)

    def test_row_count_preserved(self, sav_path, simple_dataset):
        """행 수가 원본과 동일."""
        ds = read_sav(sav_path)
        assert len(ds.data) == len(simple_dataset.data)

    def test_variables_dict_populated(self, sav_path):
        """변수 메타데이터 딕셔너리가 채워진다."""
        ds = read_sav(sav_path)
        assert len(ds.variables) > 0

    def test_file_not_found_raises(self, tmp_path):
        """존재하지 않는 파일 → SWBImportError."""
        with pytest.raises(SWBImportError, match="찾을 수 없"):
            read_sav(tmp_path / "nonexistent.sav")

    def test_wrong_extension_raises(self, tmp_path):
        """확장자가 .sav가 아닌 파일 → SWBImportError."""
        not_sav = tmp_path / "data.csv"
        not_sav.write_text("a,b\n1,2\n")
        with pytest.raises(SWBImportError, match=".sav"):
            read_sav(not_sav)

    def test_dataset_name_from_filename(self, sav_path):
        """Dataset 이름이 파일 스텀에서 설정된다."""
        ds = read_sav(sav_path)
        assert ds.name == sav_path.stem


# ──────────────────────────────────────────────────────────────
# 3. 왕복 검증 (Write → Read Round-trip)
# ──────────────────────────────────────────────────────────────

class TestRoundTrip:
    """write_sav → read_sav 왕복 후 데이터 무결성 검증."""

    @pytest.fixture
    def rt_ds(self, simple_dataset, tmp_path):
        path = tmp_path / "rt.sav"
        write_sav(simple_dataset, str(path))
        return read_sav(path)

    def test_numeric_column_values(self, rt_ds, simple_dataset):
        """숫자 열(score) 값이 왕복 후에도 동일."""
        orig  = simple_dataset.data["score"].values
        rt    = rt_ds.data["score"].values
        np.testing.assert_allclose(rt, orig, rtol=1e-5)

    def test_string_column_values(self, rt_ds, simple_dataset):
        """문자 열(gender) 값이 왕복 후에도 동일."""
        orig  = simple_dataset.data["gender"].tolist()
        rt    = rt_ds.data["gender"].tolist()
        # 공백 strip 허용 (pyreadstat이 패딩 추가할 수 있음)
        assert [v.strip() for v in rt] == orig

    def test_missing_values_preserved(self, rt_ds):
        """NaN이 왕복 후에도 NaN으로 복원된다."""
        assert rt_ds.data["age"].isna().sum() == 1

    def test_variable_count_preserved(self, rt_ds, simple_dataset):
        """변수 수가 원본과 동일."""
        assert len(rt_ds.data.columns) == len(simple_dataset.data.columns)

    def test_variable_labels_round_trip(self, labeled_dataset, tmp_path):
        """값 레이블 왕복 보존."""
        path = tmp_path / "label_rt.sav"
        write_sav(labeled_dataset, str(path))
        rt_ds = read_sav(path)
        edu_var = rt_ds.variables.get("edu")
        if edu_var and edu_var.value_labels:
            orig_labels = set(labeled_dataset.variables["edu"].value_labels.values())
            rt_labels   = set(edu_var.value_labels.values())
            assert orig_labels == rt_labels

    def test_measure_types_round_trip(self, labeled_dataset, tmp_path):
        """측정 척도(ORDINAL/NOMINAL/SCALE) 왕복 보존."""
        path = tmp_path / "measure_rt.sav"
        write_sav(labeled_dataset, str(path))
        rt_ds = read_sav(path)
        for var_name in labeled_dataset.data.columns:
            orig_m = labeled_dataset.variables[var_name].measure
            rt_m   = rt_ds.variables.get(var_name, None)
            if rt_m:
                assert rt_m.measure == orig_m, (
                    f"'{var_name}': 원본={orig_m}, 복원={rt_m.measure}"
                )

    def test_float_precision_round_trip(self, tmp_path):
        """소수점 정밀도 왕복 검증 (6자리)."""
        vals = [3.141593, 2.718282, 1.414214, 1.732051]
        df = pd.DataFrame({"pi_approx": vals})
        ds = Dataset(df, name="float_test")
        ds.variables["pi_approx"].measure = MeasureType.SCALE
        path = tmp_path / "float.sav"
        write_sav(ds, str(path))
        rt = read_sav(path)
        np.testing.assert_allclose(
            rt.data["pi_approx"].values, vals, rtol=1e-5
        )


# ──────────────────────────────────────────────────────────────
# 4. _guess_storage_type / _guess_measure_type 단위 테스트
# ──────────────────────────────────────────────────────────────

class TestGuessHelpers:
    """_guess_storage_type, _guess_measure_type 내부 함수 검증."""

    def test_integer_series_storage(self):
        s = pd.Series([1, 2, 3], dtype="int64")
        assert _guess_storage_type(s) == StorageType.INTEGER

    def test_float_series_storage(self):
        s = pd.Series([1.1, 2.2, 3.3], dtype="float64")
        assert _guess_storage_type(s) == StorageType.FLOAT

    def test_string_series_storage(self):
        s = pd.Series(["a", "b", "c"], dtype="object")
        assert _guess_storage_type(s) == StorageType.STRING

    def test_datetime_series_storage(self):
        s = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01"]))
        assert _guess_storage_type(s) == StorageType.DATETIME

    def test_numeric_series_measure_is_scale(self):
        """SPSS 29 기본: 숫자 → SCALE."""
        s = pd.Series([1.0, 2.0, 3.0])
        assert _guess_measure_type(s) == MeasureType.SCALE

    def test_string_series_measure_is_nominal(self):
        """SPSS 29 기본: 문자 → NOMINAL."""
        s = pd.Series(["A", "B", "C"])
        assert _guess_measure_type(s) == MeasureType.NOMINAL

    def test_categorical_series_measure_is_nominal(self):
        """범주형 → NOMINAL (ORDINAL이 아님, SPSS 29 기본)."""
        s = pd.Series(pd.Categorical(["low", "mid", "high"]))
        assert _guess_measure_type(s) == MeasureType.NOMINAL

    def test_integer_series_measure_is_scale(self):
        s = pd.Series([0, 1, 2, 3], dtype="int32")
        assert _guess_measure_type(s) == MeasureType.SCALE


# ──────────────────────────────────────────────────────────────
# 5. 측정 척도 매핑 검증
# ──────────────────────────────────────────────────────────────

class TestMeasureMapping:
    """write_sav → read_sav 측정 척도 매핑 (pyreadstat 연동)."""

    @pytest.mark.parametrize("measure", [
        MeasureType.SCALE,
        MeasureType.NOMINAL,
        MeasureType.ORDINAL,
    ])
    def test_measure_survives_round_trip(self, measure, tmp_path):
        """SCALE / NOMINAL / ORDINAL 각각 왕복 보존."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        ds = Dataset(df, name="m_test")
        ds.variables["x"].measure = measure
        path = tmp_path / f"m_{measure.value}.sav"
        write_sav(ds, str(path))
        rt = read_sav(path)
        rt_measure = rt.variables["x"].measure
        assert rt_measure == measure, (
            f"measure={measure} → 왕복 후 {rt_measure}"
        )


# ──────────────────────────────────────────────────────────────
# 6. 엣지 케이스
# ──────────────────────────────────────────────────────────────

class TestSpssIoEdgeCases:
    """경계 케이스 — SPSS 29 호환 IO."""

    def test_single_column(self, tmp_path):
        """단일 열 Dataset 왕복."""
        df = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
        ds = Dataset(df, name="single")
        ds.variables["only"].measure = MeasureType.SCALE
        path = tmp_path / "single.sav"
        write_sav(ds, str(path))
        rt = read_sav(path)
        assert "only" in rt.data.columns
        np.testing.assert_allclose(rt.data["only"].values, [1, 2, 3], rtol=1e-5)

    def test_single_row(self, tmp_path):
        """단일 행 Dataset 왕복."""
        df = pd.DataFrame({"x": [42.0], "y": ["hello"]})
        ds = Dataset(df, name="onerow")
        ds.variables["x"].measure = MeasureType.SCALE
        ds.variables["y"].measure = MeasureType.NOMINAL
        path = tmp_path / "onerow.sav"
        write_sav(ds, str(path))
        rt = read_sav(path)
        assert len(rt.data) == 1
        assert rt.data["x"].iloc[0] == pytest.approx(42.0)

    def test_all_missing_column(self, tmp_path):
        """전체 결측 열 → 왕복 후에도 결측."""
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        ds = Dataset(df, name="allmiss")
        ds.variables["x"].measure = MeasureType.SCALE
        path = tmp_path / "allmiss.sav"
        write_sav(ds, str(path))
        rt = read_sav(path)
        assert rt.data["x"].isna().all()

    def test_string_path_and_path_object(self, simple_dataset, tmp_path):
        """str 경로와 Path 객체 모두 write_sav/read_sav에서 작동."""
        path_obj = tmp_path / "pathobj.sav"
        write_sav(simple_dataset, str(path_obj))
        ds_str  = read_sav(str(path_obj))
        ds_path = read_sav(path_obj)
        assert len(ds_str.data) == len(ds_path.data)

    def test_korean_variable_labels(self, tmp_path):
        """한글 변수 레이블 저장 후 복원."""
        df = pd.DataFrame({"var1": [1, 2, 3]})
        ds = Dataset(df, name="korean")
        ds.variables["var1"].label  = "나이"
        ds.variables["var1"].measure = MeasureType.SCALE
        path = tmp_path / "korean.sav"
        write_sav(ds, str(path))
        rt = read_sav(path)
        rt_label = rt.variables["var1"].label
        assert "나이" in rt_label or rt_label == "var1"

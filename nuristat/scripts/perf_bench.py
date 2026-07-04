"""NuriStat 성능 벤치마크 하네스 (P5-1).

성능개선계획_2026-07.html 의 6개 지표를 측정해 docs/perf_baseline.json 에 저장한다.
Phase 완료 시 재실행하여 diff를 커밋 메시지에 남긴다.

사용법:
    PYTHONPATH=src python scripts/perf_bench.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def startup_import_s() -> float:
    """콜드 프로세스에서 nuristat.app 임포트에 걸리는 총 시간(초)."""
    env = {"PYTHONPATH": str(SRC)}
    import os

    full_env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "from nuristat.app import NuriStatApp"],
        capture_output=True,
        text=True,
        env=full_env,
        check=True,
    )
    # 마지막 라인이 최상위(cumulative) 임포트 — self는 옆 컬럼, cumulative는 그 다음
    last_line = proc.stderr.strip().splitlines()[-1]
    match = re.search(r"import time:\s*(\d+)\s*\|\s*(\d+)\s*\|", last_line)
    if not match:
        raise RuntimeError(f"importtime 출력 파싱 실패: {last_line!r}")
    cumulative_us = int(match.group(2))
    return cumulative_us / 1_000_000


def _make_grid_model(n_rows: int = 100_000, n_cols: int = 50):
    import numpy as np
    import pandas as pd

    from nuristat.ui.models.spss_grid_model import SPSSGridModel

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((n_rows, n_cols)),
        columns=[f"var{i}" for i in range(n_cols)],
    )
    return SPSSGridModel(dataframe=df)


def _ensure_qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def edit_latency_ms() -> float:
    """100,000행 x 50열 그리드에서 셀 1개 setData 호출 지연(ms)."""
    _ensure_qapp()
    model = _make_grid_model()
    index = model.index(50_000, 25)

    start = time.perf_counter()
    model.setData(index, 3.14)
    elapsed = time.perf_counter() - start
    return elapsed * 1000


def dataview_edit_sync_ms(n_rows: int = 100_000, n_cols: int = 50) -> dict[str, float]:
    """DataView 경유 셀 1개 편집 시 Dataset.data 동기화 비용(ms, P1-2).

    deferred_ms: 편집 직후 실제 소요(지연 동기화 — sync_dataset 미호출).
    eager_sync_ms: 같은 편집을 P1-2 이전처럼 즉시 Dataset.data에 반영할 때의 비용
                   (dataset.data = model.get_dataframe(), 매 편집마다 강제되던 경로).
    """
    import numpy as np
    import pandas as pd

    from nuristat.core.dataset import Dataset
    from nuristat.ui.data_view import DataView

    _ensure_qapp()
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        rng.standard_normal((n_rows, n_cols)),
        columns=[f"var{i}" for i in range(n_cols)],
    )
    dataset = Dataset(data=df.copy())
    view = DataView()
    view.set_dataset(dataset)

    start = time.perf_counter()
    view._model.setData(view._model.index(n_rows // 2, 25), 3.14)
    deferred_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    dataset.data = view._model.get_dataframe()
    eager_sync_ms = (time.perf_counter() - start) * 1000

    return {"deferred_ms": deferred_ms, "eager_sync_ms": eager_sync_ms}


def undo_memory_mb() -> float:
    """100회 편집 후 undo 스택이 차지하는 DataFrame 메모리(MB)."""
    _ensure_qapp()
    model = _make_grid_model(n_rows=10_000, n_cols=50)

    for i in range(100):
        index = model.index(i, i % 50)
        model.setData(index, float(i))

    # P1-1: undo 항목은 ("cell", row, col, old, new) 델타 또는
    # ("full", snapshot) 전체 스냅샷 — full 항목의 DataFrame만 집계한다.
    total_bytes = sum(
        item[1][0].memory_usage(deep=True).sum()
        for item in model._undo_stack
        if item[0] == "full"
    )
    return total_bytes / (1024 * 1024)


def get_dataframe_ms() -> dict[str, float]:
    """get_dataframe() 캐시 미스(dirty) vs 캐시 히트 소요시간(ms)."""
    _ensure_qapp()
    model = _make_grid_model()

    model._invalidate_df_cache()
    start = time.perf_counter()
    model.get_dataframe()
    dirty_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    model.get_dataframe()
    cached_ms = (time.perf_counter() - start) * 1000

    return {"dirty_ms": dirty_ms, "cached_ms": cached_ms}


def csv_load_s(n_rows: int = 500_000, n_cols: int = 20) -> dict[str, float]:
    """대용량 CSV(utf-8 / cp949) 로딩 시간(초)."""
    import numpy as np
    import pandas as pd

    from nuristat.io.csv_reader import read_csv

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.standard_normal((n_rows, n_cols)),
        columns=[f"col{i}" for i in range(n_cols)],
    )

    results: dict[str, float] = {}
    for encoding in ("utf-8", "cp949"):
        tmp_path = ROOT / f"_perf_bench_tmp_{encoding}.csv"
        df.to_csv(tmp_path, index=False, encoding=encoding)
        try:
            start = time.perf_counter()
            read_csv(str(tmp_path), encoding="auto")
            results[encoding] = time.perf_counter() - start
        finally:
            tmp_path.unlink(missing_ok=True)

    return results


def analysis_s(n_rows: int = 100_000) -> dict[str, float]:
    """t-test / ANOVA / 회귀 분석 실행 시간(초, 100k행)."""
    import numpy as np
    import pandas as pd

    from nuristat.core.dataset import Dataset

    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(n_rows),
            "x1": rng.standard_normal(n_rows),
            "x2": rng.standard_normal(n_rows),
            "group": rng.choice(["A", "B"], size=n_rows),
            "group3": rng.choice(["A", "B", "C"], size=n_rows),
        }
    )
    dataset = Dataset(data=df)

    results: dict[str, float] = {}

    from nuristat.analysis.ttests import run_analysis as run_ttest

    start = time.perf_counter()
    run_ttest(dataset, {"variables": {"dependent": "y", "group": "group"}})
    results["ttest"] = time.perf_counter() - start

    from nuristat.analysis.anova import run_analysis as run_anova

    start = time.perf_counter()
    run_anova(dataset, {"variables": {"dependent": "y", "factor": "group3"}})
    results["anova"] = time.perf_counter() - start

    from nuristat.analysis.regression import run_analysis as run_regression

    start = time.perf_counter()
    run_regression(dataset, {"variables": {"dependent": "y", "predictors": ["x1", "x2"]}})
    results["regression"] = time.perf_counter() - start

    return results


def run_all() -> dict:
    print("startup_import_s ...")
    startup = startup_import_s()

    print("edit_latency_ms ...")
    edit_latency = edit_latency_ms()

    print("dataview_edit_sync_ms ...")
    dataview_edit_sync = dataview_edit_sync_ms()

    print("undo_memory_mb ...")
    undo_memory = undo_memory_mb()

    print("get_dataframe_ms ...")
    get_df = get_dataframe_ms()

    print("csv_load_s ...")
    csv_load = csv_load_s()

    print("analysis_s ...")
    analysis = analysis_s()

    return {
        "startup_import_s": startup,
        "edit_latency_ms": edit_latency,
        "dataview_edit_sync_ms": dataview_edit_sync,
        "undo_memory_mb": undo_memory,
        "get_dataframe_ms": get_df,
        "csv_load_s": csv_load,
        "analysis_s": analysis,
    }


def main() -> int:
    results = run_all()
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out_path = ROOT / "docs" / "perf_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장됨: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

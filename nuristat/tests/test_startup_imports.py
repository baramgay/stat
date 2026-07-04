"""시작 시 무거운 의존성 지연 로딩 회귀 가드 (P0-5).

nuristat.app 임포트만으로 scipy/chardet 등 무거운 모듈이 즉시 로드되지
않아야 한다. 서브프로세스로 격리하여 sys.modules 오염 없이 검증한다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _run_check(code: str) -> str:
    import os

    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout.strip()


def test_app_import_does_not_load_chardet():
    code = (
        "import sys\n"
        "from nuristat.app import NuriStatApp\n"
        "print('chardet' in sys.modules)\n"
    )
    assert _run_check(code) == "False"


def test_app_import_does_not_load_scipy():
    code = (
        "import sys\n"
        "from nuristat.app import NuriStatApp\n"
        "print('scipy' in sys.modules)\n"
    )
    assert _run_check(code) == "False"


def test_analysis_package_import_does_not_load_scipy():
    code = (
        "import sys\n"
        "import nuristat.analysis\n"
        "print('scipy' in sys.modules)\n"
    )
    assert _run_check(code) == "False"


def test_analysis_lazy_symbol_triggers_scipy_load():
    code = (
        "import sys\n"
        "import nuristat.analysis as a\n"
        "a.check_normality\n"
        "print('scipy' in sys.modules)\n"
    )
    assert _run_check(code) == "True"

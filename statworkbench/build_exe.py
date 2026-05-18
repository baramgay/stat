"""Build script for StatWorkbench executable.

PyInstaller를 사용하여 Windows용 실행 파일을 생성합니다.
"""

import subprocess
import sys
import os


def build_executable():
    """Build StatWorkbench executable using PyInstaller."""
    
    # 프로젝트 루트
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")
    
    # PyInstaller 명령 구성
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "StatWorkbench",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        # 아이콘 (기본)
        "--icon", "NONE",
        # 숨김 파일 임포트
        "--hidden-import", "pandas",
        "--hidden-import", "numpy",
        "--hidden-import", "scipy",
        "--hidden-import", "statsmodels",
        "--hidden-import", "openpyxl",
        "--hidden-import", "pydantic",
        "--hidden-import", "tabulate",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "PySide6.QtGui",
        "--hidden-import", "PySide6.QtWidgets",
        # 데이터 파일 포함
        "--add-data", f"{src_dir}/statworkbench/ui/theme.py:statworkbench/ui",
        "--add-data", f"{src_dir}/statworkbench/ui/icons.py:statworkbench/ui",
        # 메인 스크립트
        os.path.join(src_dir, "statworkbench", "main.py"),
    ]
    
    print("=" * 60)
    print("StatWorkbench 실행 파일 빌드 시작")
    print("=" * 60)
    print(f"명령: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("빌드 완료!")
        print("=" * 60)
        print(f"실행 파일 위치: {project_root}/dist/StatWorkbench.exe")
        print()
        print("실행 방법:")
        print("  Windows: dist/StatWorkbench.exe")
        print("  WSL:     wine dist/StatWorkbench.exe")
    else:
        print()
        print("=" * 60)
        print("빌드 실패!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    build_executable()

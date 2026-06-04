"""Build script for NuriStat executable.

PyInstaller를 사용하여 Windows용 실행 파일을 생성합니다.
"""

import subprocess
import sys
import os


def build_executable():
    """Build NuriStat executable using PyInstaller."""
    
    # 프로젝트 루트
    project_root = os.path.dirname(os.path.abspath(__file__))

    # NuriStat.spec를 사용한다 — spec이 collect_submodules로 전체 분석 모듈과
    # 외부 의존성(scipy/statsmodels/lifelines/pyreadstat/wordcloud/docx 등)을 누락 없이
    # 번들한다. 수동 --hidden-import 나열 방식은 레지스트리 동적 import를 놓쳐 불완전했다.
    spec_path = os.path.join(project_root, "NuriStat.spec")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        spec_path,
        "--clean",
        "--noconfirm",
    ]
    
    print("=" * 60)
    print("NuriStat 실행 파일 빌드 시작")
    print("=" * 60)
    print(f"명령: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("빌드 완료!")
        print("=" * 60)
        print(f"실행 파일 위치: {project_root}/dist/NuriStat/NuriStat.exe")
        print()
        print("실행 방법:")
        print("  Windows: dist/NuriStat/NuriStat.exe")
        print("  배포:    dist/NuriStat 폴더 전체를 배포")
    else:
        print()
        print("=" * 60)
        print("빌드 실패!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    build_executable()

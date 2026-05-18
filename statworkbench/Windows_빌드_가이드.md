# StatWorkbench Windows 실행 파일 빌드 가이드

## 방법 1: Windows에서 직접 빌드 (권장)

### 1.1 Python 설치
- https://python.org 에서 Python 3.11+ 다운로드
- 설치 시 "Add Python to PATH" 체크

### 1.2 명령 프롬프트 (CMD)에서 실행

```cmd
cd C:\업무\통계패키지\statworkbench

# 의존성 설치
pip install PySide6 pandas numpy scipy statsmodels openpyxl pydantic tabulate pyinstaller

# 패키지 설치
pip install -e .

# 실행 파일 빌드
python -m PyInstaller --name StatWorkbench --onefile --windowed --clean src\statworkbench\main.py

# 빌드 완료 후
dist\StatWorkbench.exe
```

### 1.3 PowerShell에서 실행

```powershell
cd C:\업무\통계패키지\statworkbench

pip install PySide6 pandas numpy scipy statsmodels openpyxl pydantic tabulate pyinstaller
pip install -e .
python -m PyInstaller --name StatWorkbench --onefile --windowed --clean src\statworkbench\main.py
```

---

## 방법 2: WSL에서 빌드한 Linux 실행 파일 사용

WSL에서 이미 빌드된 실행 파일:
- 위치: `/mnt/c/업무/통계패키지/statworkbench/dist/StatWorkbench`
- 크기: 224MB
- 이 파일은 WSL(Linux)에서만 실행 가능

```bash
cd /mnt/c/업무/통계패키지/statworkbench
./dist/StatWorkbench
```

---

## 방법 3: Python 소스로 직접 실행 (개발/테스트용)

```cmd
cd C:\업무\통계패키지\statworkbench
pip install -e ".[dev]"
python -m statworkbench.main
```

---

## 빌드 옵션 설명

| 옵션 | 설명 |
|------|------|
| `--name StatWorkbench` | 실행 파일 이름 |
| `--onefile` | 단일 실행 파일로 패키징 |
| `--windowed` | 콘솔 창 없이 GUI 모드로 실행 |
| `--clean` | 이전 빌드 캐시 삭제 |
| `--icon ICON.ico` | 아이콘 설정 (선택) |

---

## 출력 위치

빌드 완료 후 실행 파일 위치:
```
C:\업무\통계패키지\statworkbench\dist\StatWorkbench.exe
```

---

## 주의사항

1. **Windows용 .exe는 Windows에서만 빌드 가능**
   - WSL은 Linux 실행 파일만 생성
   - Cross-compilation은 PyInstaller에서 지원하지 않음

2. **바이러스 백신 오탐지**
   - PyInstaller로 빌드한 실행 파일이 일부 백신에서 오탐지될 수 있음
   - `--onefile` 대신 `--onedir` 사용하면 감소

3. **파일 크기**
   - `--onefile` 실행 파일: 약 200-300MB
   - `--onedir` 폴더: 약 300-500MB

4. **의존성**
   - 실행 파일에 모든 Python 라이브러리가 포함됨
   - 별도 Python 설치 불필요

# -*- mode: python ; coding: utf-8 -*-
import sys
import os

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# statworkbench 전체 서브모듈을 자동 수집 — 레지스트리가 분석 모듈을 동적 import하므로
# 수동 나열 대신 collect_submodules로 누락 없이 번들한다.
_sw_modules = collect_submodules('statworkbench')
_ext_modules = [
    'scipy.stats', 'scipy.special', 'scipy.linalg', 'scipy.optimize',
    'statsmodels.api', 'statsmodels.formula.api', 'statsmodels.stats.api',
    'statsmodels.graphics.api',
    'sklearn.decomposition', 'sklearn.cluster', 'sklearn.discriminant_analysis',
    'sklearn.linear_model', 'sklearn.preprocessing', 'sklearn.metrics',
    'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_agg',
    'matplotlib', 'seaborn',
    'lifelines', 'lifelines.statistics', 'lifelines.plotting',
    'openpyxl', 'chardet', 'pyarrow', 'pydantic', 'tabulate',
    'pyreadstat', 'wordcloud', 'docx',
    'pandas', 'numpy',
    'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
    'PySide6.QtCharts', 'PySide6.QtPrintSupport', 'PySide6.QtSvg',
]

a = Analysis(
    ['src/statworkbench/main.py'],
    pathex=['.', 'src'],
    binaries=[],
    datas=[
        ('src/statworkbench/resources', 'statworkbench/resources'),
    ],
    hiddenimports=_sw_modules + _ext_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 다른 GUI 툴킷
        'tkinter', 'wx', 'PyQt5', 'PyQt6',
        # 미사용 딥러닝/ML 스택 (로컬 환경에서 전이적으로 끌려오는 대용량 — statworkbench 미사용)
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'tokenizers', 'huggingface_hub', 'hf_xet',
        'onnxruntime', 'tensorflow', 'keras', 'jax', 'jaxlib',
        'sentence_transformers', 'datasets', 'accelerate', 'safetensors',
        'numba', 'llvmlite',
        # 미사용 DB 드라이버
        'psycopg2', 'psycopg2_binary',
        # 개발/노트북 도구
        'IPython', 'jupyter', 'jupyterlab', 'notebook', 'ipykernel',
        'pytest', 'sphinx', 'Pythonwin', 'win32com',
        # 미사용 Qt 모듈 (대용량 — 데스크톱 통계 앱은 QtWidgets/Charts/Svg/Print만 사용)
        'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick',
        'PySide6.QtQuick', 'PySide6.QtQuick3D', 'PySide6.QtQuickWidgets',
        'PySide6.QtQml', 'PySide6.QtQmlModels',
        'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
        'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DExtras',
        'PySide6.Qt3DInput', 'PySide6.Qt3DAnimation', 'PySide6.Qt3DLogic',
        'PySide6.QtBluetooth', 'PySide6.QtNfc', 'PySide6.QtPositioning',
        'PySide6.QtSensors', 'PySide6.QtSerialPort', 'PySide6.QtWebSockets',
        'PySide6.QtWebChannel', 'PySide6.QtWebView', 'PySide6.QtDesigner',
        'PySide6.QtHelp', 'PySide6.QtNetworkAuth', 'PySide6.QtRemoteObjects',
        'PySide6.QtScxml', 'PySide6.QtTextToSpeech', 'PySide6.QtDataVisualization',
        'PySide6.QtSpatialAudio', 'PySide6.QtStateMachine',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StatWorkbench',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StatWorkbench',
)

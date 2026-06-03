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
        'tkinter',
        'wx',
        'PyQt5',
        'PyQt6',
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

# PARC_Analyzer.spec
import os
import sys

# ← REPLACE THIS with the output of the python command above
QT_PLUGINS_PATH = "/Users/tudoriliescu/Licenta/RomanNumeralApp/.venv/lib/python3.12/site-packages/PyQt6/Qt6/plugins"
block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('qt.conf', '.'),
        (QT_PLUGINS_PATH + '/platforms',    'PyQt6/Qt6/plugins/platforms'),
        (QT_PLUGINS_PATH + '/styles',       'PyQt6/Qt6/plugins/styles'),
        (QT_PLUGINS_PATH + '/imageformats', 'PyQt6/Qt6/plugins/imageformats'),
        ('source/',           'source/'),
        ('experiments/',      'experiments/'),
        ('dataset/metadata/', 'dataset/metadata/'),
    ],
    hiddenimports=[
        'vamp',
        'librosa', 'librosa.core', 'librosa.feature', 'librosa.beat',
        'librosa.onset', 'librosa.util', 'librosa.filters',
        'soundfile', 'sounddevice',
        'scipy', 'scipy.signal', 'scipy.io', 'scipy.io.wavfile',
        'sklearn', 'sklearn.utils', 'sklearn.utils._cython_blas',
        'sklearn.neighbors._partition_nodes',
        'numba', 'llvmlite',
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        'torch', 'torch.nn', 'torch.nn.functional',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide2', 'PySide6'],    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='PARC_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window
    codesign_identity=None,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=False,
    name='PARC_Analyzer',
)

app = BUNDLE(
    coll,
    name='PARC_Analyzer.app',
    icon=None,
    bundle_identifier='com.parc.analyzer',
    info_plist={
    'NSHighResolutionCapable': True,
    'QT_QPA_PLATFORM_PLUGIN_PATH': '@executable_path/../Frameworks/PyQt6/Qt6/plugins/platforms',
    },
)
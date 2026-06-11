# hooks/rthook_qt.py
import os, sys

if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    plugin_path = os.path.abspath(
        os.path.join(exe_dir, '..', 'Frameworks', 'PyQt6', 'Qt6', 'plugins', 'platforms')
    )
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
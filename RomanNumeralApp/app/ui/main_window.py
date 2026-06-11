"""
app/ui/main_window.py - Root window.

New in this version
- FunctionLegend strip between timeline and controls
- PDF export wired (calls export_pdf.export_lead_sheet)
- ShortcutsHelp modal triggered by ? / F1
- Help button in top bar
- Status bar with model info
- Export buttons disabled until analysis is complete
- Song title in window title
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (QDragEnterEvent, QDropEvent,
                          QKeySequence, QShortcut)
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QStackedWidget, QStatusBar,
                               QTabWidget, QVBoxLayout, QWidget)

from app.ui.chord_display import ChordDisplayWidget, CircleOfFifths
from app.ui.controls import ControlsBar
from app.ui.detail_panel import DetailPanel, SettingsPanel
from app.ui.function_legend import FunctionLegend
from app.ui.shortcuts_help import ShortcutsHelp
from app.ui.theme import AMBER, BASE_QSS, BG0, BG1, BG2, BORDER, FG0, FG1, FG2
from app.ui.timeline import ChordTimeline
from app.ui.waveform import WaveformWidget
from app.ui.welcome import WelcomeOverlay

SUPPORTED_FORMATS = "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a *.aac)"


# thread bridge

class _PlayerBridge(QObject):
    position_changed = pyqtSignal(float)
    state_changed    = pyqtSignal(str)


# inference worker

class InferenceWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, object, float)
    error    = pyqtSignal(str)

    def __init__(self, path, model, domains, device, use_ss=False):
        super().__init__()
        self._path   = path
        self._model  = model
        self._domains = domains
        self._device  = device
        self._use_ss  = use_ss

    @pyqtSlot()
    def run(self):
        try:
            from app.inference import (build_chord_timeline,
                                        extract_features_vamp,
                                        sliding_window_inference)
            from app.player import WaveformData

            feats, times = extract_features_vamp(
                self._path, use_semitone_spectrum=self._use_ss,
                progress_callback=lambda p, m: self.progress.emit(p, m))
            self.progress.emit(55, "Running model inference…")
            preds, probs = sliding_window_inference(
                self._model, feats, self._device,
                progress_callback=lambda p, m: self.progress.emit(p, m))
            self.progress.emit(95, "Building chord timeline…")
            timeline = build_chord_timeline(preds, probs, self._domains, times)
            waveform = WaveformData.from_file(self._path)
            self.progress.emit(100, "Done.")
            self.finished.emit(timeline, waveform, waveform.duration)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# main window

class MainWindow(QMainWindow):

    def __init__(self, model, label_domains, label_sizes, device, args=None):
        super().__init__()
        self._model          = model
        self._label_domains  = label_domains
        self._label_sizes    = label_sizes
        self._device         = device
        self._timeline: List[dict] = []
        self._use_semitone   = getattr(args, "use_semitone_spectrum", False)
        self._volume: float  = 0.80
        self._current_path: Optional[str] = None

        self._bridge = _PlayerBridge()
        self._bridge.position_changed.connect(self._update_position)
        self._bridge.state_changed.connect(self._update_state)

        self.setWindowTitle("MAJOR TOM - Roman Numeral Chord Analysis")
        self.setMinimumSize(1080, 720)
        self.setAcceptDrops(True)
        self.setStyleSheet(BASE_QSS)

        self._build_player()
        self._build_ui()
        self._build_shortcuts()
        self._wire_signals()
        self._shortcuts_help = ShortcutsHelp(self)

    # player

    def _build_player(self):
        from app.player import AudioPlayer
        self.player = AudioPlayer()
        self.player.volume = self._volume
        self.player.on_position_changed = (
            lambda pos: self._bridge.position_changed.emit(pos))
        self.player.on_state_changed = (
            lambda st: self._bridge.state_changed.emit(st))

    # UI

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_topbar())
        root.addWidget(self._make_progress_area())

        self._stack = QStackedWidget()
        self._stack.addWidget(self._make_welcome())   # 0
        self._stack.addWidget(self._make_content())   # 1
        root.addWidget(self._stack, 1)

        root.addWidget(self._make_controls())

        self._build_status_bar()

    def _make_topbar(self):
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet(f"background:{BG1}; border-bottom:1px solid {BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 12, 0)

        logo = QLabel("MAJOR TOM")
        logo.setStyleSheet(
            f"color:{AMBER}; font-family:'IBM Plex Mono';"
            f" font-size:16px; font-weight:700; letter-spacing:4px;")

        self.lbl_song = QLabel("Drag an audio file here or click Import")
        self.lbl_song.setStyleSheet(f"color:{FG2}; font-size:12px;")

        btn_help = QPushButton("?")
        btn_help.setFixedSize(30, 30)
        btn_help.setToolTip("Keyboard shortcuts (F1)")
        btn_help.setStyleSheet(
            f"QPushButton {{ background:{BG2}; color:{FG1}; border:1px solid {BORDER};"
            f" border-radius:15px; font-family:'IBM Plex Mono'; font-size:14px; }}"
            f"QPushButton:hover {{ border-color:{AMBER}; color:{AMBER}; }}")
        btn_help.clicked.connect(self._show_shortcuts)

        self.btn_import = QPushButton("Import Audio…")
        self.btn_import.clicked.connect(self._on_import_click)

        lay.addWidget(logo)
        lay.addSpacing(16)
        lay.addWidget(self.lbl_song, 1)
        lay.addWidget(btn_help)
        lay.addSpacing(6)
        lay.addWidget(self.btn_import)
        return bar

    def _make_progress_area(self):
        wrap = QWidget()
        wrap.setFixedHeight(30)
        wrap.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(4)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet(
            f"color:{FG2}; font-size:11px; background:transparent;")
        self.lbl_status.setFixedHeight(22)

        lay.addWidget(self.progress_bar)
        lay.addWidget(self.lbl_status)
        wrap.hide()
        self._progress_wrap = wrap
        return wrap

    def _make_welcome(self):
        self.welcome = WelcomeOverlay()
        self.welcome.import_clicked.connect(self._on_import_click)
        self.welcome.file_dropped.connect(self._load_file)
        return self.welcome

    def _make_content(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Left: chord display + CoF
        left = QWidget()
        left.setFixedWidth(280)
        left.setStyleSheet(f"border-right:1px solid {BORDER};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self.chord_display = ChordDisplayWidget()
        self.cof_widget    = CircleOfFifths()
        self.cof_widget.setStyleSheet(
            f"background:{BG1}; border-top:1px solid {BORDER}; padding:6px;")

        ll.addWidget(self.chord_display, 1)
        ll.addWidget(self.cof_widget, 0, Qt.AlignmentFlag.AlignHCenter)

        # Center: waveform + timeline + legend
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        self.waveform = WaveformWidget()
        self.timeline = ChordTimeline()
        self.legend   = FunctionLegend()

        cl.addWidget(self.waveform, 2)
        cl.addWidget(self.timeline)
        cl.addWidget(self.legend)

        # Right: Details / Settings tabs
        self.detail   = DetailPanel()
        self.settings = SettingsPanel()
        self.settings.settings_changed.connect(self._on_settings_changed)
        self.detail.set_export_enabled(False)

        tabs = QTabWidget()
        tabs.setFixedWidth(220)
        tabs.setStyleSheet(f"""
            QTabWidget::pane  {{ border:none; background:{BG1}; }}
            QTabBar::tab      {{ background:{BG2}; color:{FG2};
                                 padding:6px 14px; border:none; font-size:11px; }}
            QTabBar::tab:selected {{ background:{BG1}; color:{AMBER}; }}
        """)
        tabs.addTab(self.detail,   "Details")
        tabs.addTab(self.settings, "Settings")

        lay.addWidget(left)
        lay.addWidget(center, 1)
        lay.addWidget(tabs)
        return w

    def _make_controls(self):
        self.controls = ControlsBar()
        self.controls.set_enabled_controls(False)
        # sync volume slider with player default
        self.controls.vol_slider.setValue(int(self._volume * 100))
        return self.controls

    def _build_status_bar(self):
        sb = QStatusBar()
        sb.setStyleSheet(
            f"QStatusBar {{ background:{BG1}; color:{FG2};"
            f" font-size:10px; border-top:1px solid {BORDER}; }}")
        self.setStatusBar(sb)
        self._sb_model = QLabel("No model loaded")
        self._sb_model.setStyleSheet(f"color:{FG2}; padding:0 8px;")
        sb.addPermanentWidget(self._sb_model)
        try:
            n = self._model.num_parameters
            self._sb_model.setText(
                f"Model: {n:,} params  ·  device: {self._device}")
        except Exception:
            pass

    # shortcuts

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Space"), self).activated.connect(self._on_play_pause)
        QShortcut(QKeySequence("Left"),  self).activated.connect(
            lambda: self.player.seek_relative(-5.0))
        QShortcut(QKeySequence("Right"), self).activated.connect(
            lambda: self.player.seek_relative(5.0))
        QShortcut(QKeySequence("r"),     self).activated.connect(self._on_restart)
        QShortcut(QKeySequence("?"),     self).activated.connect(self._show_shortcuts)
        QShortcut(QKeySequence("F1"),    self).activated.connect(self._show_shortcuts)
        QShortcut(QKeySequence("Ctrl+O"),self).activated.connect(self._on_import_click)

    def _wire_signals(self):
        self.controls.play_pause_clicked.connect(self._on_play_pause)
        self.controls.restart_clicked.connect(self._on_restart)
        self.controls.seek_changed.connect(self._on_seek_frac)
        self.controls.volume_changed.connect(self._on_volume)
        self.waveform.seek_requested.connect(self.player.seek)
        self.timeline.seek_requested.connect(self.player.seek)
        self.detail.export_json_clicked.connect(self._on_export_json)
        self.detail.export_csv_clicked.connect(self._on_export_csv)
        self.detail.export_pdf_clicked.connect(self._on_export_pdf)

    # drag & drop

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self._load_file(urls[0].toLocalFile())

    # import

    def _on_import_click(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", SUPPORTED_FORMATS)
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File not found", f"Cannot open:\n{path}")
            return

        self.player.stop()
        if hasattr(self, "chord_display"):
            self.chord_display.clear()
            self.waveform.clear()
            self.timeline.clear()
            self.detail.set_chord(None)
            self.detail.set_export_enabled(False)
        self._timeline = []
        self._current_path = path

        name = Path(path).name
        self.lbl_song.setText(name)
        self.setWindowTitle(f"MAJOR TOM - {name}")
        self.controls.set_enabled_controls(False)
        self._stack.setCurrentIndex(1)

        self.progress_bar.setValue(0)
        self._progress_wrap.show()
        self.lbl_status.setText("Loading audio…")

        try:
            self.player.load(path)
            self.controls.set_duration(self.player.duration)
        except Exception as exc:
            QMessageBox.warning(self, "Audio Error", str(exc))

        self._worker = InferenceWorker(
            path, self._model, self._label_domains,
            self._device, self._use_semitone)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    # worker

    @pyqtSlot(int, str)
    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    @pyqtSlot(list, object, float)
    def _on_worker_finished(self, timeline, waveform, duration):
        self._timeline = timeline
        self._progress_wrap.hide()
        self.waveform.set_waveform(waveform)
        self.waveform.set_timeline(timeline)
        self.timeline.set_timeline(timeline, duration)
        self.controls.set_enabled_controls(True)
        self.detail.set_export_enabled(True)
        if timeline:
            self.chord_display.set_chord(timeline[0])
            self.cof_widget.set_chord(timeline[0])
            self.detail.set_chord(timeline[0])
        n = len(timeline)
        self.statusBar().showMessage(
            f"Analysis complete - {n} chord events", 4000)

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str):
        self._progress_wrap.hide()
        QMessageBox.critical(self, "Analysis Error",
                             f"Inference failed:\n\n{msg[:1000]}")

    # playback (main thread via bridge)

    @pyqtSlot(float)
    def _update_position(self, pos: float):
        self.controls.set_position(pos)
        self.waveform.set_position(pos)
        self.timeline.set_position(pos)
        if self._timeline:
            from app.inference import get_current_chord
            chord = get_current_chord(pos, self._timeline)
            if chord:
                self.chord_display.set_chord(chord)
                self.cof_widget.set_chord(chord)
                self.detail.set_chord(chord)

    @pyqtSlot(str)
    def _update_state(self, state: str):
        self.controls.set_playing(state == "playing")
        if state == "finished":
            self.player.seek(0)

    # transport

    def _on_play_pause(self):
        self.player.toggle()

    def _on_restart(self):
        self.player.seek(0)

    def _on_seek_frac(self, frac: float):
        self.player.seek(frac * self.player.duration)

    def _on_volume(self, vol: float):
        self._volume = vol
        self.player.volume = vol

    # shortcuts help

    def _show_shortcuts(self):
        self._shortcuts_help.show_near(self)

    # settings

    @pyqtSlot(dict)
    def _on_settings_changed(self, cfg: dict):
        import torch
        new_device = torch.device(cfg["device"])
        if str(new_device) != str(self._device):
            self._device = new_device
            try:
                self._model.to(self._device)
            except Exception as e:
                QMessageBox.warning(self, "Device Error", str(e))
            try:
                self._sb_model.setText(
                    f"Model: {self._model.num_parameters:,} params"
                    f"  ·  device: {self._device}")
            except Exception:
                pass
        self._use_semitone = cfg["use_semitone_spectrum"]
        if self._current_path:
            reply = QMessageBox.question(
                self, "Re-analyze?",
                "Settings changed. Re-analyze the current file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._load_file(self._current_path)

    # export

    def _on_export_json(self):
        if not self._timeline:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", self._stem() + "_chords.json", "JSON (*.json)")
        if path:
            from app.inference import export_json
            export_json(self._timeline, path)
            self.statusBar().showMessage(f"Exported JSON -> {path}", 3000)

    def _on_export_csv(self):
        if not self._timeline:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", self._stem() + "_chords.csv", "CSV (*.csv)")
        if path:
            from app.inference import export_csv
            export_csv(self._timeline, path)
            self.statusBar().showMessage(f"Exported CSV -> {path}", 3000)

    def _on_export_pdf(self):
        if not self._timeline:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Lead Sheet PDF",
            self._stem() + "_lead_sheet.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            from app.export_pdf import export_lead_sheet
            bars = self.settings.pdf_bars_per_row()
            name = Path(self._current_path).stem if self._current_path else "MAJOR TOM"
            export_lead_sheet(self._timeline, path, title=name,
                              bars_per_row=bars)
            self.statusBar().showMessage(f"Exported PDF -> {path}", 3000)
            reply = QMessageBox.question(
                self, "Open PDF?", "Lead sheet saved. Open it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                import subprocess, sys
                if sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                elif sys.platform.startswith("linux"):
                    subprocess.Popen(["xdg-open", path])
                else:
                    os.startfile(path)
        except ImportError:
            QMessageBox.warning(
                self, "Missing Dependency",
                "PDF export requires reportlab.\n\nInstall with:\n  pip install reportlab")
        except Exception as exc:
            QMessageBox.critical(self, "PDF Export Error", str(exc))

    def _stem(self) -> str:
        if self._current_path:
            return Path(self._current_path).stem
        return "parc_export"

    # lifecycle

    def closeEvent(self, event):
        self.player.stop()
        if hasattr(self, "_thread") and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        event.accept()
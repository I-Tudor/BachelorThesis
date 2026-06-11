"""
app/ui/controls.py - Transport controls bar.
"""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QSlider, QStyle,
                               QToolButton, QWidget)

from app.ui.theme import AMBER, BG1, BG2, BORDER, FG0, FG1, FG2


def _fmt_time(secs: float) -> str:
    s = int(secs)
    return f"{s // 60}:{s % 60:02d}"


class ControlsBar(QWidget):
    """Transport: Play/Pause, Restart, seek slider, time label."""

    play_pause_clicked = pyqtSignal()
    restart_clicked    = pyqtSignal()
    seek_changed       = pyqtSignal(float)   # fraction [0..1]
    volume_changed     = pyqtSignal(float)   # [0..1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration    = 0.0
        self._user_seeking = False

        self.setStyleSheet(f"""
            QWidget {{ background: {BG1}; }}
            QLabel  {{ color: {FG1}; font-family: "IBM Plex Mono"; font-size: 12px; }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(12)

        # Restart
        self.btn_restart = QToolButton()
        self.btn_restart.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.btn_restart.setIconSize(QSize(18, 18))
        self.btn_restart.setFixedSize(34, 34)
        self.btn_restart.setStyleSheet(self._btn_css())
        self.btn_restart.clicked.connect(self.restart_clicked)

        # Play/Pause
        self.btn_play = QToolButton()
        self.btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_play.setIconSize(QSize(22, 22))
        self.btn_play.setFixedSize(42, 42)
        self.btn_play.setStyleSheet(self._btn_css(primary=True))
        self.btn_play.clicked.connect(self.play_pause_clicked)

        # Time
        self.lbl_time = QLabel("0:00 / 0:00")

        # Seek bar
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderPressed.connect(self._on_seek_press)
        self.seek_slider.sliderMoved.connect(self._on_seek_move)
        self.seek_slider.sliderReleased.connect(self._on_seek_release)

        # Volume
        self.lbl_vol = QLabel("🔊")
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setFixedWidth(80)
        self.vol_slider.valueChanged.connect(
            lambda v: self.volume_changed.emit(v / 100.0))

        for w in [self.btn_restart, self.btn_play,
                  self.lbl_time, self.seek_slider,
                  self.lbl_vol, self.vol_slider]:
            lay.addWidget(w)
        lay.setStretch(3, 1)  # seek_slider expands

    def _btn_css(self, primary: bool = False) -> str:
        if primary:
            return f"""
                QToolButton {{
                    background: {AMBER}; border: none; border-radius: 21px; color: #000;
                }}
                QToolButton:hover {{ background: #F8B85A; }}
                QToolButton:pressed {{ background: #D08A30; }}
            """
        return f"""
            QToolButton {{
                background: {BG2}; border: 1px solid {BORDER};
                border-radius: 17px; color: {FG0};
            }}
            QToolButton:hover {{ background: {BORDER}; }}
        """

    def set_duration(self, secs: float):
        self._duration = secs
        self._update_time_label(0)

    def set_position(self, secs: float):
        if self._user_seeking:
            return
        if self._duration > 0:
            val = int((secs / self._duration) * 10000)
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(val)
            self.seek_slider.blockSignals(False)
        self._update_time_label(secs)

    def set_playing(self, playing: bool):
        icon = (QStyle.StandardPixmap.SP_MediaPause if playing
                else QStyle.StandardPixmap.SP_MediaPlay)
        self.btn_play.setIcon(self.style().standardIcon(icon))

    def set_enabled_controls(self, enabled: bool):
        for w in [self.btn_play, self.btn_restart, self.seek_slider]:
            w.setEnabled(enabled)


    def _update_time_label(self, pos: float):
        self.lbl_time.setText(f"{_fmt_time(pos)} / {_fmt_time(self._duration)}")

    def _on_seek_press(self):
        self._user_seeking = True

    def _on_seek_move(self, val: int):
        if self._duration > 0:
            frac = val / 10000.0
            self._update_time_label(frac * self._duration)

    def _on_seek_release(self):
        self._user_seeking = False
        frac = self.seek_slider.value() / 10000.0
        self.seek_changed.emit(frac)
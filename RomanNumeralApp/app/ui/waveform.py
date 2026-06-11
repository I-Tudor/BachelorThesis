"""
app/ui/waveform.py - Waveform display with chord-coloring and playhead.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QColor, QLinearGradient, QMouseEvent, QPainter,
                          QPainterPath, QPen)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.ui.theme import AMBER, BG0, BG2, BORDER, FG2, FUNCTION_COLORS


class WaveformWidget(QWidget):
    """
    Draws a peak waveform, color-coded by chord harmonic function.
    Emits seek_requested(float) when the user clicks.
    """
    seek_requested = pyqtSignal(float)  # seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self._peaks_pos: Optional[object] = None
        self._peaks_neg: Optional[object] = None
        self._duration: float   = 0.0
        self._position: float   = 0.0
        self._timeline: List[dict] = []
        self._n: int = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(60)
        self.setMaximumHeight(90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_waveform(self, waveform):
        """Accept a WaveformData object."""
        self._peaks_pos = waveform.peaks_pos
        self._peaks_neg = waveform.peaks_neg
        self._duration  = waveform.duration
        self._n         = waveform.n_buckets
        self.update()

    def set_timeline(self, timeline: List[dict]):
        self._timeline = timeline
        self.update()

    def set_position(self, seconds: float):
        self._position = seconds
        self.update()

    def clear(self):
        self._peaks_pos = None
        self._peaks_neg = None
        self._duration  = 0.0
        self._position  = 0.0
        self._timeline  = []
        self.update()

    # interaction

    def mousePressEvent(self, event: QMouseEvent):
        if self._duration > 0:
            frac = event.position().x() / self.width()
            self.seek_requested.emit(frac * self._duration)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton and self._duration > 0:
            frac = max(0, min(1, event.position().x() / self.width()))
            self.seek_requested.emit(frac * self._duration)

    # painting

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        cy = H / 2

        # Background
        p.fillRect(QRectF(0, 0, W, H), QColor(BG0))

        if self._peaks_pos is None:
            p.setPen(QColor(FG2))
            p.drawText(QRectF(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                       "Import an audio file to begin")
            p.end()
            return

        n = len(self._peaks_pos)
        step = W / n

        # Build chord-function color map per bucket
        bucket_colors = self._build_bucket_colors(n)

        # Draw waveform bars
        for i in range(n):
            x     = i * step
            hp    = float(self._peaks_pos[i]) * (cy - 2)
            hn    = float(-self._peaks_neg[i]) * (cy - 2)
            color = bucket_colors[i]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            # positive bar
            p.drawRect(QRectF(x, cy - hp, max(1, step - 0.3), hp))
            # negative bar
            p.drawRect(QRectF(x, cy, max(1, step - 0.3), hn))

        # Played portion dark overlay
        if self._duration > 0:
            played_frac = self._position / self._duration
            played_x    = played_frac * W
            # Unplayed region dim
            dim = QColor(0, 0, 0, 100)
            p.fillRect(QRectF(played_x, 0, W - played_x, H), dim)

        # Playhead line
        if self._duration > 0:
            px = (self._position / self._duration) * W
            pen = QPen(QColor(AMBER))
            pen.setWidthF(1.5)
            p.setPen(pen)
            p.drawLine(QPointF(px, 0), QPointF(px, H))

        p.end()

    def _build_bucket_colors(self, n: int):
        """Map each waveform bucket index to a QColor based on chord function."""
        colors = []
        default = QColor(BG2)
        default.setAlpha(180)

        if not self._timeline or self._duration <= 0:
            dim = QColor(BORDER)
            dim.setAlpha(160)
            return [dim] * n

        # Build lookup: time -> color
        tl = self._timeline
        tl_idx = 0

        for i in range(n):
            t = (i / n) * self._duration
            while tl_idx + 1 < len(tl) and tl[tl_idx + 1]["time"] <= t:
                tl_idx += 1
            chord = tl[tl_idx]
            func  = chord.get("function", "other")
            accent_s = FUNCTION_COLORS.get(func, ("#555B70", "#000"))[0]
            col = QColor(accent_s)
            col.setAlpha(140)
            colors.append(col)

        return colors
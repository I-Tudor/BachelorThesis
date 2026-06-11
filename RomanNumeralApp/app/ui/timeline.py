"""
app/ui/timeline.py - Horizontal chord-change timeline.

Shows all chord events as colored blocks. Scrubs and auto-scrolls
to keep the playhead visible.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QMouseEvent,
                          QPainter, QPen)
from PyQt6.QtWidgets import QAbstractScrollArea, QSizePolicy

from app.ui.theme import (AMBER, BG0, BG1, BG2, BORDER, FG0, FG1, FG2,
                           FUNCTION_COLORS)

PIXELS_PER_SECOND = 80.0   # zoom level
LANE_H = 48


class ChordTimeline(QAbstractScrollArea):
    """
    Scrollable timeline showing chord blocks.
    User can click to seek.
    """
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timeline: List[dict] = []
        self._duration: float      = 0.0
        self._position: float      = 0.0
        self._total_w: int         = 0
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(LANE_H + 28)
        self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)

    def set_timeline(self, timeline: List[dict], duration: float):
        self._timeline = timeline
        self._duration = duration
        self._total_w  = max(self.viewport().width(),
                             int(duration * PIXELS_PER_SECOND) + 40)
        self.horizontalScrollBar().setRange(0, self._total_w - self.viewport().width())
        self.viewport().update()

    def set_position(self, seconds: float):
        self._position = seconds
        self._auto_scroll()
        self.viewport().update()

    def clear(self):
        self._timeline = []
        self._duration = 0.0
        self._position = 0.0
        self.viewport().update()

    # scroll

    def _auto_scroll(self):
        px = self._position * PIXELS_PER_SECOND
        vw = self.viewport().width()
        sb = self.horizontalScrollBar()
        margin = vw * 0.35
        if px - sb.value() > vw - margin:
            sb.setValue(int(px - margin))
        elif px - sb.value() < margin and sb.value() > 0:
            sb.setValue(max(0, int(px - margin)))

    def _offset(self) -> int:
        return self.horizontalScrollBar().value()

    def _x_to_time(self, screen_x: float) -> float:
        return (screen_x + self._offset()) / PIXELS_PER_SECOND

    # interaction

    def mousePressEvent(self, event: QMouseEvent):
        t = self._x_to_time(event.position().x())
        self.seek_requested.emit(max(0, t))

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            t = self._x_to_time(event.position().x())
            self.seek_requested.emit(max(0, t))

    # painting

    def paintEvent(self, _):
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W = self.viewport().width()
        H = self.viewport().height()
        off = self._offset()

        p.fillRect(QRectF(0, 0, W, H), QColor(BG0))

        if not self._timeline:
            p.setPen(QColor(FG2))
            p.drawText(QRectF(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                       "No chord data loaded")
            p.end()
            return

        # Time ruler
        ruler_h = 20
        p.setPen(QColor(FG2))
        font_r = QFont("IBM Plex Mono", 8)
        p.setFont(font_r)
        tick_interval = 5.0  # seconds
        t = 0.0
        while t <= self._duration + tick_interval:
            rx = t * PIXELS_PER_SECOND - off
            if -40 <= rx <= W + 40:
                p.setPen(QColor(BORDER))
                p.drawLine(QPointF(rx, ruler_h - 6), QPointF(rx, ruler_h))
                p.setPen(QColor(FG2))
                mins = int(t) // 60
                secs = int(t) % 60
                label = f"{mins}:{secs:02d}"
                p.drawText(QRectF(rx - 18, 2, 36, 14), Qt.AlignmentFlag.AlignCenter, label)
            t += tick_interval

        # Chord blocks
        block_y = ruler_h
        block_h = H - ruler_h - 2
        font_ch = QFont("IBM Plex Mono", 11, QFont.Weight.Bold)
        font_sm = QFont("IBM Plex Mono", 8)

        for ev in self._timeline:
            bx = ev["time"] * PIXELS_PER_SECOND - off
            bw = max(2.0, (ev["end"] - ev["time"]) * PIXELS_PER_SECOND - 1)

            if bx + bw < 0 or bx > W:
                continue

            func    = ev.get("function", "other")
            accent  = FUNCTION_COLORS.get(func, ("#555B70", "#1A1A1A"))[0]
            dim     = FUNCTION_COLORS.get(func, ("#555B70", "#1A1A1A"))[1]

            # Block fill
            col_fill = QColor(dim)
            col_fill.setAlpha(200)
            p.setBrush(col_fill)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(bx, block_y, bw, block_h), 3, 3)

            # Top accent line
            pen_top = QPen(QColor(accent))
            pen_top.setWidthF(2)
            p.setPen(pen_top)
            p.drawLine(QPointF(bx, block_y), QPointF(bx + bw, block_y))

            # Label
            if bw > 16:
                label = ev.get("chord_label", "?")
                p.setFont(font_ch)
                fm = QFontMetrics(font_ch)
                if fm.horizontalAdvance(label) > bw - 6:
                    p.setFont(font_sm)
                p.setPen(QColor(accent))
                p.drawText(QRectF(bx + 3, block_y + 4, bw - 6, block_h - 8),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           label)

        # Playhead
        px = self._position * PIXELS_PER_SECOND - off
        if 0 <= px <= W:
            pen_ph = QPen(QColor(AMBER))
            pen_ph.setWidthF(2.0)
            p.setPen(pen_ph)
            p.drawLine(QPointF(px, 0), QPointF(px, H))
            # Diamond head
            p.setBrush(QColor(AMBER))
            p.setPen(Qt.PenStyle.NoPen)
            pts = [QPointF(px, 2), QPointF(px + 5, 9),
                   QPointF(px, 16), QPointF(px - 5, 9)]
            from PyQt6.QtGui import QPolygonF
            p.drawPolygon(QPolygonF(pts))

        p.end()
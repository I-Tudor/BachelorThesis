"""
app/ui/function_legend.py - Compact harmonic function colour legend.

Displayed inline below the timeline. Shows tonic / subdominant / dominant
colour swatches with labels. Clicking a swatch dims/highlights chords of
that function (future: filter).
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.ui.theme import AMBER, BG0, BG1, BORDER, FG2, FUNCTION_COLORS


class FunctionLegend(QWidget):
    """One-line legend strip."""

    _ITEMS = [
        ("Tonic",       "tonic",       "I  III  VI"),
        ("Subdominant", "subdominant", "II  IV"),
        ("Dominant",    "dominant",    "V  VII"),
        ("Other",       "other",       "secondary"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(26)
        self.setStyleSheet(f"background:{BG0};")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        p.fillRect(QRectF(0, 0, W, H), QColor(BG0))

        # Top separator
        p.setPen(QColor(BORDER))
        p.drawLine(0, 0, W, 0)

        slot_w = W / len(self._ITEMS)
        font_lbl = QFont("IBM Plex Sans", 9)
        font_deg = QFont("IBM Plex Mono", 8)

        for i, (label, func, degrees) in enumerate(self._ITEMS):
            cx = i * slot_w + slot_w / 2
            accent, _ = FUNCTION_COLORS.get(func, (FG2, BG1))

            # Colour swatch
            swatch_w = 8
            swatch_x = cx - slot_w / 2 + 10
            p.setBrush(QColor(accent))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(swatch_x, H / 2 - 4, swatch_w, 8), 2, 2)

            # Label
            p.setFont(font_lbl)
            p.setPen(QColor(accent))
            p.drawText(
                QRectF(swatch_x + swatch_w + 4, 0, slot_w - swatch_w - 20, H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            # Degrees hint
            p.setFont(font_deg)
            p.setPen(QColor(FG2))
            p.drawText(
                QRectF(cx + 4, 0, slot_w / 2, H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                degrees,
            )

            # Divider
            if i > 0:
                p.setPen(QColor(BORDER))
                p.drawLine(int(i * slot_w), 4, int(i * slot_w), H - 4)

        p.end()
"""
app/ui/shortcuts_help.py - Modal keyboard shortcuts reference panel.

Press ? or F1 to show/hide. Dismisses on Escape, click-outside, or the
same shortcut key.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import (QColor, QFont, QKeySequence, QPainter,
                          QPainterPath, QShortcut)
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QWidget

from app.ui.theme import AMBER, BG0, BG1, BG2, BORDER, FG0, FG1, FG2


# shortcut table

SHORTCUTS = [
    ("Transport",   None),
    ("Space",       "Play / Pause"),
    ("R",           "Restart from beginning"),
    ("← / ->",       "Seek −5 s / +5 s"),
    ("",            ""),
    ("File",        None),
    ("Ctrl+O",      "Import audio file"),
    ("",            ""),
    ("View",        None),
    ("? / F1",      "Toggle this help panel"),
    ("Escape",      "Close overlays"),
]


# widget

class ShortcutsHelp(QDialog):
    """Semi-transparent modal overlay listing keyboard shortcuts."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint |
                                  Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(False)
        self.setFixedWidth(320)

        QShortcut(QKeySequence("Escape"), self).activated.connect(self.close)
        QShortcut(QKeySequence("?"),      self).activated.connect(self.close)
        QShortcut(QKeySequence("F1"),     self).activated.connect(self.close)

    def show_near(self, parent: QWidget):
        """Position and show relative to parent window."""
        if self.isVisible():
            self.close()
            return
        self._resize_to_content()
        if parent:
            gp = parent.mapToGlobal(parent.rect().center())
            self.move(gp.x() - self.width() // 2,
                      gp.y() - self.height() // 2)
        self.show()
        self.raise_()

    def _resize_to_content(self):
        rows = len(SHORTCUTS)
        self.setFixedHeight(50 + rows * 26 + 24)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        # Frosted panel
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, W, H), 12, 12)

        bg = QColor(BG1)
        bg.setAlpha(245)
        p.fillPath(path, bg)

        # Border
        p.setPen(QColor(BORDER))
        p.drawPath(path)

        # Title
        p.setFont(QFont("IBM Plex Mono", 11, QFont.Weight.Bold))
        p.setPen(QColor(AMBER))
        p.drawText(QRectF(0, 14, W, 24), Qt.AlignmentFlag.AlignHCenter, "Keyboard Shortcuts")

        # Rows
        y = 48
        for key, desc in SHORTCUTS:
            if desc is None:
                # Section header
                p.setFont(QFont("IBM Plex Mono", 8))
                p.setPen(QColor(FG2))
                p.drawText(QRectF(16, y, W - 32, 20),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            key.upper())
                # separator line
                p.setPen(QColor(BORDER))
                p.drawLine(int(W * 0.25), int(y + 10), int(W - 16), int(y + 10))
                y += 24
                continue

            if not key and not desc:
                y += 8
                continue

            # Key badge
            if key:
                p.setFont(QFont("IBM Plex Mono", 9, QFont.Weight.Medium))
                p.setPen(QColor(BG0))
                badge_w = max(48, len(key) * 8 + 16)
                badge_rect = QRectF(16, y + 2, badge_w, 18)
                p.setBrush(QColor(AMBER))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(badge_rect, 4, 4)
                p.setPen(QColor(BG0))
                p.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, key)

            # Description
            p.setFont(QFont("IBM Plex Sans", 10))
            p.setPen(QColor(FG0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawText(QRectF(80, y, W - 96, 22),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       desc)
            y += 26

        p.end()

    def mousePressEvent(self, event):
        # Click anywhere to close
        self.close()
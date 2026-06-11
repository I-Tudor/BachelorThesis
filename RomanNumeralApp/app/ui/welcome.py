"""
app/ui/welcome.py - Drag-and-drop welcome screen.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QColor, QDragEnterEvent, QDropEvent,
                          QFont, QLinearGradient, QPainter, QPainterPath)
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from app.ui.theme import AMBER, BG0, BG1, BG2, BORDER, FG0, FG1, FG2


class WelcomeOverlay(QWidget):
    """Full-area welcome / drag-target shown before any file is loaded."""

    import_clicked = pyqtSignal()

    # Re-broadcast drag events so the parent window can handle them
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_over = False
        self.setAcceptDrops(True)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(18)

        # Push content to center vertically
        lay.addStretch(2)

        btn = QPushButton("Import Audio File")
        btn.setFixedWidth(200)
        btn.setFixedHeight(42)
        btn.setObjectName("primary")
        btn.clicked.connect(self.import_clicked)
        lay.addWidget(btn, 0, Qt.AlignmentFlag.AlignHCenter)

        hint = _label("or drag & drop an MP3, WAV, FLAC, OGG, M4A", FG2, 12)
        lay.addWidget(hint, 0, Qt.AlignmentFlag.AlignHCenter)

        lay.addStretch(1)

        shortcuts = _label(
            "Space  Play/Pause     ←/->  Seek 5s     R  Restart",
            FG2, 10,
        )
        lay.addWidget(shortcuts, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(24)

    # drag & drop

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            self._drag_over = True
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, _):
        self._drag_over = False
        self.update()

    def dropEvent(self, event: QDropEvent):
        self._drag_over = False
        self.update()
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())

    # painting

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        # Background
        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0, QColor(BG1))
        grad.setColorAt(1, QColor(BG0))
        p.fillRect(QRectF(0, 0, W, H), grad)

        # Dashed drop-zone border when dragging
        if self._drag_over:
            from PyQt6.QtCore import Qt as _Qt
            from PyQt6.QtGui import QPen
            pen = QPen(QColor(AMBER))
            pen.setWidth(2)
            pen.setStyle(_Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(_Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(20, 20, W - 40, H - 40), 12, 12)

        # Big watermark
        font = QFont("IBM Plex Mono", 96, QFont.Weight.Bold)
        p.setFont(font)
        col = QColor(AMBER)
        col.setAlphaF(0.04)
        p.setPen(col)
        p.drawText(QRectF(0, 0, W, H),
                   Qt.AlignmentFlag.AlignCenter, "MAJOR TOM")

        # Subtitle
        font2 = QFont("IBM Plex Sans", 13)
        p.setFont(font2)
        p.setPen(QColor(FG2))
        p.drawText(
            QRectF(0, H / 2 - 56, W, 30),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Roman Numeral Chord Analysis",
        )

        p.end()


def _label(text: str, color: str, size: int):
    from PyQt6.QtWidgets import QLabel
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-size:{size}px;"
        f" font-family:'IBM Plex Mono'; background:transparent;")
    return lbl